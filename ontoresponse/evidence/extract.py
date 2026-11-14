"""Extract shared probe artifacts and real frozen-expert observations."""

import gc
import time
from pathlib import Path

import torch
import yaml

from ..common import budget, file_info, read_jsonl, unique, write_jsonl
from ..data.audio import quality, read_audio
from ..data.prepare import shard
from ..experts.base import AudioInput, FrozenExpert
from ..resources import resource_path
from ..transforms.probes import apply_probe
from .cache import Cache


def _batched_measurements(cfg, records, expert, cache, probes, policy):
    """Queue cache misses across recordings/views while preserving every correspondence."""
    size = policy["batch_size"]
    jobs = []
    for record in records:
        if not record.get("views"):
            continue
        measurements = []
        record["experts"][expert.spec["id"]] = dict(
            status="complete",
            identity=expert.identity,
            measurements=measurements,
            inference_policy=policy,
        )
        for view in record["views"]:
            entry = dict(status="missing", probe_id=view["probe_id"], cache_key=None)
            measurements.append(entry)
            if view["status"] != "valid":
                continue
            spec = dict(
                stage="expert",
                artifact_key=view["artifact_key"],
                expert=expert.identity,
                cache_revision=cfg["runtime"].get("cache_revision", 2),
                inference_policy=policy,
                schema=dict(
                    dims="runtime_verified",
                    score="spoof_minus_bonafide",
                    views=[p["id"] for p in probes],
                ),
            )
            key = cache.key(spec)
            try:
                hit = cache.get(key, spec)
                if hit is not None:
                    entry.update(status="valid", cache_key=key)
                else:
                    jobs.append(
                        (
                            record["id"] + "::" + view["probe_id"],
                            view["artifact_key"],
                            key,
                            spec,
                            entry,
                        )
                    )
            except Exception as error:
                entry.update(status="failed", error=str(error))
    for start in range(0, len(jobs), size):
        chunk = jobs[start : start + size]
        try:
            inputs = [
                AudioInput(item[0], cache.get(item[1])[0]["waveform"]) for item in chunk
            ]
            prediction = expert.predict_utterances(inputs, batch_size=size)
            if prediction.ids != [item.id for item in inputs]:
                raise ValueError("Batched prediction changed request order")
            for i, (_, _, key, spec, entry) in enumerate(chunk):
                meta = dict(prediction.metadata)
                meta["utterance_measurements"] = [meta["utterance_measurements"][i]]
                cache.put(
                    key,
                    spec,
                    dict(
                        embedding=prediction.embeddings[i].numpy(),
                        margin=prediction.spoof_margins[i : i + 1].numpy(),
                    ),
                    meta,
                )
                entry.update(status="valid", cache_key=key)
        except Exception as error:
            for *_, entry in chunk:
                entry.update(status="failed", cache_key=None, error=str(error))


def extract(cfg, manifest, max_origins=24, shard_index=0, shard_count=1, output=None):
    rows = read_jsonl(manifest)
    unique(rows)
    allowed = {"id", "audio", "origin_id", "source_group"}
    if any(set(r) != allowed for r in rows):
        raise ValueError(
            "Inference manifest contains audit/label fields or lacks grouping"
        )
    budget(cfg["runtime"]["budget"], max_origins)
    if len({r["origin_id"] for r in rows}) > max_origins:
        raise ValueError("Manifest exceeds explicit source budget")
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index")
    rows = [r for r in rows if shard(r["source_group"], shard_count) == shard_index]
    cache = Cache(cfg["cache"], cfg.get("pilot", {}).get("max_cache_bytes"))
    probes = yaml.safe_load(resource_path("configs/probes.yaml").read_text())["probes"]
    ffmpeg = (
        __import__("subprocess")
        .run(["ffmpeg", "-version"], capture_output=True, text=True)
        .stdout.splitlines()[0]
    )
    records = []
    view_records = {}
    for r in rows:
        try:
            x = read_audio(r["audio"])
            view_records[r["id"]] = []
            for p in [
                {"id": "original", "family": "identity", "operation": "identity"}
            ] + probes:
                spec = {
                    "stage": "audio",
                    "audio": file_info(r["audio"]),
                    "operation": p,
                    "cache_revision": cfg["runtime"].get("cache_revision", 2),
                    "ffmpeg_version": ffmpeg,
                    "seed_rule": "audio_hash_noise_shared_direction_v1",
                }
                key = cache.key(spec)
                try:
                    hit = cache.get(key, spec)
                    if hit is None:
                        y, meta = apply_probe(x, p)
                        hit = cache.put(key, spec, {"waveform": y}, meta)
                    view_records[r["id"]].append(
                        {
                            "probe_id": p["id"],
                            "family": p["family"],
                            "artifact_key": key,
                            "status": "valid",
                        }
                    )
                except Exception as e:
                    view_records[r["id"]].append(
                        {
                            "probe_id": p["id"],
                            "family": p["family"],
                            "artifact_key": None,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
            records.append(
                {
                    "id": r["id"],
                    "status": "pending",
                    "quality": quality(x),
                    "views": view_records[r["id"]],
                    "experts": {},
                }
            )
        except Exception as e:
            records.append(
                {"id": r["id"], "status": "failed", "error": str(e), "experts": {}}
            )
    for e in cfg["experts"]:
        expert = None
        previous_tf32 = torch.backends.cudnn.allow_tf32
        started = time.monotonic()
        try:
            policy = cfg["runtime"].get("expert_inference", {}).get(e["id"], {})
            batch_size = policy.get("batch_size", 1)
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("Positive expert inference batch size required")
            if "cudnn_allow_tf32" in policy:
                torch.backends.cudnn.allow_tf32 = policy["cudnn_allow_tf32"]
            expert = FrozenExpert(e, cfg["crop"], cfg["runtime"]["device"])
            if batch_size > 1:
                _batched_measurements(cfg, records, expert, cache, probes, policy)
                print(
                    f"extracted {e['id']} batch={batch_size} on shard {shard_index}/{shard_count}: "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
                continue
            for record in records:
                if record["id"] not in view_records:
                    continue
                measurements = []
                for v in record["views"]:
                    if v["status"] != "valid":
                        measurements.append(
                            {
                                "status": "missing",
                                "probe_id": v["probe_id"],
                                "cache_key": None,
                            }
                        )
                        continue
                    spec = {
                        "stage": "expert",
                        "artifact_key": v["artifact_key"],
                        "expert": expert.identity,
                        "cache_revision": cfg["runtime"].get("cache_revision", 2),
                        "schema": {
                            "dims": "runtime_verified",
                            "score": "spoof_minus_bonafide",
                            "views": [p["id"] for p in probes],
                        },
                    }
                    key = cache.key(spec)
                    try:
                        hit = cache.get(key, spec)
                        if hit is None:
                            x = cache.get(v["artifact_key"])[0]["waveform"]
                            prediction = expert.predict_utterances(
                                [AudioInput(record["id"], x)]
                            )
                            hit = cache.put(
                                key,
                                spec,
                                {
                                    "embedding": prediction.embeddings[0].numpy(),
                                    "margin": prediction.spoof_margins[:1].numpy(),
                                },
                                prediction.metadata,
                            )
                        measurements.append(
                            {
                                "status": "valid",
                                "probe_id": v["probe_id"],
                                "cache_key": key,
                            }
                        )
                    except Exception as err:
                        measurements.append(
                            {
                                "status": "failed",
                                "probe_id": v["probe_id"],
                                "cache_key": None,
                                "error": str(err),
                            }
                        )
                record["experts"][e["id"]] = {
                    "status": "complete",
                    "identity": expert.identity,
                    "measurements": measurements,
                }
        except Exception as err:
            for record in records:
                record["experts"][e["id"]] = {"status": "blocked", "error": str(err)}
        finally:
            torch.backends.cudnn.allow_tf32 = previous_tf32
            if expert:
                expert.close()
            del expert
            gc.collect()
        print(
            f"extracted {e['id']} on shard {shard_index}/{shard_count}: {time.monotonic() - started:.1f}s",
            flush=True,
        )
    for r in records:
        originals = [
            v["measurements"][0]["status"] == "valid"
            for v in r["experts"].values()
            if v["status"] == "complete"
        ]
        r["status"] = "scored" if any(originals) else "review"
        r["complete_all_experts"] = len(originals) == len(cfg["experts"]) and all(
            originals
        )
    out = (
        Path(output)
        if output
        else Path(cfg["output"]) / f"extraction-{shard_index}-of-{shard_count}.jsonl"
    )
    write_jsonl(out, records)
    return {
        "index": str(out),
        "requested": len(records),
        "scored": sum(r["status"] == "scored" for r in records),
        "complete_all_experts": sum(r["complete_all_experts"] for r in records),
    }


def merge(indices, expected_manifest, destination):
    rows = [r for p in indices for r in read_jsonl(p)]
    unique(rows)
    expected = read_jsonl(expected_manifest)
    unique(expected)
    if {r["id"] for r in rows} != {r["id"] for r in expected}:
        raise ValueError("Shard union has missing/unexpected IDs")
    write_jsonl(destination, sorted(rows, key=lambda r: r["id"]))
    return {"requested": len(rows), "status": "complete_union"}
