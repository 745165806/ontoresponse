"""Extract shared probe artifacts and real frozen-expert observations."""

import gc
import time
from pathlib import Path

import yaml

from ..common import budget, file_info, read_jsonl, unique, write_jsonl
from ..data.audio import quality, read_audio
from ..data.prepare import shard
from ..experts.base import AudioInput, FrozenExpert
from ..resources import resource_path
from ..transforms.probes import apply_probe
from .cache import Cache


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
        started = time.monotonic()
        try:
            expert = FrozenExpert(e, cfg["crop"], cfg["runtime"]["device"])
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
