from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from ..common import Blocked, atomic_json, digest, read_jsonl, sha256, write_jsonl
from ..evidence.dataset import load_batch
from ..resources import resource_path
from ..train.calibrate import Platt
from ..train.fit import forward, restore
from .metrics import choose_threshold
from .report import summarize


def calibrate(cfg, checkpoint, index, manifest, output=None):
    if (Path(checkpoint).parent / "freeze.json").exists():
        raise ValueError(
            "Frozen calibration is immutable; use a new experiment directory"
        )
    cp, model, stats, reg = restore(checkpoint)
    if cp["index_sha256"] != sha256(index) or cp["manifest_sha256"] != sha256(manifest):
        raise ValueError("Calibration must use frozen training ledger/cache identity")
    cal = load_batch(cfg, index, manifest, "calibration", cp["dims"])
    val = load_batch(cfg, index, manifest, "fusion_val", cp["dims"])
    if set(cal.ids) & set(cp["train_ids"] + cp["val_ids"]):
        raise ValueError("Calibration role leakage")
    with torch.inference_mode():
        c = forward(model, cal, stats, reg)
        v = forward(model, val, stats, reg)
    if not c.observed.all() or not v.observed.all():
        raise Blocked("Calibration/threshold contains missing evidence")
    params = Platt.fit(c.logits.numpy(), cal.labels.numpy(), "calibration")
    threshold = choose_threshold(
        val.labels.numpy(), Platt.apply(v.logits.numpy(), params), "fusion_val"
    )
    payload = {
        "status": "calibrated_smoke",
        "checkpoint_sha256": sha256(checkpoint),
        "calibration": params,
        "threshold": threshold,
        "calibration_ids": cal.ids,
        "calibration_groups": sorted(set(cal.groups)),
        "train_and_val_groups": sorted(
            set(load_batch(cfg, index, manifest, "fusion_train").groups + val.groups)
        ),
        "manifest_sha256": sha256(manifest),
        "index_sha256": sha256(index),
        "probability_scope": "integration dev prior only; no arbitrary deployment posterior claim",
    }
    atomic_json(output or Path(checkpoint).parent / "calibration.json", payload)
    return payload


def freeze(cfg, checkpoint, test_manifest, output=None):
    cp, model, stats, reg = restore(checkpoint)
    calibration_path = Path(checkpoint).parent / "calibration.json"
    if not calibration_path.is_file():
        raise Blocked("Calibrate selected checkpoint before freezing")
    cal = json.loads(calibration_path.read_text())
    if cal["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("Stale calibration")
    rows = read_jsonl(test_manifest)
    from ..data.prepare import validate_splits

    validate_splits(rows)
    if any(r["role"] != "test" for r in rows):
        raise ValueError("Frozen test ledger must contain test requests only")
    if set(r["id"] for r in rows) & set(
        cp["train_ids"] + cp["val_ids"] + cal["calibration_ids"]
    ):
        raise ValueError("Test ID leakage")
    used = set(cal["train_and_val_groups"] + cal["calibration_groups"])
    if used & set(r["source_group"] for r in rows):
        raise ValueError("Test group leakage")
    if cfg["mode"] == "research_holdout":
        if not all(
            e.get("supervision", {}).get("independent_holdout") is True
            for e in cfg["experts"]
        ):
            raise Blocked("Expert supervision overlaps fusion roles or remains unknown")
        if any(r.get("source_overlap_status") != "known_disjoint" for r in rows):
            raise Blocked("Research test source mapping incomplete")
    from ..transforms.environment import exposure_audit

    ledger = exposure_audit(
        [[]], ["lowpass", "additive_white_noise", "mp3"], [["aac"], ["opus"]], None
    )
    payload = {
        "status": "frozen_integration_smoke",
        "mode": cfg["mode"],
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "calibration": cal,
        "registry_hash": reg.digest,
        "probe_config_sha256": sha256(resource_path("configs/probes.yaml")),
        "ontology_sha256": sha256(resource_path("configs/ontology.yaml")),
        "test_manifest_sha256": sha256(test_manifest),
        "test_ids": [r["id"] for r in rows],
        "test_audio_sha256": {
            r["id"]: sha256(r["audio"]) if Path(r["audio"]).is_file() else None
            for r in rows
        },
        "experts": cfg["experts"],
        "crop": cfg["crop"],
        "exposure": ledger,
        "selection_rule": cp["selection_rule"],
        "primary_metric": "fusion_unseen_condition_macro_EER",
        "source_mapping_complete": all(
            r.get("source_overlap_status") == "known_disjoint" for r in rows
        ),
        "limitations": [
            "engineering smoke, not official full benchmark",
            "expert augmentation/selection independence not fully certified",
        ],
    }
    path = Path(output or Path(checkpoint).parent / "freeze.json")
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "Freeze lock already exists with different contents; use a new experiment path"
            )
    else:
        atomic_json(path, payload)
    return payload


def check_lock(cfg, checkpoint, manifest, lock):
    if lock["checkpoint_sha256"] != sha256(checkpoint) or lock[
        "test_manifest_sha256"
    ] != sha256(manifest):
        raise ValueError("Frozen checkpoint/test ledger drift")
    current_audio = {
        r["id"]: sha256(r["audio"]) if Path(r["audio"]).is_file() else None
        for r in read_jsonl(manifest)
    }
    if current_audio != lock["test_audio_sha256"]:
        raise ValueError("Frozen test waveform content drift")
    if lock["experts"] != cfg["experts"] or lock["crop"] != cfg["crop"]:
        raise ValueError("Frozen expert/crop drift")
    if lock["probe_config_sha256"] != sha256(
        resource_path("configs/probes.yaml")
    ) or lock["ontology_sha256"] != sha256(resource_path("configs/ontology.yaml")):
        raise ValueError("Frozen probe/ontology drift")


def evaluate(cfg, checkpoint, index, manifest, lock_path, output=None):
    lock = json.loads(Path(lock_path).read_text())
    check_lock(cfg, checkpoint, manifest, lock)
    cp, model, stats, reg = restore(checkpoint)
    from ..evidence.cache import Cache
    from ..ontology.registry import validate_evidence

    records = read_jsonl(index)
    cache = Cache(cfg["cache"])
    validate_evidence(cfg, records, cache)
    for row in records:
        if row.get("views") and row["views"][0]["status"] == "valid":
            meta = cache.get(row["views"][0]["artifact_key"])[1]
            if meta["spec"].get("audio_file_sha256") != lock["test_audio_sha256"].get(
                row["id"]
            ):
                raise ValueError("Test cache audio identity differs from freeze lock")
    data = load_batch(cfg, index, manifest, "test", cp["dims"])
    if data.ids != lock["test_ids"]:
        raise ValueError("Frozen request order differs")
    with torch.inference_mode():
        out = forward(model, data, stats, reg)
    probs = Platt.apply(out.logits.numpy(), lock["calibration"]["calibration"])
    threshold = lock["calibration"]["threshold"]["threshold"]
    rows = []
    for i, ident in enumerate(data.ids):
        observed = bool(out.observed[i])
        audit = data.audit[i]
        rows.append(
            {
                "id": ident,
                "status": "scored" if observed else "review",
                "label": int(data.labels[i]),
                "source_group": data.groups[i],
                "origin_id": data.origins[i],
                "condition": audit.get("condition", "no_new_processing"),
                "source_overlap_status": audit.get("source_overlap_status", "unknown"),
                "logit": float(out.logits[i]) if observed else None,
                "p": float(probs[i]) if observed else None,
                "threshold": threshold,
                "checkpoint_sha256": lock["checkpoint_sha256"],
                "registry_hash": reg.digest,
            }
        )
    root = Path(output or Path(checkpoint).parent / "evaluation")
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "per_utterance_predictions.jsonl", rows)
    report = summarize(rows)
    report.update(
        status="integration_smoke_only",
        lock_sha256=sha256(lock_path),
        source_mapping_complete=lock["source_mapping_complete"],
    )
    atomic_json(root / "report.json", report)
    return report


def infer(cfg, checkpoint, audio, output=None):
    cp, model, stats, reg = restore(checkpoint)
    if cp["expert_specs"] != cfg["experts"] or cp["crop"] != cfg["crop"]:
        raise ValueError("Inference expert/crop identity differs from training")
    from ..evidence.extract import extract

    ident = "inference-" + (
        sha256(audio) if Path(audio).is_file() else digest(str(Path(audio)))
    )
    local = copy.deepcopy(cfg)
    local["output"] = str(Path(cfg["output"]) / "inference" / ident)
    root = Path(local["output"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "input.jsonl"
    write_jsonl(
        manifest,
        [{"id": ident, "origin_id": ident, "source_group": ident, "audio": str(audio)}],
    )
    result = extract(local, manifest, 1)
    try:
        data = load_batch(local, result["index"], manifest, "inference", cp["dims"])
        with torch.inference_mode():
            out = forward(model, data, stats, reg)
        observed = bool(out.observed[0])
        calibration_path = Path(checkpoint).parent / "calibration.json"
        cal = (
            json.loads(calibration_path.read_text())
            if calibration_path.exists()
            else None
        )
        if cal and cal["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Stale calibration")
        logit = float(out.logits[0]) if observed else None
        probability = (
            float(Platt.apply([logit], cal["calibration"])[0])
            if observed and cal
            else float(out.probabilities()[0])
            if observed
            else None
        )
        row = {
            "id": ident,
            "status": "scored" if observed else "review",
            "logit": logit,
            "p": probability,
            "calibrated": bool(cal),
        }
    except ValueError as err:
        row = {
            "id": ident,
            "status": "review",
            "logit": None,
            "p": None,
            "error": str(err),
        }
    atomic_json(output or root / "decision.json", row)
    return row


def execute(args, cfg):
    root = Path(cfg["output"])
    index = args.index or root / "extraction-0-of-1.jsonl"
    manifest = args.manifest or root / "audit_manifest.jsonl"
    if args.command == "calibrate":
        return calibrate(cfg, args.checkpoint, index, manifest, args.output)
    if args.command == "freeze":
        return freeze(cfg, args.checkpoint, manifest, args.output)
    if args.command == "evaluate":
        if args.role != "test":
            raise ValueError("This entrypoint evaluates frozen test only")
        return evaluate(cfg, args.checkpoint, index, manifest, args.lock, args.output)
    if args.command == "infer":
        return infer(cfg, args.checkpoint, args.audio, args.output)
    if args.command == "report":
        report = summarize(read_jsonl(args.manifest))
        atomic_json(args.output or root / "recomputed-report.json", report)
        return report
