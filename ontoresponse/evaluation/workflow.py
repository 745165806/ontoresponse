from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import torch

from ..common import Blocked, atomic_json, compute_device, read_jsonl, write_jsonl
from ..evidence.dataset import load_batch
from ..train.calibrate import Platt
from ..train.fit import forward, restore
from .metrics import choose_threshold
from .report import summarize


def calibrate(cfg, checkpoint, index, manifest, output=None):
    if (Path(checkpoint).parent / "freeze.json").exists():
        raise ValueError(
            "Frozen calibration is immutable; use a new experiment directory"
        )
    cp, model, stats, reg = restore(checkpoint, compute_device(cfg))
    cal = load_batch(cfg, index, manifest, "calibration", cp["dims"])
    val = load_batch(cfg, index, manifest, "fusion_val", cp["dims"])
    if val.ids != cp["val_ids"]:
        raise ValueError("Calibration must use the model-selection validation role")
    if set(cal.ids) & set(cp["train_ids"] + cp["val_ids"]):
        raise ValueError("Calibration role leakage")
    if set(cal.groups) & set(cp.get("train_groups", []) + cp.get("val_groups", [])):
        raise ValueError("Calibration source group leakage")
    with torch.inference_mode():
        c = forward(model, cal, stats, reg)
        v = forward(model, val, stats, reg)
    if not c.observed.all() or not v.observed.all():
        raise Blocked("Calibration/threshold contains missing evidence")
    params = Platt.fit(c.logits.cpu().numpy(), cal.labels.numpy(), "calibration")
    threshold = choose_threshold(
        val.labels.numpy(), Platt.apply(v.logits.cpu().numpy(), params), "fusion_val"
    )
    train_groups = cp.get("train_groups")
    if train_groups is None:
        train_groups = load_batch(cfg, index, manifest, "fusion_train").groups
    payload = {
        "status": "calibrated_smoke",
        "run_id": cp.get("run_id"),
        "checkpoint": str(checkpoint),
        "calibration": params,
        "threshold": threshold,
        "calibration_ids": cal.ids,
        "calibration_groups": sorted(set(cal.groups)),
        "train_and_val_groups": sorted(set(train_groups + val.groups)),
        "probability_scope": "integration dev prior only; no arbitrary deployment posterior claim",
    }
    atomic_json(output or Path(checkpoint).parent / "calibration.json", payload)
    return payload


def evaluation_plan(cfg, checkpoint, test_manifest):
    """Check role separation and collect the selected operating point, without hashes."""
    cp, model, stats, reg = restore(checkpoint)
    calibration_path = Path(checkpoint).parent / "calibration.json"
    if not calibration_path.is_file():
        raise Blocked("Calibrate selected checkpoint before freezing")
    cal = json.loads(calibration_path.read_text())
    if cp.get("run_id") and cal.get("run_id") != cp["run_id"]:
        raise ValueError("Stale calibration")
    all_rows = read_jsonl(test_manifest)
    from ..data.prepare import validate_splits

    validate_splits(all_rows)
    rows = [r for r in all_rows if r["role"] == "test"]
    if not rows:
        raise ValueError("No test requests in the manifest")
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
        "run_id": cp.get("run_id"),
        "calibration": cal,
        "test_ids": [r["id"] for r in rows],
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
    return payload


def freeze(cfg, checkpoint, test_manifest, output=None):
    """Optional compatibility snapshot; ordinary evaluation needs no freeze step."""
    payload = evaluation_plan(cfg, checkpoint, test_manifest)
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
    if [r["id"] for r in read_jsonl(manifest) if r["role"] == "test"] != lock[
        "test_ids"
    ]:
        raise ValueError("Test request IDs differ from the selected evaluation plan")
    if lock["experts"] != cfg["experts"] or lock["crop"] != cfg["crop"]:
        raise ValueError("Frozen expert/crop drift")


def evaluate(cfg, checkpoint, index, manifest, lock_path=None, output=None):
    current = evaluation_plan(cfg, checkpoint, manifest)
    lock = json.loads(Path(lock_path).read_text()) if lock_path else current
    check_lock(cfg, checkpoint, manifest, lock)
    if lock.get("run_id") and lock["run_id"] != current["run_id"]:
        raise ValueError("Evaluation plan belongs to another trained model")
    if lock["calibration"]["calibration"] != current["calibration"]["calibration"]:
        raise ValueError("Evaluation plan uses a different calibration")
    cp, model, stats, reg = restore(checkpoint, compute_device(cfg))
    from ..evidence.cache import Cache
    from ..ontology.registry import validate_evidence

    records = read_jsonl(index)
    cache = Cache(cfg["cache"])
    validate_evidence(cfg, records, cache)
    data = load_batch(cfg, index, manifest, "test", cp["dims"])
    if data.ids != lock["test_ids"]:
        raise ValueError("Frozen request order differs")
    with torch.inference_mode():
        out = forward(model, data, stats, reg)
    probs = Platt.apply(out.logits.cpu().numpy(), lock["calibration"]["calibration"])
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
                "model_id": cp.get("run_id", Path(checkpoint).parent.name),
            }
        )
    root = Path(output or Path(checkpoint).parent / "evaluation")
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "per_utterance_predictions.jsonl", rows)
    report = summarize(rows)
    report.update(
        status="integration_smoke_only",
        source_mapping_complete=lock["source_mapping_complete"],
    )
    atomic_json(root / "report.json", report)
    return report


def infer(cfg, checkpoint, audio, output=None):
    cp, model, stats, reg = restore(checkpoint, compute_device(cfg))
    if cp["expert_specs"] != cfg["experts"] or cp["crop"] != cfg["crop"]:
        raise ValueError("Inference expert/crop identity differs from training")
    from ..evidence.extract import extract

    ident = "inference-" + uuid.uuid4().hex[:12]
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
        if cal and cp.get("run_id") and cal.get("run_id") != cp["run_id"]:
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
