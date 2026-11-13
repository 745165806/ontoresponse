"""Lightweight pilot checks: inputs, role separation, resources and explicit budgets."""

import shutil
from pathlib import Path

from .common import Blocked, atomic_json, compute_device, read_jsonl
from .data.pilot import check_pilot_budget
from .data.prepare import validate_splits


def preflight(cfg):
    check_pilot_budget(cfg)
    root = Path(cfg["output"])
    root.mkdir(parents=True, exist_ok=True)
    blocked = []
    if shutil.disk_usage(root).free < cfg["pilot"].get("min_free_disk_bytes", 0):
        blocked.append("Insufficient free disk for pilot budget")
    for name in ("audit_manifest.jsonl", "inference_manifest.jsonl"):
        if not (root / name).is_file():
            blocked.append("Input manifest missing: " + name)
    if not blocked:
        try:
            rows = read_jsonl(root / "audit_manifest.jsonl")
            validate_splits(rows)
            counts = {
                role: len({r["origin_id"] for r in rows if r["role"] == role})
                for role in cfg["pilot"]["origins_by_role"]
            }
            if counts != cfg["pilot"]["origins_by_role"]:
                raise ValueError("Origin counts differ from the configured role budget")
            projection = [
                {k: r[k] for k in ("id", "origin_id", "source_group", "audio")}
                for r in rows
            ]
            inference = read_jsonl(root / "inference_manifest.jsonl")
            if sorted(projection, key=lambda r: r["id"]) != sorted(
                inference, key=lambda r: r["id"]
            ):
                raise ValueError(
                    "Inference manifest must be the label-free data projection"
                )
            missing = [r["id"] for r in rows if not Path(r["audio"]).is_file()]
            if missing:
                raise ValueError("Audio missing: " + ", ".join(missing[:5]))
            if cfg["pilot"].get("protocol_version") == 2:
                from .preregistration import require_preregistration

                require_preregistration(cfg)
        except (Blocked, ValueError, KeyError, OSError) as err:
            blocked.append(str(err))
    for expert in cfg["experts"]:
        if expert.get("task_trained") is not True:
            blocked.append("Task-trained expert required: " + expert["id"])
        for name in ("checkpoint", "frontend"):
            if name == "frontend" and not expert.get(name):
                continue
            if not expert.get(name) or not Path(expert[name]).is_file():
                blocked.append(expert["id"] + ": missing " + name)
    try:
        device = str(compute_device(cfg))
    except Blocked as err:
        blocked.append(str(err))
        device = cfg.get("runtime", {}).get("device")
    result = {
        "status": "BLOCKED" if blocked else "PASS_INTEGRATION_PILOT",
        "blocked": blocked,
        "research_holdout_pass": False,
        "claim_scope": "integration_pilot_only",
        "device": device,
        "budget": cfg["pilot"],
        "limitations": [
            "Public expert training/model-selection overlap must be disclosed; no strict holdout claim"
        ],
    }
    atomic_json(root / "preflight.json", result)
    return result


def require_pass(cfg):
    report = preflight(cfg)
    if report["blocked"]:
        raise Blocked("; ".join(report["blocked"]))
    return report
