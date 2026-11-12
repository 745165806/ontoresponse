"""Fail-closed audit gate; an explicit pilot budget never grants a holdout label."""

import json
import shutil
from pathlib import Path

from .common import Blocked, atomic_json, code_hash, digest, sha256
from .data.pilot import check_pilot_budget


def preflight(cfg):
    check_pilot_budget(cfg)
    p = cfg["pilot"]
    root = Path(cfg["output"])
    blocked = []
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < p.get("min_free_disk_bytes", 0):
        blocked.append("Insufficient free disk for explicit pilot budget")
    if not all(
        k in p
        for k in [
            "max_cache_bytes",
            "max_output_bytes",
            "h1_gate",
            "h1_models",
            "h2_models",
        ]
    ):
        blocked.append("Missing staged hypothesis or disk/cache budgets")
    if p.get("protocol_version") == 2:
        from .preregistration import require_preregistration

        try:
            require_preregistration(cfg)
        except (Blocked, ValueError, KeyError, OSError) as err:
            blocked.append("v2 preregistration: " + str(err))
    doc = Path(p["required_next_stage_document"])
    if not doc.is_file():
        blocked.append(
            "CODEX_NEXT_STAGE.md is missing; requested instructions cannot be read"
        )
    elif p.get("reviewed_next_stage_sha256") != sha256(doc):
        blocked.append(
            "Next-stage document exists but its reviewed hash is not recorded"
        )
    quality_path = Path(p["quality_gate"])
    if not quality_path.is_file():
        blocked.append(
            "Full tests / ontology / installed-package evidence not yet sealed"
        )
    else:
        quality = json.loads(quality_path.read_text())
        if quality.get("code_sha256") != code_hash() or not quality.get("passed"):
            blocked.append(
                "Quality evidence does not pass or is stale for current code"
            )
    data_path = root / "data-audit.json"
    ledger = root / "audit_manifest.jsonl"
    if not data_path.is_file() or not ledger.is_file():
        blocked.append("New pilot grouped ledger has not been audited")
    else:
        data = json.loads(data_path.read_text())
        if data["ledger_sha256"] != sha256(ledger) or data["origins"] != sum(
            p["origins_by_role"].values()
        ):
            blocked.append("Pilot ledger/budget drift")
        if data["expert_ids"] != sorted(e["id"] for e in cfg["experts"]):
            blocked.append("Audited expert set differs from config")
        for field in [
            "upstream_exact_ID_overlap",
            "upstream_speaker_overlap",
            "prior_development_test_speaker_overlap",
            "waveform_cross_role_duplicates",
            "consumed_exploration_origin_overlap",
        ]:
            if data[field] != 0:
                blocked.append("Data audit failed: " + field)
    audit_path = Path(p["supervision_audit"])
    if not audit_path.is_file():
        blocked.append("Source provenance audit missing")
    else:
        audit = json.loads(audit_path.read_text())
        known = {r["id"]: r for r in audit["experts"]}
        for e in cfg["experts"]:
            record = known.get(e["id"], {})
            if not record.get("provenance_pass"):
                blocked.append("Uncertified expert: " + e["id"])
            elif not Path(e["checkpoint"]).is_file():
                blocked.append("Missing task checkpoint: " + e["id"])
            elif sha256(e["checkpoint"]) != record["checkpoint_sha256"]:
                blocked.append("Expert checkpoint drift: " + e["id"])
    result = {
        "status": "BLOCKED" if blocked else "PASS_INTEGRATION_PILOT",
        "blocked": blocked,
        "research_holdout_pass": False,
        "claim_scope": "integration_pilot_only",
        "config_digest": digest(cfg),
        "code_sha256": code_hash(),
        "supervision_audit_sha256": sha256(audit_path)
        if audit_path.is_file()
        else None,
        "data_audit_sha256": sha256(data_path) if data_path.is_file() else None,
        "reviewed_next_stage_sha256": sha256(doc) if doc.is_file() else None,
        "quality_gate_sha256": sha256(quality_path) if quality_path.is_file() else None,
        "budget": p,
        "limitations": (
            [
                "Source-connected v2 integration; generic SSL and task HPO certification remain separate from mapping"
            ]
            if p.get("protocol_version") == 2
            else [
                "Recording-level official source map is UNKNOWN/BLOCKED; known task ID/speaker integration scope only"
            ]
        )
        + [
            "Generic SSL pretraining overlap remains unknown",
            "GPU visibility is not required for bounded CPU integration pilot",
        ],
    }
    atomic_json(root / "preflight.json", result)
    return result


def require_pass(cfg):
    report = preflight(cfg)
    if report["status"] != "PASS_INTEGRATION_PILOT":
        raise Blocked("; ".join(report["blocked"]))
    return report
