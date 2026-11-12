"""Prepare a bounded source-certified base ledger, then audit materialized T versions."""

from pathlib import Path

from ..common import Blocked, atomic_json, budget, read_jsonl, sha256, write_jsonl
from ..preregistration import validate_development_data
from .connected_split import allocate_connected
from .prepare import validate_splits
from .source_audit import excluded_components, source_universe


def prepare_connected_base(cfg, candidate_manifest):
    p = cfg["pilot"]
    root = Path(cfg["output"])
    base = root / "base_manifest.jsonl"
    if any(
        (root / name).exists()
        for name in (
            "base_manifest.jsonl",
            "audit_manifest.jsonl",
            "preregistration.json",
            "experiment-start.json",
        )
    ):
        raise ValueError("Do not overwrite a prepared/frozen connected split")
    budget(cfg["runtime"]["budget"], sum(p["origins_by_role"].values()))
    universe, components = source_universe(cfg)
    excluded = excluded_components(cfg, universe, components)
    mapping = {r["id"]: r for r in universe}
    candidates = read_jsonl(candidate_manifest)
    # Caller must supply original target recording IDs, not T/P artifact IDs.
    for row in candidates:
        if row["origin_id"] not in mapping:
            raise Blocked("Unmapped candidate")
        if row["speaker_id"] != mapping[row["origin_id"]]["speaker_id"]:
            raise Blocked("Candidate speaker differs from source map")
        if (
            "label" in mapping[row["origin_id"]]
            and row["label"] != mapping[row["origin_id"]]["label"]
        ):
            raise Blocked("Candidate label differs from audited source metadata")
    selected, report = allocate_connected(
        candidates,
        components,
        excluded,
        p["origins_by_role"],
        minimum=p["h1_gate"].get("min_groups_per_class", 5),
        seed=p.get("split_seed", 271828),
        time_limit=p.get("allocation_timeout_seconds", 30),
    )
    root.mkdir(parents=True, exist_ok=True)
    report.update(
        candidate_manifest_sha256=sha256(candidate_manifest),
        source_map_sha256=sha256(p["source_map_manifest"]),
        official_archive_sha256=sha256(p["official_map_archive"]),
        excluded_components=len(excluded),
    )
    if not selected:
        atomic_json(root / "base-split-audit.json", report)
        return report
    missing = [r["id"] for r in selected if not Path(r["audio"]).is_file()]
    if missing:
        report.update(
            status="BLOCKED_MISSING_SELECTED_AUDIO", missing_selected_audio=missing
        )
        atomic_json(root / "base-split-audit.json", report)
        return report
    report["role_distribution"] = validate_splits(selected)
    for row in selected:
        row["base_audio_file_sha256"] = sha256(row["audio"])
    write_jsonl(base, selected)
    report.update(
        status="PREPARED_CONNECTED_BASE",
        base_manifest=str(base),
        base_manifest_sha256=sha256(base),
        scope="integration_pretrained only; source mapping certified, task HPO / generic SSL open risks separately disclosed",
    )
    atomic_json(root / "base-split-audit.json", report)
    return report


def audit_materialized(cfg):
    """Recompute source/role checks and actual cross-role waveform duplicate audit."""
    root = Path(cfg["output"])
    path = root / "data-audit.json"
    if path.exists() or Path(cfg["pilot"]["preregistration"]).exists():
        raise ValueError("Do not overwrite audited/frozen data")
    checked = validate_development_data(cfg, require_data_audit=False)
    from ..transforms.probes import audio_hash
    from .audio import read_audio

    rows = read_jsonl(root / "audit_manifest.jsonl")
    owners = {}
    duplicates = []
    for row in rows:
        fingerprint = audio_hash(read_audio(row["audio"]))
        if fingerprint in owners and owners[fingerprint] != row["role"]:
            duplicates.append(row["id"])
        owners[fingerprint] = row["role"]
    if duplicates:
        raise Blocked("Cross-role waveform duplicate: " + str(duplicates[:10]))
    result = dict(
        checked,
        origins=sum(checked["unique_origins"].values()),
        ledger_sha256=sha256(root / "audit_manifest.jsonl"),
        expert_ids=sorted(e["id"] for e in cfg["experts"]),
        upstream_exact_ID_overlap=0,
        upstream_speaker_overlap=0,
        prior_development_test_speaker_overlap=0,
        waveform_cross_role_duplicates=0,
        consumed_exploration_origin_overlap=0,
        normalization_selection="fusion_train only; calibration/test not fitted",
        waveform_checks=len(rows),
        scope="connected source integration; not strict research_holdout",
    )
    atomic_json(path, result)
    return result
