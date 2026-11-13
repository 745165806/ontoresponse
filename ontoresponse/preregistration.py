"""Source/role checks and an optional human-readable development configuration snapshot."""

from pathlib import Path

from .common import Blocked, atomic_json, digest, read_jsonl
from .data.prepare import validate_splits
from .data.source_audit import coverage_protocols, excluded_components, source_universe


def evidence_paths(cfg):
    p = cfg["pilot"]
    root = Path(cfg["output"])
    paths = {
        "ledger": root / "audit_manifest.jsonl",
        "inference": root / "inference_manifest.jsonl",
        "data_audit": root / "data-audit.json",
        "source_map": Path(p["source_map_manifest"]),
        "consumption": Path(p["consumption_ledger"]),
        "supervision": Path(p["supervision_audit"]),
    }
    for expert in cfg["experts"]:
        paths["checkpoint:" + expert["id"]] = Path(expert["checkpoint"])
    paths.update(
        {"coverage:" + name: path for name, path in coverage_protocols(cfg).items()}
    )
    if paths["ledger"].is_file():
        for row in read_jsonl(paths["ledger"]):
            paths["current_audio:" + row["id"]] = Path(row["audio"])
    return paths


def validate_development_data(cfg, require_data_audit=True):
    """Recompute connectivity on the full supplied source universe, including exclusions."""
    p = cfg["pilot"]
    paths = evidence_paths(cfg)
    missing = [
        str(path)
        for key, path in paths.items()
        if not path.is_file() and (require_data_audit or key != "data_audit")
    ]
    if missing:
        raise Blocked("Missing v2 provenance: " + ", ".join(missing))
    universe, components = source_universe(cfg)
    mapped = {r["id"]: r for r in universe}
    excluded_groups = excluded_components(cfg, universe, components)
    rows = read_jsonl(paths["ledger"])
    validate_splits(rows)
    groups_by_class = {}
    ownership = {}
    counts = {}
    for row in rows:
        origin = row["origin_id"]
        if origin not in mapped:
            raise Blocked("Unmapped selected origin " + origin)
        group = components[origin]
        if row["source_group"] != group:
            raise Blocked("Ledger differs from full source connectivity")
        if row["speaker_id"] != mapped[origin]["speaker_id"]:
            raise Blocked("Speaker metadata drift")
        if group in excluded_groups:
            raise Blocked("Consumed/expert-exposed source component reused")
        if group in ownership and ownership[group] != row["role"]:
            raise Blocked("Connected component crosses roles")
        ownership[group] = row["role"]
        groups_by_class.setdefault((row["role"], row["label"]), set()).add(group)
        counts.setdefault(row["role"], set()).add(origin)
    if {k: len(v) for k, v in counts.items()} != p["origins_by_role"]:
        raise Blocked("Unique origin role budget drift")
    minimum = p["h1_gate"].get("min_groups_per_class", 5)
    if any(
        len(groups_by_class.get((role, y), set())) < minimum
        for role in counts
        for y in (0, 1)
    ):
        raise Blocked("Insufficient connected groups per role/class")
    # Paired condition inventories stop loss/metric weighting from drifting by origin.
    expected = p["conditions_by_role"]
    inventory = {}
    for row in rows:
        key = (row["role"], row["origin_id"])
        condition = row.get("condition")
        inventory.setdefault(key, []).append(condition)
    for (role, origin), conditions in inventory.items():
        wanted = expected[role]
        if role == "fusion_train" and p.get("training_t_candidates"):
            from .data.development import conditions_for_origin

            wanted = conditions_for_origin({"role": role, "origin_id": origin}, p)
        if len(conditions) != len(set(conditions)) or set(conditions) != set(wanted):
            raise Blocked("Unpaired/duplicate condition inventory: " + origin)
    inference = read_jsonl(paths["inference"])
    projection = [
        {k: r[k] for k in ("id", "origin_id", "source_group", "audio")} for r in rows
    ]
    if sorted(inference, key=lambda r: r["id"]) != sorted(
        projection, key=lambda r: r["id"]
    ):
        raise Blocked("Inference manifest differs from label-free ledger projection")
    return {
        "unique_origins": {k: len(v) for k, v in counts.items()},
        "groups_per_class": {
            f"{r}:{y}": len(v) for (r, y), v in groups_by_class.items()
        },
        "source_mapping_complete": True,
        "research_holdout_pass": False,
        "limitation": "Mapped integration experiment; generic SSL and unknown task HPO remain disclosed",
    }


def validate_v2_recipe(cfg):
    p = cfg["pilot"]
    if p.get("protocol_version") != 2:
        raise ValueError("Expected v2 recipe")
    if not p.get("require_convergence") or not p.get(
        "require_h1_support_before_confirmation"
    ):
        raise Blocked("v2 requires convergence and H1 confirmation gates")
    if len(set(p["seeds"])) < 3:
        raise Blocked("v2 requires at least three training seeds")
    required = {
        "score_lr",
        "score_mlp",
        "original_mlp",
        "tta_stats",
        "all_views_mlp",
        "response_mlp",
    }
    if set(p["h1_models"]) != required:
        raise Blocked("v2 H1 baseline inventory differs")
    grid = p.get("candidate_grid", [])
    if (
        len(grid) != p["trials_per_family"]
        or not grid
        or len({digest(c) for c in grid}) != len(grid)
    ):
        raise Blocked("v2 requires equal nonduplicate candidate budgets")
    primary = p.get("validation_eer_conditions")
    if (
        not primary
        or len(primary) != len(set(primary))
        or not set(primary) <= set(p["conditions_by_role"]["fusion_val"])
    ):
        raise Blocked("Primary validation conditions not frozen")
    if p.get("min_epochs", 0) < 40 or p["max_optimizer_steps"] <= 20:
        raise Blocked("v2 convergence budget cannot be a smoke run")
    return True


def freeze_development(cfg):
    validate_v2_recipe(cfg)
    root = Path(cfg["output"])
    path = Path(cfg["pilot"].get("preregistration", root / "experiment-config.json"))
    if path.exists():
        raise ValueError("Preregistration exists; do not refreeze")
    if (
        (root / "experiment-start.json").exists()
        or (root / "runs").exists()
        or list(root.glob("extraction-*.jsonl"))
    ):
        raise Blocked("Preregistration must precede extraction and any model outputs")
    report = validate_development_data(cfg)
    lock = {
        "status": "frozen_development_v2",
        "config": cfg,
        "files": {
            key: {"path": str(value)} for key, value in evidence_paths(cfg).items()
        },
        "data": report,
        "test_used": False,
    }
    atomic_json(path, lock)
    return lock


def require_preregistration(cfg):
    """Compatibility name: validate current inputs; no seal or document approval needed."""
    validate_v2_recipe(cfg)
    return {
        "status": "development_inputs_valid",
        "data": validate_development_data(cfg),
        "test_used": False,
    }
