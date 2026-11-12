"""Development-only schedules and selection. Audit conditions never enter forward."""

from collections import defaultdict

import numpy as np
import torch

from ..common import Blocked, digest
from ..evaluation.metrics import metrics
from ..evaluation.stage_gate import discrimination_gate
from .sampling import origin_uniform_epoch


def require_role(batch, role):
    if len(batch.audit) != len(batch.ids) or any(
        r.get("role") != role for r in batch.audit
    ):
        raise ValueError("Expected " + role + " evidence only")
    if len(batch.ids) != len(set(batch.ids)):
        raise ValueError("Duplicate measurement ID")
    labels = {}
    for origin, label in zip(batch.origins, batch.labels.tolist()):
        if label not in (0, 1) or (origin in labels and labels[origin] != label):
            raise ValueError("Conflicting/missing origin label")
        labels[origin] = label


def epoch_schedule(batch, seed, epoch):
    """One valid T per origin; independent of architecture, dropout RNG and labels."""
    valid = batch.valid[:, :, 0].any(1).tolist()
    chosen = origin_uniform_epoch(batch.origins, batch.ids, valid, seed, epoch)
    chosen.sort(key=lambda i: digest([seed, epoch, "origin_order_v2", batch.ids[i]]))
    return torch.tensor(chosen, dtype=torch.long)


def validation_metrics(batch, logits, primary_conditions=None, observed=None):
    require_role(batch, "fusion_val")
    if observed is not None and not bool(observed.all()):
        raise Blocked(
            "Validation coverage failure; no silently filtered model selection"
        )
    y = batch.labels.numpy()
    s = logits.detach().cpu().numpy()
    if primary_conditions is None:
        return metrics(y, s)
    if not primary_conditions or len(set(primary_conditions)) != len(
        primary_conditions
    ):
        raise ValueError("Explicit unique primary validation conditions required")
    conditions = np.array([r.get("condition") for r in batch.audit])
    if any(c is None for c in conditions):
        raise ValueError("Missing audit condition")
    if not set(primary_conditions) <= set(conditions):
        raise Blocked("Prespecified validation condition is missing")
    per_condition = {}
    for condition in sorted(set(conditions)):
        ix = conditions == condition
        per_condition[condition] = metrics(y[ix], s[ix])
    return {
        "eer": float(np.mean([per_condition[c]["eer"] for c in primary_conditions])),
        "metric": "prespecified_condition_macro_EER",
        "primary_conditions": list(primary_conditions),
        "per_condition": per_condition,
        "N_requested": len(y),
        "N_scored": len(y),
        "N_failed": 0,
    }


def select_family_trials(results, seeds, families):
    """One common hyperparameter candidate per family, averaged across fixed seeds."""
    selected = {}
    summaries = []
    for family in families:
        candidates = defaultdict(list)
        for row in results:
            if row["model"] == family:
                candidates[row.get("trial_id", "default")].append(row)
        if not candidates:
            raise ValueError("Missing baseline family " + family)
        ranked = []
        for trial, rows in candidates.items():
            if len(rows) != len(seeds) or sorted(r["seed"] for r in rows) != sorted(
                seeds
            ):
                raise ValueError("Incomplete/duplicate paired seed budget")
            mean_eer = float(np.mean([r["val_eer"] for r in rows]))
            mean_bce = float(np.mean([r["val_bce"] for r in rows]))
            if not np.isfinite([mean_eer, mean_bce]).all():
                raise ValueError("Undefined validation selection metric")
            ranked.append((mean_eer, mean_bce, trial, rows))
        eer, bce, trial, rows = min(ranked, key=lambda r: r[:3])
        selected[family] = rows
        summaries.append(
            {
                "model": family,
                "trial_id": trial,
                "mean_val_eer": eer,
                "mean_val_bce": bce,
                "selection_role": "fusion_val",
                "test_used": False,
            }
        )
    return selected, summaries


def h1_decision(cfg, results, val):
    require_role(val, "fusion_val")
    p = cfg["pilot"]
    definition = p["h1_gate"]
    families = p["h1_models"]
    selected, summaries = select_family_trials(results, p["seeds"], families)
    baseline = selected["original_mlp"]
    candidates = p.get("h1_response_candidates", ["response_mlp", "all_views_mlp"])

    def avg(name, key):
        return float(np.mean([r[key] for r in selected[name]]))

    response = min(candidates, key=lambda n: (avg(n, "val_eer"), avg(n, "val_bce"), n))
    response_by_seed = {r["seed"]: r for r in selected[response]}
    improved = sum(
        response_by_seed[r["seed"]]["val_eer"] < r["val_eer"] - 1e-9 for r in baseline
    )
    supported = (
        avg("original_mlp", "val_eer") - avg(response, "val_eer")
        >= definition["min_absolute_val_eer_improvement"]
        and improved >= definition["improved_seeds_at_least"]
        and (
            not definition.get("mean_val_bce_must_not_worsen", True)
            or avg(response, "val_bce") <= avg("original_mlp", "val_bce")
        )
    )
    # Check each primary condition independently; pooling could hide too few groups.
    conditions = p.get("validation_eer_conditions")
    checks = {}
    for condition in conditions or [None]:
        indices = [
            i
            for i, r in enumerate(val.audit)
            if condition is None or r.get("condition") == condition
        ]
        checks[str(condition)] = discrimination_gate(
            [int(val.labels[i]) for i in indices],
            [val.groups[i] for i in indices],
            [r["val_eer"] for r in baseline],
            min_improvement=definition["min_absolute_val_eer_improvement"],
            min_groups_per_class=definition.get("min_groups_per_class", 5),
        )
    insufficient = any(
        r["status"] == "INSUFFICIENT_DISCRIMINATION" for r in checks.values()
    )
    unconverged = [
        {
            "model": family,
            "seed": r["seed"],
            "trial_id": r.get("trial_id", "default"),
            "stop_reason": r["stop_reason"],
        }
        for family, rows in selected.items()
        for r in rows
        if r["parameters"] and not r["convergence_claim"]
    ]
    status = "SUPPORTED_IN_PILOT" if supported else "NOT_SUPPORTED_IN_PILOT"
    if p.get("require_convergence", False) and unconverged:
        status = "INCONCLUSIVE_CONVERGENCE"
    if insufficient:
        status = "INSUFFICIENT_DISCRIMINATION"
    return {
        "status": status,
        "selected_response": response,
        "family_selection": summaries,
        "original_mean_val_eer": avg("original_mlp", "val_eer"),
        "response_mean_val_eer": avg(response, "val_eer"),
        "improved_seeds": improved,
        "definition": definition,
        "test_used": False,
        "discrimination": checks,
        "unconverged_selected_runs": unconverged,
        "H2_allowed": status == "SUPPORTED_IN_PILOT",
        "interpretation": "Validation-only gate; insufficient discrimination or convergence is inconclusive, never H1_FAILED",
    }
