from collections import Counter, defaultdict

import numpy as np

from ..common import unique
from .metrics import metrics, paired_bootstrap


def summarize(rows):
    unique(rows)
    statuses = Counter(r["status"] for r in rows)
    scored = [r for r in rows if r["status"] == "scored"]
    result = {
        "N_requested": len(rows),
        "N_scored": len(scored),
        "N_failed": len(rows) - len(scored),
        "statuses": dict(statuses),
        "effective_subset_only": len(scored) != len(rows),
    }
    groups = defaultdict(list)
    for r in scored:
        groups[r["condition"]].append(r)
    conditions = {}
    for cond, part in groups.items():
        try:
            conditions[cond] = metrics(
                [r["label"] for r in part],
                [r["logit"] for r in part],
                [r["p"] for r in part],
                threshold=None,
            )
            pred = np.array([r["p"] >= r["threshold"] for r in part])
            y = np.array([r["label"] for r in part])
            conditions[cond].update(
                actual_fpr=float(pred[y == 0].mean()),
                actual_tpr=float(pred[y == 1].mean()),
            )
        except ValueError as e:
            conditions[cond] = {"status": "undefined", "error": str(e), "n": len(part)}
    result["conditions"] = conditions
    unseen = [
        v["eer"]
        for k, v in conditions.items()
        if k != "no_new_processing" and "eer" in v
    ]
    result["unseen_macro_eer"] = float(np.mean(unseen)) if unseen else None
    result["no_new_processing"] = conditions.get("no_new_processing")
    # Conservative fixed-policy bounds count every failed request, by class.
    bounds = {}
    for y, name in ((0, "fpr"), (1, "tpr")):
        all_y = [r for r in rows if r.get("label") == y]
        valid = [r for r in scored if r.get("label") == y]
        positive = sum(r["p"] >= r["threshold"] for r in valid)
        missing = len(all_y) - len(valid)
        bounds[name] = (
            [positive / len(all_y), (positive + missing) / len(all_y)]
            if all_y
            else None
        )
    result["failure_policy_bounds"] = bounds
    return result


def paired_report(rows_a, rows_b, replicates=2000):
    unique(rows_a)
    unique(rows_b)
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    if set(a) != set(b):
        raise ValueError("Requested ID sets differ")
    shared = []
    for ident, r in a.items():
        other = b[ident]
        if (r["label"], r["source_group"], r["condition"]) != (
            other["label"],
            other["source_group"],
            other["condition"],
        ):
            raise ValueError("Paired audit mismatch")
        if r["status"] == other["status"] == "scored":
            shared.append(ident)
    if not shared:
        raise ValueError("No shared scored requests")
    # Macro condition EER must be used for the primary comparison, not pooled EER.
    conditions = sorted(
        {a[i]["condition"] for i in shared if a[i]["condition"] != "no_new_processing"}
    )
    if not conditions:
        conditions = ["no_new_processing"]
    return {
        "N_common_scored": len(shared),
        "N_requested": len(a),
        "condition_bootstrap": {
            condition: paired_bootstrap(
                [a[i]["label"] for i in shared if a[i]["condition"] == condition],
                [a[i]["logit"] for i in shared if a[i]["condition"] == condition],
                [b[i]["logit"] for i in shared if a[i]["condition"] == condition],
                [
                    a[i]["source_group"]
                    for i in shared
                    if a[i]["condition"] == condition
                ],
                replicates=replicates,
            )
            for condition in conditions
        },
    }
