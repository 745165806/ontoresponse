"""Validation-only research gates; no confirmation scores accepted."""

import math
from collections import defaultdict


def discrimination_gate(
    labels,
    groups,
    baseline_eers,
    *,
    min_improvement=0.01,
    min_groups_per_class=5,
    degenerate_fraction=None,
    max_degenerate_fraction=0.1,
    role="fusion_val",
):
    if role != "fusion_val":
        raise ValueError("Stage decisions require fusion_val, never test")
    if len(labels) != len(groups) or not labels or not baseline_eers:
        raise ValueError("Missing validation evidence")
    if not 0 < min_improvement <= 1 or min_groups_per_class < 2:
        raise ValueError("Invalid gate design")
    by_label = defaultdict(set)
    for label, group in zip(labels, groups):
        if label not in (0, 1) or not group:
            raise ValueError("Invalid label/group")
        by_label[int(label)].add(group)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in baseline_eers):
        raise ValueError("Invalid EER")
    reasons = []
    counts = {str(y): len(by_label[y]) for y in (0, 1)}
    total_groups = len(set(groups))
    # Exact probability a G-draw cluster bootstrap contains only one class.
    # Groups may carry both labels; they are not duplicated independent units.
    exact_degenerate = sum(
        ((total_groups - counts[str(y)]) / total_groups) ** total_groups for y in (0, 1)
    )
    if degenerate_fraction is None:
        degenerate_fraction = exact_degenerate
    if min(counts.values()) < min_groups_per_class:
        reasons.append("TOO_FEW_INDEPENDENT_GROUPS_PER_CLASS")
    if sum(baseline_eers) / len(baseline_eers) < min_improvement:
        reasons.append("BASELINE_BOUND_PRECLUDES_PRESPECIFIED_IMPROVEMENT")
    if degenerate_fraction is not None:
        if not math.isfinite(degenerate_fraction) or not 0 <= degenerate_fraction <= 1:
            raise ValueError("Invalid bootstrap evidence")
        if degenerate_fraction > max_degenerate_fraction:
            reasons.append("DEGENERATE_GROUP_BOOTSTRAP")
    return {
        "status": "INSUFFICIENT_DISCRIMINATION" if reasons else "ELIGIBLE_FOR_H1",
        "reasons": reasons,
        "groups_per_label": counts,
        "role": role,
        "test_used": False,
        "min_groups_per_class": min_groups_per_class,
        "min_improvement": min_improvement,
        "bootstrap_degenerate_fraction": degenerate_fraction,
        "exact_class_degenerate_probability": exact_degenerate,
        "source_mapping_note": "Counts presume supplied connected groups are certified; unknown mapping is a separate blocking gate",
    }
