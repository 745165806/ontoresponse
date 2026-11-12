"""Score-free, whole-component role allocation under explicit class/group quotas.

The integer solver is used only to prepare dataset roles, never for detection.
It receives capacities and stable hash costs; no embeddings, logits or outcomes.
"""

import math
from collections import Counter, defaultdict

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..common import Blocked, digest


def allocate_connected(
    candidates,
    components,
    excluded,
    role_targets,
    minimum=5,
    seed=271828,
    time_limit=30,
):
    if (
        not role_targets
        or minimum < 2
        or any(
            type(n) is not int or n < 2 * minimum or n % 2
            for n in role_targets.values()
        )
    ):
        raise ValueError(
            "Even balanced per-role origin budgets and >=2 groups/class required"
        )
    if len({r["id"] for r in candidates}) != len(candidates) or len(
        {r["origin_id"] for r in candidates}
    ) != len(candidates):
        raise ValueError("One unique candidate per target origin is required")
    if any(
        r["id"] != r["origin_id"]
        or r["origin_id"] not in components
        or r["label"] not in (0, 1)
        for r in candidates
    ):
        raise ValueError("Unmapped/noncanonical candidate or invalid label")
    groups = defaultdict(lambda: {0: [], 1: []})
    excluded_n = 0
    for row in candidates:
        group = components[row["origin_id"]]
        if group in excluded:
            excluded_n += 1
            continue
        groups[group][row["label"]].append(row)
    keys = sorted(groups, key=lambda g: digest([seed, "component", g]))
    roles = sorted(role_targets)
    totals = {str(y): sum(len(groups[g][y]) for g in keys) for y in (0, 1)}
    label_groups = {str(y): sum(bool(groups[g][y]) for g in keys) for y in (0, 1)}
    report = {
        "status": "PENDING",
        "seed": seed,
        "candidate_origins": len(candidates),
        "excluded_candidate_origins": excluded_n,
        "eligible_components": len(keys),
        "eligible_origins_by_class": totals,
        "eligible_components_by_class": label_groups,
        "role_targets": role_targets,
        "minimum_groups_per_class": minimum,
        "test_scores_read": False,
        "source_mapping_scope": "Requires separately certified full-universe components, never speaker-only fallback",
        "class_balance": "equal origins per class in each role",
        "solver": "scipy.optimize.milp",
        "scipy_version": scipy.__version__,
        "reasons": [],
    }
    needed = sum(role_targets.values()) // 2
    for y in (0, 1):
        if totals[str(y)] < needed:
            report["reasons"].append(f"INSUFFICIENT_ORIGINS_CLASS_{y}")
        if label_groups[str(y)] < minimum * len(roles):
            report["reasons"].append(f"INSUFFICIENT_DISJOINT_COMPONENTS_CLASS_{y}")
    if report["reasons"]:
        report["status"] = "BLOCKED_INSUFFICIENT_COMPONENTS"
        return [], report
    # Each component/role has z, two class-participation binaries b_y, and two integer counts n_y.
    variables = {}
    upper = []
    cost = []

    def add(key, cap, obj):
        variables[key] = len(upper)
        upper.append(cap)
        cost.append(obj)

    for group in keys:
        for role in roles:
            tie = int(digest([seed, "allocation", group, role])[:12], 16) / 16**12
            # First maximize class-bearing independent groups under the budget;
            # total hash tie cost is < 1/2, so it cannot trade away one group.
            add((group, role, "z"), 1, tie / (2 * len(keys) + 1))
            for y in (0, 1):
                cap = min(
                    len(groups[group][y]),
                    math.ceil((role_targets[role] // 2) / minimum),
                )
                add((group, role, "b", y), int(cap > 0), -1)
                add((group, role, "n", y), cap, 0)
    rr = []
    cc = []
    data = []
    lower = []
    higher = []

    def constraint(terms, lo, hi):
        idx = len(lower)
        lower.append(lo)
        higher.append(hi)
        for key, value in terms:
            rr.append(idx)
            cc.append(variables[key])
            data.append(value)

    for group in keys:
        constraint([((group, role, "z"), 1) for role in roles], 0, 1)
        for role in roles:
            for y in (0, 1):
                cap = upper[variables[(group, role, "n", y)]]
                constraint(
                    [((group, role, "b", y), 1), ((group, role, "z"), -1)], -np.inf, 0
                )
                constraint(
                    [((group, role, "n", y), 1), ((group, role, "b", y), -cap)],
                    -np.inf,
                    0,
                )
                constraint(
                    [((group, role, "n", y), 1), ((group, role, "b", y), -1)], 0, np.inf
                )
    for role in roles:
        for y in (0, 1):
            constraint(
                [((g, role, "n", y), 1) for g in keys],
                role_targets[role] // 2,
                role_targets[role] // 2,
            )
            constraint([((g, role, "b", y), 1) for g in keys], minimum, np.inf)
    matrix = coo_matrix((data, (rr, cc)), shape=(len(lower), len(upper))).tocsc()
    solved = milp(
        np.array(cost),
        integrality=np.ones(len(upper)),
        bounds=Bounds(np.zeros(len(upper)), upper),
        constraints=LinearConstraint(matrix, lower, higher),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    report.update(solver_status=int(solved.status), solver_message=str(solved.message))
    if solved.status != 0:
        report["status"] = (
            "BLOCKED_INFEASIBLE"
            if solved.status == 2
            else "BLOCKED_ALLOCATION_UNRESOLVED"
        )
        report["reasons"].append(
            "No optimal allocation certified; no fallback or score-based resplit"
        )
        return [], report
    values = np.rint(solved.x)
    if (
        np.max(np.abs(solved.x - values)) > 1e-5
        or (matrix @ values < np.array(lower) - 1e-6).any()
        or (matrix @ values > np.array(higher) + 1e-6).any()
    ):
        raise Blocked("Integer allocation failed independent constraint verification")
    # Given the certified role/group membership, round-robin recording quotas
    # avoid concentrating nearly all origins in just a few active components.
    for role in roles:
        for y in (0, 1):
            active = [g for g in keys if values[variables[(g, role, "b", y)]]]
            active.sort(key=lambda g: digest([seed, "within_role", role, y, g]))
            for group in active:
                values[variables[(group, role, "n", y)]] = 1
            remaining = role_targets[role] // 2 - len(active)
            while remaining:
                progressed = False
                for group in active:
                    idx = variables[(group, role, "n", y)]
                    if values[idx] < upper[idx]:
                        values[idx] += 1
                        remaining -= 1
                        progressed = True
                        if not remaining:
                            break
                if not progressed:
                    raise Blocked(
                        "Round-robin quota disagrees with feasibility witness"
                    )
    if (matrix @ values < np.array(lower) - 1e-6).any() or (
        matrix @ values > np.array(higher) + 1e-6
    ).any():
        raise Blocked("Round-robin allocation failed constraint verification")
    selected = []
    ownership = {}
    for group in keys:
        for role in roles:
            if values[variables[(group, role, "z")]]:
                ownership[group] = role
            for y in (0, 1):
                n = int(values[variables[(group, role, "n", y)]])
                available = sorted(
                    groups[group][y],
                    key=lambda r: digest([seed, "recording", r["origin_id"]]),
                )
                selected.extend(
                    dict(r, role=role, source_group=group) for r in available[:n]
                )
    selected.sort(key=lambda r: (r["role"], digest([seed, r["id"]])))
    report.update(
        status="ALLOCATED_METADATA_ONLY",
        component_ownership=ownership,
        selected_origin_hash=digest([r["origin_id"] for r in selected]),
        precision=split_precision(selected),
        selection="maximize class-bearing independent groups, then stable hash tie costs; whole-component roles, class/group caps and within-role class round-robin recording quotas; no scores",
        reproducibility="Pin SciPy/solver and actual allocated manifest hash; no claim of identical optimal ties across solver versions",
    )
    return selected, report


def split_precision(rows):
    result = {}
    for role in sorted({r["role"] for r in rows}):
        part = [r for r in rows if r["role"] == role]
        all_groups = {r["source_group"] for r in part}
        g = len(all_groups)
        counts = {
            y: Counter(r["source_group"] for r in part if r["label"] == y)
            for y in (0, 1)
        }
        summary = {}
        for y in (0, 1):
            n = sum(counts[y].values())
            sizes = list(counts[y].values())
            summary[str(y)] = {
                "origins": n,
                "groups": len(sizes),
                "maximum_group_fraction": max(sizes) / n,
                "kish_group_count_descriptive_only": n * n / sum(c * c for c in sizes),
            }
        result[role] = {
            "classes": summary,
            "groups": g,
            "exact_class_degenerate_bootstrap_probability": sum(
                ((g - len(counts[y])) / g) ** g for y in (0, 1)
            ),
            "iid_zero_FPR_one_sided_95pct_upper": 1
            - 0.05 ** (1 / summary["0"]["origins"]),
            "precision_status": "NOT_POWERED_FROM_METADATA_ONLY",
            "warning": "Kish and IID bound are descriptive, not EER power/cluster CI; independent source groups and real development score variability still required",
        }
    return result
