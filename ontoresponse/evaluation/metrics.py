"""Detection metrics and paired uncertainty, with spoof-oriented scores."""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve


def _eer(labels, scores):
    """Linearly cross the ROC after grouping tied scores; inputs are validated upstream."""
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    d = fpr - (1 - tpr)
    j = np.flatnonzero(d >= 0)[0]
    if j == 0:
        return float(fpr[0])
    alpha = -d[j - 1] / (d[j] - d[j - 1])
    return float(fpr[j - 1] + alpha * (fpr[j] - fpr[j - 1]))


def metrics(labels, scores, probabilities=None, threshold=None):
    """Report utterance metrics; higher scores indicate spoof, labels are 0/1."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or not len(s) or not np.isfinite(s).all() or set(y) != {0, 1}:
        raise ValueError("Metrics need finite scores and both classes")
    out = {
        "n": len(y),
        "bonafide": int((y == 0).sum()),
        "spoof": int((y == 1).sum()),
        "eer": _eer(y, s),
        "eer_method": "ROC_linear_interpolation_grouped_ties",
        "auroc": float(roc_auc_score(y, s)),
        "spoof_auprc": float(average_precision_score(y, s)),
    }
    if probabilities is not None:
        prob = np.asarray(probabilities)
        out.update(
            nll=float(log_loss(y, prob, labels=[0, 1])),
            brier=float(np.mean((prob - y) ** 2)),
        )
    if threshold is not None:
        pred = s >= threshold
        out.update(
            threshold=float(threshold),
            actual_fpr=float(pred[y == 0].mean()),
            actual_tpr=float(pred[y == 1].mean()),
            actual_fnr=float((~pred[y == 1]).mean()),
        )
    return out


def choose_threshold(labels, probabilities, role, target_fpr=0.01):
    if role not in ("fusion_val", "threshold"):
        raise ValueError("Operating threshold requires development data")
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    if set(y) != {0, 1} or not np.isfinite(p).all():
        raise ValueError("Invalid threshold observations")
    candidates = np.unique(np.r_[p, np.nextafter(p.max(), np.inf)])
    allowed = [x for x in candidates if (p[y == 0] >= x).mean() <= target_fpr]
    threshold = float(min(allowed))
    return {
        "threshold": threshold,
        "role": role,
        "target_dev_fpr": target_fpr,
        "actual_dev_fpr": float((p[y == 0] >= threshold).mean()),
        "n_dev_bonafide": int((y == 0).sum()),
        "low_fpr_estimate_supported": int((y == 0).sum()) >= int(1 / target_fpr),
    }


def paired_bootstrap(labels, score_a, score_b, groups, seed=17, replicates=2000):
    y = np.asarray(labels)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    g = np.asarray(groups)
    if not (len(y) == len(a) == len(b) == len(g)):
        raise ValueError("Pairing length mismatch")
    unique = np.unique(g)
    rng = np.random.default_rng(seed)
    deltas = []
    degenerate = 0
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(g == v) for v in sampled])
        if len(np.unique(y[ix])) < 2:
            degenerate += 1
            continue
        deltas.append(metrics(y[ix], a[ix])["eer"] - metrics(y[ix], b[ix])["eer"])
    return {
        "metric": "EER(a)-EER(b)",
        "independent_groups": len(unique),
        "replicates": replicates,
        "valid_replicates": len(deltas),
        "degenerate_replicates": degenerate,
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
    }


def paired_macro_bootstrap(
    labels, score_a, score_b, groups, conditions, seed=17, replicates=2000
):
    y = np.asarray(labels)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    g = np.asarray(groups)
    c = np.asarray(conditions)
    if not (len(y) == len(a) == len(b) == len(g) == len(c)):
        raise ValueError("Pairing length mismatch")
    names = sorted(set(c) - {"no_new_processing"})
    if not names:
        raise ValueError("No prespecified unseen conditions")
    unique = np.unique(g)
    rng = np.random.default_rng(seed)
    deltas = []
    degenerate = 0

    for _ in range(replicates):
        ix = np.concatenate(
            [
                np.flatnonzero(g == v)
                for v in rng.choice(unique, len(unique), replace=True)
            ]
        )
        parts = [ix[c[ix] == name] for name in names]
        if any(len(np.unique(y[part])) < 2 for part in parts):
            degenerate += 1
            continue
        deltas.append(
            float(
                np.mean(
                    [_eer(y[part], a[part]) - _eer(y[part], b[part]) for part in parts]
                )
            )
        )
    return {
        "metric": "macro_EER(a)-macro_EER(b)",
        "conditions": names,
        "independent_groups": len(unique),
        "replicates": replicates,
        "valid_replicates": len(deltas),
        "degenerate_replicates": degenerate,
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
    }
