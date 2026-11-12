"""Build fusion models and controlled ablations from the same input evidence."""

from torch import nn

from reference.core import EvidenceFusion

from .baselines import BASELINES, OrdinaryFusion
from .controls import BestSingleExpert, InformationControl, ProbeSubset
from .relation_tied import RelationTiedFusion


class MaskAblation(nn.Module):
    def __init__(
        self, model, keep_original_only=False, missing_probe=None, single_expert=None
    ):
        super().__init__()
        self.model = model
        self.keep_original_only = keep_original_only
        self.missing_probe = missing_probe
        self.single_expert = single_expert

    def forward(self, hidden, scores, valid, probe_features, quality):
        valid = valid.clone()
        if self.keep_original_only:
            valid[:, :, 1:] = False
        if self.missing_probe is not None:
            valid[:, :, self.missing_probe + 1] = False
        if self.single_expert is not None:
            i = self.single_expert
            hidden = [hidden[i]]
            scores = scores[:, i : i + 1]
            valid = valid[:, i : i + 1]
        return self.model(hidden, scores, valid, probe_features, quality)


def build_model(dims, reg, name, width=64, seed=17):
    """Select a named model; keep legacy names compatible with saved checkpoints."""
    if name == "best_single_expert":
        return BestSingleExpert()
    if name.endswith("_p3"):
        return ProbeSubset(build_model(dims, reg, name[:-3], width, seed))
    if name in ("no_embedding_response", "no_score_response", "identity_copy_views"):
        return InformationControl(
            build_model(dims, reg, "typed_response", width, seed), name
        )
    if name == "relation_tied":
        return RelationTiedFusion(
            dims, reg.families, reg.features.shape[1], width=width
        )
    if name in BASELINES:
        return OrdinaryFusion(dims, reg.families, reg.features.shape[1], name, width)
    single = int(name.split("_")[-1]) if name.startswith("single_expert_") else None
    use_dims = [dims[single]] if single is not None else dims
    modes = {
        "untyped_response": "untyped",
        "untyped_union": "untyped",
        "set_attention": "set_attention",
        "full_attention": "set_attention",
    }
    mode = modes.get(name, "typed")
    allowed = {
        "typed_response",
        "untyped_response",
        "untyped_union",
        "set_attention",
        "full_attention",
        "wrong_family",
        "no_response",
        "missing_probe",
        "drop_anchor",
        "drop_same_probe",
        "drop_same_family",
    }
    if name not in allowed and single is None:
        raise ValueError("Unknown model " + name)
    model = EvidenceFusion(
        use_dims,
        reg.families,
        reg.features.shape[1],
        quality_dim=4,
        width=width,
        mode=mode,
    )
    if name == "wrong_family":
        model.adjacency.copy_(reg.wrong_members(len(dims), seed))
    if name == "drop_anchor":
        model.adjacency[:2] = 0
    if name == "drop_same_probe":
        model.adjacency[2] = 0
    if name == "drop_same_family":
        model.adjacency[3] = 0
    if name in ("no_response", "missing_probe") or single is not None:
        model = MaskAblation(
            model, name == "no_response", 0 if name == "missing_probe" else None, single
        )
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_width(dims, reg, name, target, seed=17):
    # Capacity matching is based on dimensions/parameter count, without reading labels or scores.
    candidates = range(8, 257, 4)
    results = []
    for width in candidates:
        model = build_model(dims, reg, name, width, seed)
        results.append((abs(parameter_count(model) - target), width))
    return min(results)[1]
