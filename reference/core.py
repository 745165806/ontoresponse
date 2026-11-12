"""Minimal direct-classification reference, not a complete audio detector.

Input contains only already-extracted embeddings, spoof-oriented margins,
validity, measured quality, and known probe metadata. Labels enter only the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RELATIONS = (
    "anchor_to_response",
    "response_to_anchor",
    "same_probe_cross_expert",
    "same_family_within_expert",
)


def compile_relations(n_experts: int, families: Sequence[str | int]) -> Tensor:
    """A[r, target, source]. Family names are used only through equality."""
    if n_experts < 1 or not families:
        raise ValueError("At least one expert and probe are required")
    k = len(families)
    n = n_experts * (k + 1)
    a = torch.zeros(len(RELATIONS), n, n)

    def r(i: int, p: int) -> int:
        return n_experts + i * k + p

    for i in range(n_experts):
        for p in range(k):
            a[0, r(i, p), i] = 1
            a[1, i, r(i, p)] = 1
            for j in range(n_experts):
                if i != j:
                    a[2, r(i, p), r(j, p)] = 1
            for q in range(k):
                if p != q and families[p] == families[q]:
                    a[3, r(i, p), r(i, q)] = 1
    return a


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Pool node axis 1; invalid values are explicitly zeroed."""
    if mask.dtype != torch.bool or x.shape[:2] != mask.shape:
        raise ValueError("Invalid pool mask")
    clean = torch.where(mask[..., None], x, torch.zeros_like(x))
    return clean.sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


@dataclass
class ExpertStats:
    hidden_means: list[Tensor]
    hidden_scales: list[Tensor]
    score_means: Tensor
    score_scales: Tensor
    fit_role: str = "fusion_train"

    @classmethod
    def fit(
        cls, hidden: Sequence[Tensor], scores: Tensor, valid: Tensor, *, role: str
    ) -> "ExpertStats":
        if role != "fusion_train":
            raise ValueError("Statistics may only be fitted on fusion_train")
        _check_input(hidden, scores, valid)
        means, scales, sm, ss = [], [], [], []
        for i, h in enumerate(hidden):
            m = valid[:, i, 0]
            if m.sum() < 2:
                raise ValueError(
                    "Need at least two valid original observations per expert"
                )
            x, s = h[m, 0], scores[m, i, 0]
            means.append(x.mean(0).detach())
            scales.append(x.std(0, unbiased=False).clamp_min(1e-5).detach())
            sm.append(s.mean().detach())
            ss.append(s.std(unbiased=False).clamp_min(1e-5).detach())
        return cls(means, scales, torch.stack(sm), torch.stack(ss))

    def transform(
        self, hidden: Sequence[Tensor], scores: Tensor, valid: Tensor
    ) -> tuple[list[Tensor], Tensor]:
        _check_input(hidden, scores, valid)
        if len(hidden) != len(self.hidden_means):
            raise ValueError("Expert count differs from fitted statistics")
        out = []
        for i, h in enumerate(hidden):
            z = (h - self.hidden_means[i].to(h)) / self.hidden_scales[i].to(h)
            out.append(torch.where(valid[:, i, :, None], z, torch.zeros_like(z)))
        s = (
            scores - self.score_means.to(scores)[None, :, None]
        ) / self.score_scales.to(scores)[None, :, None]
        return out, torch.where(valid, s, torch.zeros_like(s))


def _check_input(hidden: Sequence[Tensor], scores: Tensor, valid: Tensor) -> None:
    if scores.ndim != 3 or valid.shape != scores.shape or valid.dtype != torch.bool:
        raise ValueError(
            "scores/valid must be [batch, experts, views] with boolean valid"
        )
    if not scores.is_floating_point():
        raise ValueError("Scores must be floating point")
    b, e, v = scores.shape
    if len(hidden) != e or b < 1 or e < 1 or v < 2:
        raise ValueError(
            "Expected one hidden tensor per expert and original + probe views"
        )
    if not torch.isfinite(scores[valid]).all():
        raise ValueError("Nonfinite score marked as valid")
    for i, h in enumerate(hidden):
        if h.ndim != 3 or h.shape[:2] != (b, v) or not h.is_floating_point():
            raise ValueError("Each hidden tensor must be [batch, views, expert_dim]")
        if (
            h.device != scores.device
            or h.dtype != scores.dtype
            or valid.device != scores.device
        ):
            raise ValueError("All tensors must share device and floating dtype")
        if not torch.isfinite(h[valid[:, i]]).all():
            raise ValueError("Nonfinite embedding marked as valid")


class ResponseTokenizer(nn.Module):
    """Original and response tokens using expert-specific adapters."""

    def __init__(
        self,
        expert_dims: Sequence[int],
        n_probes: int,
        probe_feature_dim: int,
        width: int = 64,
    ):
        super().__init__()
        if not expert_dims or min(expert_dims) < 1 or n_probes < 1:
            raise ValueError("Invalid expert dimensions or probe count")
        self.dims = tuple(expert_dims)
        self.e = len(expert_dims)
        self.k = n_probes
        self.probe_feature_dim = probe_feature_dim
        self.width = width
        self.orig = nn.ModuleList(
            [nn.Sequential(nn.Linear(d + 1, width), nn.GELU()) for d in expert_dims]
        )
        # delta_h, original score, probe score, score delta, log norm, probe metadata
        self.resp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d + 4 + probe_feature_dim, width), nn.GELU())
                for d in expert_dims
            ]
        )
        self.expert_id = nn.Embedding(self.e, width)
        self.kind = nn.Embedding(2, width)

    def forward(
        self,
        hidden: Sequence[Tensor],
        scores: Tensor,
        valid: Tensor,
        probe_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _check_input(hidden, scores, valid)
        b, e, v = scores.shape
        if e != self.e or v != self.k + 1:
            raise ValueError("Input differs from fixed model registry")
        if probe_features.shape != (self.k, self.probe_feature_dim):
            raise ValueError("Wrong known-probe feature shape")
        if (
            probe_features.device != scores.device
            or probe_features.dtype != scores.dtype
        ):
            raise ValueError("Probe metadata device/dtype differs")
        if not torch.isfinite(probe_features).all():
            raise ValueError("Nonfinite probe metadata")
        pm = probe_features[None].expand(b, -1, -1)
        originals, responses, r_masks = [], [], []
        for i, h in enumerate(hidden):
            if h.shape[-1] != self.dims[i]:
                raise ValueError("Embedding dimension differs from registry")
            hi = torch.where(valid[:, i, :, None], h, torch.zeros_like(h))
            si = torch.where(valid[:, i], scores[:, i], torch.zeros_like(scores[:, i]))
            oi = self.orig[i](torch.cat([hi[:, 0], si[:, 0, None]], -1))
            oi = oi + self.expert_id.weight[i] + self.kind.weight[0]
            originals.append(oi)
            dh = hi[:, 1:] - hi[:, 0, None]
            s0 = si[:, 0, None, None].expand(-1, self.k, -1)
            sp = si[:, 1:, None]
            mag = torch.log1p(torch.linalg.vector_norm(dh, dim=-1, keepdim=True))
            ri = self.resp[i](torch.cat([dh, s0, sp, sp - s0, mag, pm], -1))
            ri = ri + self.expert_id.weight[i] + self.kind.weight[1]
            responses.append(ri)
            r_masks.append(valid[:, i, 1:] & valid[:, i, 0, None])
        z = torch.cat([torch.stack(originals, 1), torch.cat(responses, 1)], 1)
        mask = torch.cat([valid[:, :, 0], torch.cat(r_masks, 1)], 1)
        return torch.where(mask[..., None], z, torch.zeros_like(z)), mask


class RelationalLayer(nn.Module):
    def __init__(self, width: int, n_relations: int, dropout: float = 0.1):
        super().__init__()
        self.self_map = nn.Linear(width, width, bias=False)
        self.maps = nn.ModuleList(
            [nn.Linear(width, width, bias=False) for _ in range(n_relations)]
        )
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: Tensor, mask: Tensor, adjacency: Tensor) -> Tensor:
        if adjacency.ndim != 3 or adjacency.shape[0] != len(self.maps):
            raise ValueError("Wrong relation count")
        if adjacency.shape[1:] != (z.shape[1], z.shape[1]):
            raise ValueError("Wrong relation shape")
        if not torch.isfinite(adjacency).all() or (adjacency < 0).any():
            raise ValueError("Adjacency must be finite and nonnegative")
        z = torch.where(mask[..., None], z, torch.zeros_like(z))
        msg = self.self_map(z)
        for r, layer in enumerate(self.maps):
            a = adjacency[r][None] * mask[:, None, :].to(z.dtype)
            agg = torch.bmm(a, z) / a.sum(-1, keepdim=True).clamp_min(1)
            msg = msg + layer(agg)
        z = self.norm(z + self.dropout(F.gelu(msg)))
        return torch.where(mask[..., None], z, torch.zeros_like(z))


@dataclass
class FusionOutput:
    logits: Tensor
    observed: Tensor

    def probabilities(self) -> Tensor:
        """NaN means no valid original expert, not a 0.5 probability."""
        p = self.logits.sigmoid()
        return torch.where(self.observed, p, torch.full_like(p, float("nan")))


class EvidenceFusion(nn.Module):
    """typed/untyped graph, all-token MLP, or untyped full self-attention.

    This is a numerical scaffold. It does not include real expert inference,
    probability calibration, official benchmark metrics, or production caches.
    """

    def __init__(
        self,
        expert_dims: Sequence[int],
        families: Sequence[str | int],
        probe_feature_dim: int,
        quality_dim: int = 4,
        width: int = 64,
        layers: int = 2,
        mode: str = "typed",
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in {"typed", "untyped", "none", "flat", "set_attention"}:
            raise ValueError("Unknown fusion mode")
        self.mode, self.e, self.k = mode, len(expert_dims), len(families)
        self.quality_dim = quality_dim
        self.tokenizer = ResponseTokenizer(
            expert_dims, self.k, probe_feature_dim, width
        )
        a = compile_relations(self.e, families)
        if mode == "untyped":
            a = (a.sum(0, keepdim=True) > 0).float()
        elif mode == "none":
            a = a[:0]
        self.register_buffer("adjacency", a)
        self.layers = nn.ModuleList()
        if mode in {"typed", "untyped", "none"}:
            self.layers.extend(
                [RelationalLayer(width, len(a), dropout) for _ in range(layers)]
            )
        elif mode == "set_attention":
            if width % 4:
                raise ValueError("Attention width must be divisible by 4")
            self.layers.extend(
                [
                    nn.TransformerEncoderLayer(
                        d_model=width,
                        nhead=4,
                        dim_feedforward=width * 2,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="gelu",
                    )
                    for _ in range(layers)
                ]
            )
        n = self.e * (self.k + 1)
        read_dim = (
            n * (width + 1) + quality_dim
            if mode == "flat"
            else width * 2 + quality_dim + 2
        )
        self.head = nn.Sequential(
            nn.Linear(read_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        hidden: Sequence[Tensor],
        scores: Tensor,
        valid: Tensor,
        probe_features: Tensor,
        quality: Tensor,
    ) -> FusionOutput:
        z, mask = self.tokenizer(hidden, scores, valid, probe_features)
        if quality.shape != (scores.shape[0], self.quality_dim):
            raise ValueError("Wrong measured quality shape")
        if (
            quality.device != scores.device
            or quality.dtype != scores.dtype
            or not torch.isfinite(quality).all()
        ):
            raise ValueError("Quality must be finite with matching dtype/device")
        if self.mode == "set_attention":
            safe_mask = mask.clone()
            empty = ~safe_mask.any(1)
            safe_mask[empty, 0] = True  # prevent all-masked softmax; remove below
            for layer in self.layers:
                z = layer(z, src_key_padding_mask=~safe_mask)
                z = torch.where(mask[..., None], z, torch.zeros_like(z))
        elif self.mode != "flat":
            for layer in self.layers:
                z = layer(z, mask, self.adjacency)
        if self.mode == "flat":
            pooled = torch.cat([z.flatten(1), mask.to(z.dtype), quality], -1)
        else:
            coverage = torch.stack(
                [mask[:, : self.e].float().mean(1), mask[:, self.e :].float().mean(1)],
                -1,
            ).to(z.dtype)
            pooled = torch.cat(
                [
                    masked_mean(z[:, : self.e], mask[:, : self.e]),
                    masked_mean(z[:, self.e :], mask[:, self.e :]),
                    quality,
                    coverage,
                ],
                -1,
            )
        return FusionOutput(self.head(pooled).squeeze(-1), valid[:, :, 0].any(1))


def source_weighted_bce(
    output: FusionOutput, labels: Tensor, weights: Tensor | None = None
) -> Tensor:
    if labels.shape != output.logits.shape or not torch.isfinite(labels).all():
        raise ValueError("Wrong labels")
    if ((labels != 0) & (labels != 1)).any():
        raise ValueError("Expected binary labels")
    if not output.observed.all():
        raise ValueError("Training batch contains no-evidence rows; audit explicitly")
    losses = F.binary_cross_entropy_with_logits(output.logits, labels, reduction="none")
    if weights is None:
        return losses.mean()
    if (
        weights.shape != losses.shape
        or not torch.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid origin/version weights")
    return (losses * weights).sum() / weights.sum()
