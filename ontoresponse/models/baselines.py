"""Same-cache ordinary fusion; all strong models retain masks and probe metadata."""

from __future__ import annotations

import torch
from torch import nn

from reference.core import FusionOutput, ResponseTokenizer, _check_input

BASELINES = (
    "score_lr",
    "score_mlp",
    "original_mlp",
    "tta_stats",
    "all_views_mlp",
    "response_mlp",
    "no_delta_mlp",
)


class OrdinaryFusion(nn.Module):
    def __init__(
        self,
        dims,
        families,
        probe_feature_dim,
        mode="response_mlp",
        width=64,
        dropout=0.1,
    ):
        super().__init__()
        self.mode, self.e, self.k = mode, len(dims), len(families)
        if mode not in BASELINES:
            raise ValueError("Unknown baseline")
        self.dims = tuple(dims)
        self.pdim = probe_feature_dim
        self.tokenizer = None
        # Metadata is fixed by registry and exposed to every all-view baseline.
        metadata = self.k * probe_feature_dim + self.e
        if mode.startswith("score_"):
            n = 2 * self.e + 4
        elif mode == "tta_stats":
            n = self.e * 5 + metadata + 4
        elif mode == "original_mlp":
            n = sum(dims) + 2 * self.e + 4
        elif mode in ("all_views_mlp", "no_delta_mlp"):
            n = (self.k + 1) * (sum(dims) + 2 * self.e) + metadata + 4
        else:
            self.tokenizer = ResponseTokenizer(dims, self.k, probe_feature_dim, width)
            n = self.e * (self.k + 1) * (width + 1) + 4
        self.head = (
            nn.Linear(n, 1)
            if mode == "score_lr"
            else nn.Sequential(
                nn.Linear(n, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, 1),
            )
        )

    def forward(self, hidden, scores, valid, probe_features, quality):
        _check_input(hidden, scores, valid)
        if (
            len(hidden) != self.e
            or scores.shape[2] != self.k + 1
            or probe_features.shape != (self.k, self.pdim)
        ):
            raise ValueError("Fixed registry input mismatch")
        if quality.shape != (len(scores), 4) or not torch.isfinite(quality).all():
            raise ValueError("Invalid quality")
        clean = [
            torch.where(valid[:, i, :, None], h, torch.zeros_like(h))
            for i, h in enumerate(hidden)
        ]
        s = torch.where(valid, scores, torch.zeros_like(scores))
        m = valid.to(scores.dtype)
        metadata = torch.cat(
            [
                probe_features.flatten()[None].expand(len(scores), -1),
                torch.arange(self.e, device=s.device, dtype=s.dtype)[None].expand(
                    len(s), -1
                ),
            ],
            1,
        )
        if self.mode.startswith("score_"):
            x = torch.cat([s[:, :, 0], m[:, :, 0], quality], 1)
        elif self.mode == "original_mlp":
            x = torch.cat(
                [*[h[:, 0] for h in clean], s[:, :, 0], m[:, :, 0], quality], 1
            )
        elif self.mode == "tta_stats":
            mean = s.sum(2) / m.sum(2).clamp_min(1)
            variance = torch.where(
                valid, (s - mean[:, :, None]) ** 2, torch.zeros_like(s)
            ).sum(2) / m.sum(2).clamp_min(1)
            high = torch.where(valid, s, torch.full_like(s, -torch.inf)).max(2).values
            low = torch.where(valid, s, torch.full_like(s, torch.inf)).min(2).values
            high = torch.where(valid.any(2), high, torch.zeros_like(high))
            low = torch.where(valid.any(2), low, torch.zeros_like(low))
            x = torch.cat([mean, variance, low, high, m.mean(2), metadata, quality], 1)
        elif self.tokenizer is not None:
            z, mask = self.tokenizer(hidden, scores, valid, probe_features)
            x = torch.cat([z.flatten(1), mask.to(z.dtype), quality], 1)
        else:
            x = torch.cat(
                [
                    *[h.flatten(1) for h in clean],
                    s.flatten(1),
                    m.flatten(1),
                    metadata,
                    quality,
                ],
                1,
            )
        return FusionOutput(self.head(x).squeeze(-1), valid[:, :, 0].any(1))
