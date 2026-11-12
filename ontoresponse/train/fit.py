from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from reference.core import ExpertStats, source_weighted_bce

from ..common import atomic_json, budget, code_hash, read_jsonl, sha256, write_jsonl
from ..evaluation.metrics import metrics
from ..evidence.cache import Cache
from ..evidence.dataset import load_batch, origin_weights
from ..models.factory import build_model, matched_width, parameter_count
from ..ontology.registry import OntologyRegistry, validate_evidence


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def forward(model, batch, stats, reg):
    expected = getattr(model, "expert_identities", None)
    if expected is not None and any(
        name not in expected or expected[name] != identity
        for name, identity in batch.identities.items()
    ):
        raise ValueError("Frozen expert measurement identity differs from training")
    hidden, scores = stats.transform(*batch.inputs())
    return model(hidden, scores, batch.valid, reg.features, batch.quality)


def restore(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    reg = OntologyRegistry()
    if checkpoint["registry_hash"] != reg.digest:
        raise ValueError("Model/registry drift")
    model = build_model(
        checkpoint["dims"],
        reg,
        checkpoint["model_name"],
        checkpoint["width"],
        checkpoint["seed"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.expert_identities = checkpoint["expert_identities"]
    return checkpoint, model, checkpoint["stats"], reg


def fit(
    cfg,
    index,
    manifest,
    model_name="response_mlp",
    seed=17,
    steps=20,
    output=None,
    width=64,
    capacity="fixed",
    resume=None,
):
    ledger = read_jsonl(manifest)
    budget(cfg["runtime"]["budget"], len({r["origin_id"] for r in ledger}), steps)
    if steps < 1:
        raise ValueError("Training requires >=1 step")
    reg = validate_evidence(cfg, read_jsonl(index), Cache(cfg["cache"]))
    train = load_batch(cfg, index, manifest, "fusion_train")
    val = load_batch(
        cfg, index, manifest, "fusion_val", [h.shape[-1] for h in train.hidden]
    )
    if not train.valid[:, :, 0].any(1).all() or not val.valid[:, :, 0].any(1).all():
        raise ValueError(
            "Training/selection requests contain missing all experts; audit failures explicitly"
        )
    if set(train.labels.tolist()) != {0.0, 1.0} or set(val.labels.tolist()) != {
        0.0,
        1.0,
    }:
        raise ValueError("Both classes needed in train/selection")
    dims = [h.shape[-1] for h in train.hidden]
    set_seed(seed)
    if capacity == "matched":
        target = parameter_count(build_model(dims, reg, "typed_response", 64, seed))
        width = matched_width(dims, reg, model_name, target, seed)
    set_seed(seed)
    stats = ExpertStats.fit(*train.inputs(), role="fusion_train")
    model = build_model(dims, reg, model_name, width, seed)
    model.expert_identities = train.identities
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    start = 0
    if resume:
        cp = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            cp["train_ids"] != train.ids
            or cp["model_name"] != model_name
            or cp["seed"] != seed
            or cp["width"] != width
        ):
            raise ValueError("Resume identity mismatch")
        if cp["manifest_sha256"] != sha256(manifest) or cp["index_sha256"] != sha256(
            index
        ):
            raise ValueError("Resume evidence mismatch")
        model.load_state_dict(cp["last_model_state"])
        optimizer.load_state_dict(cp["optimizer_state"])
        stats = cp["stats"]
        torch.set_rng_state(cp["rng_state"])
        start = cp["last_step"]
        if start >= steps:
            raise ValueError("Resume total step limit must exceed saved step count")
    root = Path(
        output or Path(cfg["output"]) / "runs" / f"{model_name}-{capacity}-seed{seed}"
    )
    if (root / "freeze.json").exists():
        raise ValueError("Frozen run is immutable; use a new experiment directory")
    if (root / "model.pt").exists() and resume is None:
        raise ValueError(
            "Existing run must not be overwritten; use explicit resume or a new directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    weights = origin_weights(train.origins)
    best = None
    history = []
    start_time = time.monotonic()
    if resume:
        best = (
            tuple(cp["best_key"]),
            cp["model_state"],
            cp["best_step"],
            cp["val_metrics"],
        )
        history = cp["history"]
    for step in range(start, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = forward(model, train, stats, reg)
        loss = source_weighted_bce(out, train.labels, weights)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            pred = forward(model, val, stats, reg)
            val_loss = float(
                source_weighted_bce(pred, val.labels, origin_weights(val.origins))
            )
            measured = metrics(
                val.labels.numpy(), pred.logits.numpy(), pred.probabilities().numpy()
            )
        key = (measured["eer"], val_loss)
        if best is None or key < best[0]:
            best = (
                key,
                {k: v.detach().clone() for k, v in model.state_dict().items()},
                step + 1,
                measured,
            )
        history.append(
            {
                "step": step + 1,
                "train_bce": float(loss.detach()),
                "val_bce": val_loss,
                "val_eer": measured["eer"],
            }
        )
    cp = {
        "schema": "ontoresponse-fusion-1",
        "mode": cfg["mode"],
        "model_name": model_name,
        "width": width,
        "capacity": capacity,
        "dims": dims,
        "seed": seed,
        "model_state": best[1],
        "last_model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "last_step": steps,
        "rng_state": torch.get_rng_state(),
        "stats": stats,
        "registry_hash": reg.digest,
        "probe_features": reg.features,
        "expert_specs": cfg["experts"],
        "crop": cfg["crop"],
        "expert_identities": train.identities,
        "index_sha256": sha256(index),
        "manifest_sha256": sha256(manifest),
        "train_ids": train.ids,
        "val_ids": val.ids,
        "selection_rule": "min fusion_val EER; tie min source-weighted BCE; no test used",
        "best_key": best[0],
        "best_step": best[2],
        "val_metrics": best[3],
        "history": history,
        "code_sha256": code_hash(),
    }
    atomic_torch(root / "model.pt", cp)
    # Roundtrip must reproduce selected eval outputs on the same actual evidence.
    _, loaded, loaded_stats, loaded_reg = restore(root / "model.pt")
    model.load_state_dict(best[1])
    model.eval()
    with torch.inference_mode():
        expected = forward(model, val, stats, reg)
        actual = forward(loaded, val, loaded_stats, loaded_reg)
        torch.testing.assert_close(expected.logits, actual.logits, rtol=0, atol=0)
    result = {
        "status": "trained_smoke",
        "level": "real_data_integration_pretrained"
        if cfg["mode"] == "integration_pretrained"
        else "research_holdout",
        "model": model_name,
        "seed": seed,
        "parameters": parameter_count(model),
        "width": width,
        "capacity": capacity,
        "optimizer_steps": steps,
        "best_step": best[2],
        "val_metrics": best[3],
        "train_requests": len(train.ids),
        "val_requests": len(val.ids),
        "train_origins": len(set(train.origins)),
        "train_version_weights": weights.tolist(),
        "duration_seconds": time.monotonic() - start_time,
        "checkpoint": str(root / "model.pt"),
        "checkpoint_sha256": sha256(root / "model.pt"),
        "save_load_exact": True,
        "selection_rule": cp["selection_rule"],
        "budget": "same lr=0.001, AdamW, <=20 full-batch steps; single prespecified width or count-matched width",
        "limitation": "bounded pipeline check; dev overlaps public/local expert selection; no generalization claim",
    }
    atomic_json(root / "result.json", result)
    write_jsonl(root / "history.jsonl", history)
    return result


def ablate(cfg, index, manifest, model_name, seed, steps, output=None):
    names = [
        "typed_response",
        "untyped_response",
        "set_attention",
        "wrong_family",
        "drop_anchor",
        "drop_same_probe",
        "drop_same_family",
        "no_response",
        "no_delta_mlp",
        "missing_probe",
    ]
    names += [f"single_expert_{i}" for i in range(len(cfg["experts"]))]
    results = []
    for name in names:
        dest = Path(output) / name if output else None
        results.append(fit(cfg, index, manifest, name, seed, steps, dest))
    return {"status": "ablation_smoke", "retrained": True, "results": results}
