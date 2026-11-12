"""Explicit pilot optimizer budget, grouped ledgers and convergence history.

Never called by smoke fit; never overwrites an existing experiment. Test tensors
are only loaded for scoring after model/calibration/operating-point locks exist.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from reference.core import ExpertStats, source_weighted_bce

from ..common import (
    Blocked,
    atomic_json,
    code_hash,
    digest,
    read_jsonl,
    sha256,
    write_jsonl,
)
from ..data.pilot import check_pilot_budget
from ..evaluation.metrics import metrics
from ..evidence.cache import Cache
from ..evidence.dataset import load_batch, origin_weights
from ..models.factory import build_model, parameter_count
from ..ontology.registry import validate_evidence
from ..train.calibrate import Platt
from .development import (
    epoch_schedule,
    h1_decision,
    require_role,
    select_family_trials,
    validation_metrics,
)
from .fit import atomic_torch, forward, set_seed


def subset(batch, indices):
    ix = indices.tolist()
    return replace(
        batch,
        ids=[batch.ids[i] for i in ix],
        hidden=[h[indices] for h in batch.hidden],
        scores=batch.scores[indices],
        valid=batch.valid[indices],
        quality=batch.quality[indices],
        labels=batch.labels[indices],
        origins=[batch.origins[i] for i in ix],
        groups=[batch.groups[i] for i in ix],
        audit=[batch.audit[i] for i in ix],
    )


def operating_point(labels, calibrated_logits, role, target_fpr):
    if role != "fusion_val" or not 0 < target_fpr < 1:
        raise ValueError("Development-only operating point")
    y = np.asarray(labels)
    z = np.asarray(calibrated_logits, dtype=np.float64)
    if set(y) != {0, 1} or not np.isfinite(z).all():
        raise ValueError("Invalid threshold data")
    candidates = np.unique(np.r_[z, np.nextafter(z.max(), np.inf)])
    threshold = min(v for v in candidates if (z[y == 0] >= v).mean() <= target_fpr)
    return {
        "space": "calibrated_logit_float64",
        "comparison": ">=",
        "threshold": float(threshold),
        "role": role,
        "target_dev_fpr": target_fpr,
        "actual_dev_fpr": float((z[y == 0] >= threshold).mean()),
        "n_dev_bonafide": int((y == 0).sum()),
        "warning": "pilot empirical operating point; small development counts limit FPR precision",
    }


def fit_one(cfg, name, seed, train, val, reg):
    p = cfg["pilot"]
    require_role(train, "fusion_train")
    require_role(val, "fusion_val")
    suffix = "-" + p["trial_id"] if p.get("trial_id") else ""
    out = Path(cfg["output"]) / "runs" / f"{name}-seed{seed}{suffix}"
    if out.exists():
        raise ValueError("Do not overwrite pilot runs")
    out.mkdir(parents=True)
    set_seed(seed)
    dims = [h.shape[-1] for h in train.hidden]
    model = build_model(dims, reg, name, p["width"], seed)
    model.expert_identities = train.identities
    stats = ExpertStats.fit(*train.inputs(), role="fusion_train")
    optimizer = (
        None
        if name == "best_single_expert"
        else torch.optim.AdamW(
            model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
        )
    )
    history = []
    best = None
    steps = 0
    stale = 0
    stop = "max_epochs"
    if name == "best_single_expert":
        candidates = []
        for i in range(len(dims)):
            mask = val.valid[:, i, 0]
            measured = validation_metrics(
                val, val.scores[:, i, 0], p.get("validation_eer_conditions"), mask
            )
            candidates.append((measured["eer"], i))
        model.chosen_expert.fill_(min(candidates)[1])
    epochs = 1 if optimizer is None else p["max_epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        order = epoch_schedule(train, seed, epoch)
        loss_sum = 0.0
        batches = 0
        gradient_norms = []
        for start in [] if optimizer is None else range(0, len(order), p["batch_size"]):
            if steps >= p["max_optimizer_steps"]:
                break
            batch = subset(train, order[start : start + p["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            result = forward(model, batch, stats, reg)
            loss = source_weighted_bce(
                result, batch.labels, origin_weights(batch.origins)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if not torch.isfinite(grad):
                raise ValueError("Nonfinite gradient norm")
            gradient_norms.append(float(grad))
            optimizer.step()
            steps += 1
            loss_sum += float(loss.detach())
            batches += 1
        model.eval()
        with torch.inference_mode():
            a = forward(model, train, stats, reg)
            b = forward(model, val, stats, reg)
            train_loss = float(
                source_weighted_bce(a, train.labels, origin_weights(train.origins))
            )
            val_loss = float(
                source_weighted_bce(b, val.labels, origin_weights(val.origins))
            )
            val_metric = validation_metrics(
                val, b.logits, p.get("validation_eer_conditions"), b.observed
            )
        improved = best is None or val_loss < best["val_bce"] - p.get("min_delta", 1e-6)
        if improved:
            best = {
                "epoch": epoch,
                "val_bce": val_loss,
                "val_eer": val_metric["eer"],
                "val_metrics": val_metric,
                "state": copy.deepcopy(model.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps,
                "optimization_bce": loss_sum / max(1, batches),
                "train_eval_bce": train_loss,
                "val_bce": val_loss,
                "val_eer": val_metric["eer"],
                "best_epoch": best["epoch"],
                "validation_metrics": val_metric,
                "selected_train_origins": len(order),
                "train_measurement_count": len(train.ids),
                "epoch_schedule_sha256": digest([train.ids[i] for i in order.tolist()]),
                "learning_rate": p["learning_rate"],
                "gradient_norm_max": max(gradient_norms, default=0.0),
                "gradient_norm_mean": float(np.mean(gradient_norms))
                if gradient_norms
                else 0.0,
            }
        )
        write_jsonl(out / "convergence.jsonl", history)
        if epoch >= p["min_epochs"] and stale >= p["patience_epochs"]:
            stop = "validation_patience"
            break
        if steps >= p["max_optimizer_steps"]:
            stop = "max_optimizer_steps"
            break
    checkpoint = {
        "schema": "ontoresponse-fusion-1",
        "mode": "integration_pretrained",
        "claim_scope": "integration_pilot_only",
        "model_name": name,
        "dims": dims,
        "width": p["width"],
        "seed": seed,
        "model_state": best["state"],
        "stats": stats,
        "registry_hash": reg.digest,
        "expert_specs": cfg["experts"],
        "expert_identities": train.identities,
        "crop": cfg["crop"],
        "last_model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "rng_state": torch.get_rng_state(),
        "last_step": steps,
        "history": history,
        "train_ids": train.ids,
        "val_ids": val.ids,
        "best_epoch": best["epoch"],
        "best_step": history[best["epoch"] - 1]["optimizer_steps"],
        "selection_rule": "min fusion_val BCE within fixed convergence budget",
        "code_sha256": code_hash(),
        "pilot_budget": p,
        "ledger_sha256": sha256(Path(cfg["output"]) / "audit_manifest.jsonl"),
    }
    atomic_torch(out / "model.pt", checkpoint)
    model.load_state_dict(best["state"])
    model.eval()
    from .fit import restore

    _, restored, loaded_stats, loaded_reg = restore(out / "model.pt")
    with torch.inference_mode():
        torch.testing.assert_close(
            forward(model, val, stats, reg).logits,
            forward(restored, val, loaded_stats, loaded_reg).logits,
            atol=0,
            rtol=0,
        )
    result = {
        "model": name,
        "seed": seed,
        "parameters": parameter_count(model),
        "epochs": len(history),
        "steps": steps,
        "best_epoch": best["epoch"],
        "val_bce": best["val_bce"],
        "val_eer": best["val_eer"],
        "val_metrics": best["val_metrics"],
        "trial_id": p.get("trial_id", "default"),
        "width": p["width"],
        "learning_rate": p["learning_rate"],
        "stop_reason": stop,
        "convergence_claim": stop == "validation_patience",
        "checkpoint": str(out / "model.pt"),
        "checkpoint_sha256": sha256(out / "model.pt"),
        "save_load_exact": True,
    }
    atomic_json(out / "result.json", result)
    return result


def run_training(cfg):
    from ..pilot_gate import require_pass

    require_pass(cfg)
    check_pilot_budget(cfg)
    root = Path(cfg["output"])
    manifest = root / "audit_manifest.jsonl"
    index = root / "extraction-0-of-1.jsonl"
    reg = validate_evidence(cfg, read_jsonl(index), Cache(cfg["cache"]))
    train = load_batch(cfg, index, manifest, "fusion_train")
    val = load_batch(
        cfg, index, manifest, "fusion_val", [h.shape[-1] for h in train.hidden]
    )
    results = []
    from ..evidence.diagnostics import crop_coverage, scaler_diagnostics

    stats = ExpertStats.fit(*train.inputs(), role="fusion_train")
    atomic_json(
        root / "scaler-diagnostics.json",
        scaler_diagnostics(
            train, stats, [e["id"] for e in cfg["experts"]], "fusion_train"
        ),
    )
    atomic_json(
        root / "crop-coverage.json",
        crop_coverage(
            read_jsonl(manifest), cfg["crop"], [e["id"] for e in cfg["experts"]]
        ),
    )

    def run_names(names):
        grid = cfg["pilot"].get(
            "candidate_grid",
            [
                {
                    "width": cfg["pilot"]["width"],
                    "learning_rate": cfg["pilot"]["learning_rate"],
                }
            ],
        )
        if not grid or len({digest(c) for c in grid}) != len(grid):
            raise ValueError("Invalid candidate grid")
        for name in names:
            for candidate in grid:
                variant = copy.deepcopy(cfg)
                if (
                    set(candidate) != {"width", "learning_rate"}
                    or candidate["width"] < 1
                    or candidate["learning_rate"] <= 0
                ):
                    raise ValueError("Only explicit positive width/lr grid supported")
                variant["pilot"].update(candidate)
                if "candidate_grid" in cfg["pilot"]:
                    variant["pilot"]["trial_id"] = "trial-" + digest(candidate)[:12]
                for seed in cfg["pilot"]["seeds"]:
                    if (
                        sum(
                            path.stat().st_size
                            for path in root.rglob("*")
                            if path.is_file()
                        )
                        >= cfg["pilot"]["max_output_bytes"]
                    ):
                        raise Blocked(
                            "Output quota reached; preserve outputs and stop before next run"
                        )
                    results.append(fit_one(variant, name, seed, train, val, reg))
                    atomic_json(root / "training-results.json", results)
                    if (
                        sum(
                            path.stat().st_size
                            for path in root.rglob("*")
                            if path.is_file()
                        )
                        > cfg["pilot"]["max_output_bytes"]
                    ):
                        raise Blocked(
                            "Output quota exceeded by last checkpoint; no confirmation scoring"
                        )
                    print(
                        name,
                        seed,
                        results[-1]["trial_id"],
                        results[-1]["steps"],
                        results[-1]["stop_reason"],
                        flush=True,
                    )

    run_names(cfg["pilot"]["h1_models"])
    gate = h1_decision(cfg, results, val)
    atomic_json(root / "H1-gate.json", gate)
    if gate["H2_allowed"]:
        run_names(cfg["pilot"]["h2_models"] + cfg["pilot"]["confirmation_models"])
    else:
        atomic_json(
            root / "H2-not-run.json",
            {
                "reason": gate["status"],
                "not_run": cfg["pilot"]["h2_models"]
                + cfg["pilot"]["confirmation_models"],
            },
        )
    candidates = sorted(
        {
            r["model"]
            for r in results
            if r["model"]
            in (
                "response_mlp",
                "all_views_mlp",
                "tta_stats",
                "set_attention",
                "full_attention",
            )
        }
    )
    _, summaries = select_family_trials(results, cfg["pilot"]["seeds"], candidates)
    chosen = min(
        summaries, key=lambda r: (r["mean_val_eer"], r["mean_val_bce"], r["model"])
    )
    selection = {
        "role": "fusion_val",
        "selected_nonontology": chosen["model"],
        "selected_trial_id": chosen["trial_id"],
        "rule": "fixed-seed mean validation EER, tie mean BCE then model/trial name",
        "family_selection": summaries,
        "results_sha256": sha256(root / "training-results.json"),
    }
    atomic_json(root / "selection.json", selection)
    return results


def lock_and_evaluate(cfg, results):
    root = Path(cfg["output"])
    manifest = root / "audit_manifest.jsonl"
    index = root / "extraction-0-of-1.jsonl"
    p = cfg["pilot"]
    if p.get("require_h1_support_before_confirmation", False):
        gate = json.loads((root / "H1-gate.json").read_text())
        if gate.get("status") != "SUPPORTED_IN_PILOT":
            atomic_json(
                root / "confirmation-not-run.json",
                {"reason": gate["status"], "test_loaded": False},
            )
            return []
    # Calibrate/evaluate only one selected trial per family, never all tuned trials.
    if p.get("candidate_grid"):
        chosen, _ = select_family_trials(
            results, p["seeds"], sorted({r["model"] for r in results})
        )
        results = [r for family in chosen.values() for r in family]
    unconverged = [
        {"model": r["model"], "seed": r["seed"]}
        for r in results
        if r["parameters"] and not r["convergence_claim"]
    ]
    if p.get("require_convergence", False) and unconverged:
        atomic_json(
            root / "confirmation-not-run.json",
            {
                "reason": "INCONCLUSIVE_CONVERGENCE",
                "unconverged_selected_runs": unconverged,
                "test_loaded": False,
            },
        )
        return []
    if p.get("protocol_version") == 2:
        from ..preregistration import require_preregistration

        require_preregistration(cfg)
    cal = load_batch(cfg, index, manifest, "calibration")
    val = load_batch(cfg, index, manifest, "fusion_val")
    # Complete every lock before reading any test tensor or selecting using test scores.
    locks = []
    from .fit import restore

    for run in results:
        cp, model, stats, reg = restore(run["checkpoint"])
        with torch.inference_mode():
            c = forward(model, cal, stats, reg)
            v = forward(model, val, stats, reg)
        if not c.observed.all() or not v.observed.all():
            raise Blocked("Calibration/validation coverage failure")
        params = Platt.fit(c.logits.numpy(), cal.labels.numpy(), "calibration")
        z = v.logits.double().numpy() * params["scale"] + params["bias"]
        point = operating_point(
            val.labels.numpy(), z, "fusion_val", p["target_dev_fpr"]
        )
        lock = {
            "status": "frozen_integration_pilot",
            "model_sha256": sha256(run["checkpoint"]),
            "calibration": params,
            "operating_point": point,
            "ledger_sha256": sha256(manifest),
            "index_sha256": sha256(index),
            "config": cfg,
            "registry_hash": reg.digest,
            "code_sha256": code_hash(),
            "data_audit_sha256": sha256(root / "data-audit.json"),
            "selection_sha256": sha256(root / "selection.json"),
            "test_requests": [
                r["id"] for r in read_jsonl(manifest) if r["role"] == "test"
            ],
            "research_holdout": False,
        }
        path = Path(run["checkpoint"]).parent / "freeze.json"
        if path.exists():
            raise ValueError("Pilot lock already exists")
        atomic_json(path, lock)
        locks.append((run, lock))
    test = load_batch(cfg, index, manifest, "test")
    reports = []
    for run, lock in locks:
        _, model, stats, reg = restore(run["checkpoint"])
        with torch.inference_mode():
            out = forward(model, test, stats, reg)
        params = lock["calibration"]
        z = out.logits.double().numpy() * params["scale"] + params["bias"]
        prob = torch.tensor(z).sigmoid().numpy()
        point = lock["operating_point"]
        records = []
        for i, ident in enumerate(test.ids):
            ok = bool(out.observed[i])
            records.append(
                {
                    "id": ident,
                    "origin_id": test.origins[i],
                    "source_group": test.groups[i],
                    "status": "scored" if ok else "review",
                    "label": int(test.labels[i]),
                    "condition": test.audit[i].get("condition", "no_new_processing"),
                    "logit": float(out.logits[i]) if ok else None,
                    "calibrated_logit": float(z[i]) if ok else None,
                    "p": float(prob[i]) if ok else None,
                    "raw_logit": float(out.logits[i]) if ok else None,
                    "p_calibrated": float(prob[i]) if ok else None,
                    "prediction": int(z[i] >= point["threshold"]) if ok else None,
                    "speaker_group": test.audit[i]["speaker_id"],
                    "calibrator_id": digest(params),
                    "model_id": run["checkpoint_sha256"],
                    "threshold_space": point["space"],
                    "threshold": point["threshold"],
                    "model_sha256": lock["model_sha256"],
                }
            )
        write_jsonl(
            Path(run["checkpoint"]).parent / "per_utterance_predictions.jsonl", records
        )
        valid = np.array([r["status"] == "scored" for r in records])
        y = test.labels.numpy()[valid]
        measured = (
            metrics(y, out.logits.numpy()[valid], prob[valid])
            if set(y) == {0.0, 1.0}
            else {"status": "undefined"}
        )
        decisions = z[valid] >= point["threshold"]
        measured.update(
            N_requested=len(records),
            N_scored=int(valid.sum()),
            N_failed=int((~valid).sum()),
            actual_fpr=float(decisions[y == 0].mean()) if (y == 0).any() else None,
            actual_tpr=float(decisions[y == 1].mean()) if (y == 1).any() else None,
            threshold_space=point["space"],
        )
        if p.get("validation_eer_conditions"):
            per_condition = {}
            for condition in sorted({r["condition"] for r in records}):
                ix = np.array([r["condition"] == condition for r in records])
                ok = ix & valid
                entry = {
                    "N_requested": int(ix.sum()),
                    "N_scored": int(ok.sum()),
                    "N_failed": int((ix & ~valid).sum()),
                }
                if set(test.labels.numpy()[ok]) == {0.0, 1.0}:
                    entry.update(
                        metrics(
                            test.labels.numpy()[ok], out.logits.numpy()[ok], prob[ok]
                        )
                    )
                else:
                    entry["status"] = "undefined_single_class_or_missing"
                per_condition[condition] = entry
            conditions = p["validation_eer_conditions"]
            complete = all(
                c in per_condition
                and per_condition[c].get("N_failed") == 0
                and "eer" in per_condition[c]
                for c in conditions
            )
            measured.update(
                primary_metric="prespecified_condition_macro_EER",
                primary_conditions=conditions,
                macro_eer=float(np.mean([per_condition[c]["eer"] for c in conditions]))
                if complete
                else None,
                per_condition=per_condition,
                primary_status="complete"
                if complete
                else "undefined_coverage_or_class_failure",
            )
        report = {
            "model": run["model"],
            "seed": run["seed"],
            "metrics": measured,
            "scope": "integration_pilot_only",
        }
        atomic_json(Path(run["checkpoint"]).parent / "test-report.json", report)
        reports.append(report)
    atomic_json(root / "test-reports.json", reports)
    return reports
