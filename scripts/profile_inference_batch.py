"""Choose execution batch sizes on already-consumed development observations."""

import argparse
import dataclasses
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from ontoresponse.common import atomic_json, read_jsonl
from ontoresponse.data.audio import crops
from ontoresponse.evidence.cache import Cache
from ontoresponse.evidence.dataset import load_batch
from ontoresponse.experts.base import AudioInput, FrozenExpert
from ontoresponse.train.fit import forward, restore
from ontoresponse.train.pilot import subset


def validate_native(args, cfg, inputs, positions):
    report = json.loads(Path(args.output).read_text())
    selected_outputs = {}
    native = []
    for spec in cfg["experts"]:
        trials = [
            t
            for t in report["experts"][spec["id"]]
            if t["batch_size"] > 1
            and t["cudnn_allow_tf32"] is True
            and t.get("peak_allocated_bytes", float("inf")) < 18 * 1024**3
        ]
        fastest = max(t["views_per_second"] for t in trials)
        chosen = min(
            (t for t in trials if t["views_per_second"] >= 0.98 * fastest),
            key=lambda t: t["batch_size"],
        )
        size = chosen["batch_size"]
        torch.backends.cudnn.allow_tf32 = chosen["cudnn_allow_tf32"]
        expert = FrozenExpert(spec, cfg["crop"], args.device)
        actual = expert.predict_utterances(inputs, batch_size=size)
        expert.close()
        captured = []
        handle = None
        if spec["kind"] == "ssl_aasist":
            handle = expert.model.out_layer.register_forward_pre_hook(
                lambda m, a: captured.append(a[0].detach().clone())
            )
        elif spec["kind"] == "rawnet2":
            handle = expert.model.fc1_gru.register_forward_hook(
                lambda m, a, o: captured.append(o.detach().clone())
            )
        hidden, logits = [], []
        windows = [crops(item.waveform, cfg["crop"]) for item in inputs]
        assert all(len(x) == 1 for x in windows), (
            "Current profile requires the locked single crop"
        )
        with torch.inference_mode():
            for chunk in torch.cat(windows).split(size):
                x = chunk.to(args.device)
                value = (
                    expert.model(x, Freq_aug=False)
                    if spec["kind"] == "aasist"
                    else expert.model(x)
                )
                h, s = value if spec["kind"] == "aasist" else (captured[-1], value)
                hidden.append(h.cpu())
                logits.append(s.cpu())
        h, s = torch.cat(hidden), torch.cat(logits)
        margins = s[:, spec["spoof_index"]] - s[:, spec["bonafide_index"]]
        torch.testing.assert_close(actual.embeddings, h, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.spoof_margins, margins, atol=1e-6, rtol=1e-5)
        baseline = report["experts"][spec["id"]][0]
        selection = dict(chosen)
        selection["single_view_strict_match"] = selection.pop("eligible", False)
        selection["single_view_strict_mismatch"] = selection.pop("error", None)
        selection["native_batch_equivalence_passed"] = True
        report["selected"][spec["id"]] = dict(
            selection,
            speedup=chosen["views_per_second"] / baseline["views_per_second"],
            eligibility_basis="native upstream same-batch hidden/logit equivalence",
        )
        native.append(
            dict(
                expert=spec["id"],
                batch_size=size,
                native_hidden_max_error=float((actual.embeddings - h).abs().max()),
                native_margin_max_error=float(
                    (actual.spoof_margins - margins).abs().max()
                ),
                passed=True,
            )
        )
        selected_outputs[spec["id"]] = actual
        if handle:
            handle.remove()
        del expert
        gc.collect()
        torch.cuda.empty_cache()
    cfg["cache"] = "artifacts/round1-smoke"
    full = load_batch(
        cfg,
        "reports/round1-smoke/fold0/extraction-0-of-1.jsonl",
        "reports/round1-smoke/fold0/inference_manifest.jsonl",
        "inference",
    )
    before = subset(full, torch.tensor(positions))
    after = dataclasses.replace(
        before,
        hidden=[
            selected_outputs[e["id"]].embeddings.reshape(len(positions), 7, -1)
            for e in cfg["experts"]
        ],
        scores=torch.stack(
            [
                selected_outputs[e["id"]].spoof_margins.reshape(len(positions), 7)
                for e in cfg["experts"]
            ],
            1,
        ),
    )
    report["fusion_checks"] = []
    for name in ("typed_response", "all_views_mlp", "response_mlp", "relation_tied"):
        for seed in (17, 29, 43):
            _, model, stats, reg = restore(
                f"reports/round1-batch-20261113/final/{name}-seed{seed}/model.pt",
                args.device,
            )
            with torch.inference_mode():
                a = forward(model, before, stats, reg).logits
                b = forward(model, after, stats, reg).logits
            flips = int(((a >= 0) != (b >= 0)).sum())
            item = dict(
                model=name,
                seed=seed,
                max_logit_abs_error=float((a - b).abs().max()),
                decision_changes=flips,
                mean_logit_abs_error=float((a - b).abs().mean()),
                finite=bool(torch.isfinite(b).all()),
            )
            report["fusion_checks"].append(item)
            print(item, flush=True)
            if flips or not item["finite"]:
                report["status"] = "fusion_stability_failed"
                atomic_json(args.output, report)
                raise RuntimeError(
                    "Batched inference changed a development fusion decision"
                )
            del model
    report.update(
        status="passed_native_batch",
        native_checks=native,
        precision_note="Float32 batch-dependent rounding/top-k differences from batch1 are measured, not claimed bitwise identical; native batched implementation matches and development fusion decisions unchanged",
    )
    atomic_json(args.output, report)
    print(json.dumps(report["selected"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", default="reports/batch-throughput-20261114/profile.json"
    )
    parser.add_argument(
        "--validate-native",
        action="store_true",
        help="Validate fastest native batches and unchanged development fusion decisions",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(4)
    rows = read_jsonl("reports/round1-smoke/fold0/extraction-0-of-1.jsonl")
    positions = list(range(0, 64, 4))
    cache = Cache("artifacts/round1-smoke")
    inputs = [
        AudioInput(
            f"{row['id']}::{view['probe_id']}",
            cache.get(view["artifact_key"])[0]["waveform"],
        )
        for index in positions
        for row in [rows[index]]
        for view in row["views"]
    ]
    if args.validate_native:
        validate_native(args, cfg, inputs, positions)
        return
    report = dict(
        device=args.device,
        origins=len(positions),
        views=len(inputs),
        inputs="already-consumed 2019 development smoke; label-free inputs only",
        atol=1e-4,
        rtol=1e-4,
        experts={},
        selected={},
    )
    selected_outputs = {}
    original_tf32 = torch.backends.cudnn.allow_tf32
    for spec in cfg["experts"]:
        torch.backends.cudnn.allow_tf32 = original_tf32
        expert = FrozenExpert(spec, cfg["crop"], args.device)
        expert.predict_utterances(inputs[:2])
        start = time.perf_counter()
        reference = expert.predict_utterances(inputs)
        elapsed = time.perf_counter() - start
        trials = [
            dict(
                batch_size=1,
                cudnn_allow_tf32=original_tf32,
                eligible=True,
                seconds=elapsed,
                views_per_second=len(inputs) / elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(expert.device),
                hidden_max_abs_error=0,
                margin_max_abs_error=0,
            )
        ]
        best = trials[0]
        best_output = reference
        for tf32 in (True, False):
            torch.backends.cudnn.allow_tf32 = tf32
            for size in (8, 16, 32, 64):
                trial = dict(batch_size=size, cudnn_allow_tf32=tf32)
                try:
                    expert.predict_utterances(
                        inputs[: min(size, len(inputs))], batch_size=size
                    )
                    start = time.perf_counter()
                    actual = expert.predict_utterances(inputs, batch_size=size)
                    elapsed = time.perf_counter() - start
                    error = None
                    try:
                        torch.testing.assert_close(
                            actual.embeddings,
                            reference.embeddings,
                            atol=1e-4,
                            rtol=1e-4,
                        )
                        torch.testing.assert_close(
                            actual.spoof_margins,
                            reference.spoof_margins,
                            atol=1e-4,
                            rtol=1e-4,
                        )
                    except AssertionError as exc:
                        error = str(exc)[:500]
                    peak = actual.metadata["cuda_peak_allocated_bytes"]
                    trial.update(
                        seconds=elapsed,
                        views_per_second=len(inputs) / elapsed,
                        peak_allocated_bytes=peak,
                        hidden_max_abs_error=float(
                            (actual.embeddings - reference.embeddings).abs().max()
                        ),
                        margin_max_abs_error=float(
                            (actual.spoof_margins - reference.spoof_margins).abs().max()
                        ),
                        eligible=error is None and peak < 32 * 1024**3,
                        error=error,
                    )
                    if (
                        trial["eligible"]
                        and trial["views_per_second"] > best["views_per_second"]
                    ):
                        best, best_output = trial, actual
                except torch.cuda.OutOfMemoryError:
                    trial.update(eligible=False, error="CUDA OOM")
                trials.append(trial)
                print(spec["id"], trial, flush=True)
                torch.cuda.empty_cache()
        report["experts"][spec["id"]] = trials
        report["selected"][spec["id"]] = dict(
            best, speedup=best["views_per_second"] / trials[0]["views_per_second"]
        )
        selected_outputs[spec["id"]] = best_output
        atomic_json(args.output, report)
        expert.close()
        del expert
        gc.collect()
        torch.cuda.empty_cache()
    torch.backends.cudnn.allow_tf32 = original_tf32
    # Measure the downstream fusion outputs with the unchanged, saved normalization.
    cfg["cache"] = "artifacts/round1-smoke"
    full = load_batch(
        cfg,
        "reports/round1-smoke/fold0/extraction-0-of-1.jsonl",
        "reports/round1-smoke/fold0/inference_manifest.jsonl",
        "inference",
    )
    before = subset(full, torch.tensor(positions))
    after = dataclasses.replace(
        before,
        hidden=[
            selected_outputs[e["id"]].embeddings.reshape(len(positions), 7, -1)
            for e in cfg["experts"]
        ],
        scores=torch.stack(
            [
                selected_outputs[e["id"]].spoof_margins.reshape(len(positions), 7)
                for e in cfg["experts"]
            ],
            1,
        ),
    )
    report["fusion_checks"] = []
    for name in ("typed_response", "all_views_mlp", "response_mlp", "relation_tied"):
        for seed in (17, 29, 43):
            path = f"reports/round1-batch-20261113/final/{name}-seed{seed}/model.pt"
            _, model, stats, reg = restore(path, args.device)
            with torch.inference_mode():
                a = forward(model, before, stats, reg).logits
                b = forward(model, after, stats, reg).logits
            torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-4)
            report["fusion_checks"].append(
                dict(
                    model=name,
                    seed=seed,
                    max_logit_abs_error=float((a - b).abs().max()),
                    passed=True,
                )
            )
            del model
    report["status"] = "passed"
    atomic_json(args.output, report)
    print(json.dumps(report["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
