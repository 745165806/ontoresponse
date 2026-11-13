"""Full, frozen single-expert benchmark; no probes, fitting or eval selection.

The author's native bonafide output is the primary score. The fusion pipeline's
spoof-minus-bonafide margin is recorded separately, without selecting between
them on eval. Every requested ID remains in the prediction ledger.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ontoresponse.common import atomic_json, read_jsonl, same_expert
from ontoresponse.data.audio import crops, read_audio
from ontoresponse.evaluation.metrics import metrics
from ontoresponse.experts.base import AudioInput, FrozenExpert


def protocol_rows(recipe):
    rows = []
    for line in Path(recipe["protocol"]).read_text().splitlines():
        speaker, ident, unused, attack, label = line.split()
        if label not in ("bonafide", "spoof"):
            raise ValueError("Unknown protocol class")
        rows.append(
            dict(id=ident, speaker=speaker, attack=attack, label=int(label == "spoof"))
        )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate official ID")
    counts = Counter(r["label"] for r in rows)
    budget = recipe["budget"]
    if (len(rows), counts[0], counts[1]) != (
        budget["expected_requests"],
        budget["expected_bonafide"],
        budget["expected_spoof"],
    ):
        raise ValueError("Protocol counts differ from explicit full budget")
    return rows


def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text())
    if (
        recipe["budget"]["mode"] != "full_evaluation_only"
        or recipe["budget"]["optimizer_steps"] != 0
    ):
        raise ValueError("Only explicitly authorized evaluation is implemented")
    base = yaml.safe_load(Path(recipe["base_config"]).read_text())
    return recipe, base


def job_spec(base, job):
    spec = dict(next(e for e in base["experts"] if e["id"] == job["base_expert"]))
    for key in ("checkpoint", "state_key"):
        if key in job:
            spec[key] = job[key]
    spec["id"] = job["id"]
    return spec


class Waveforms(Dataset):
    """Only IDs and audio paths enter the inference dataset, never labels."""

    def __init__(self, ids, root, crop):
        self.ids, self.root, self.crop = ids, Path(root), crop

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ident = self.ids[index]
        path = self.root / (ident + ".flac")
        try:
            wave = read_audio(path)
            return dict(
                id=ident,
                waveform=crops(wave, self.crop)[0],
                samples=len(wave),
                status="valid",
            )
        except Exception as err:
            return dict(id=ident, status="failed", error=str(err))


def collate(items):
    return items


def official_eer(recipe, labels, bonafide_scores):
    path = Path(recipe["official_metric"])
    spec = importlib.util.spec_from_file_location("pinned_author_metric", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    y, scores = np.asarray(labels), np.asarray(bonafide_scores)
    eer, threshold = module.compute_eer(scores[y == 0], scores[y == 1])
    return dict(
        eer=float(eer),
        eer_percent=float(eer * 100),
        eer_summary_threshold=float(threshold),
        threshold_is_deployment_rule=False,
        method="author_stable_sort_nearest_DET",
    )


def summarize(recipe, rows, predictions):
    if [r["id"] for r in predictions] != [r["id"] for r in rows]:
        raise ValueError("Prediction ledger is not the complete ordered protocol")
    valid = [r for r in predictions if r["status"] == "scored"]
    report = dict(
        N_requested=len(rows),
        N_scored=len(valid),
        N_failed=len(rows) - len(valid),
        status="COMPLETE" if len(valid) == len(rows) else "INCOMPLETE",
        score_direction_chosen_on_eval=False,
        checkpoint_selected_on_eval=False,
    )
    if set(r["label"] for r in valid) == {0, 1}:
        y = np.array([r["label"] for r in valid])
        native = np.array([r["native_bonafide_score"] for r in valid])
        margin = np.array([r["spoof_margin"] for r in valid])
        report["native_author_eer"] = official_eer(recipe, y, native)
        report["native_grouped_roc"] = metrics(y, -native)
        report["margin_author_eer"] = official_eer(recipe, y, -margin)
        report["margin_grouped_roc"] = metrics(y, margin)
        report["per_attack_native_author_eer"] = {
            attack: official_eer(
                recipe,
                [r["label"] for r in valid if r["label"] == 0 or r["attack"] == attack],
                [
                    r["native_bonafide_score"]
                    for r in valid
                    if r["label"] == 0 or r["attack"] == attack
                ],
            )
            for attack in sorted({r["attack"] for r in valid if r["label"] == 1})
        }
    return report


def preflight(recipe, base):
    rows = protocol_rows(recipe)
    missing = [
        r["id"]
        for r in rows
        if not (Path(recipe["audio_root"]) / (r["id"] + ".flac")).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} missing official inputs, examples: {missing[:3]}"
        )
    jobs = []
    for job in recipe["jobs"]:
        spec = job_spec(base, job)
        jobs.append(
            dict(
                id=job["id"],
                checkpoint=spec["checkpoint"],
                gpu=job["gpu"],
                batch_size=job["batch_size"],
            )
        )
    result = dict(
        status="READY",
        requests=len(rows),
        labels=dict(Counter(r["label"] for r in rows)),
        missing=missing,
        jobs=jobs,
        budget=recipe["budget"],
        crop=recipe["crop"],
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_count=torch.cuda.device_count(),
    )
    atomic_json(Path(recipe["output"]) / "preflight.json", result)
    return result


def worker(recipe, base, job, phase, resume=False):
    torch.set_num_threads(recipe["threads"])
    torch.manual_seed(recipe["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA required for this full-eval recipe")
    device = torch.device("cuda:0")
    root = Path(recipe["output"]) / job["id"]
    root.mkdir(parents=True, exist_ok=True)
    rows = protocol_rows(recipe)
    started = time.monotonic()
    expert = FrozenExpert(job_spec(base, job), recipe["crop"], device)
    intended_lock = dict(
        expert=expert.identity,
        recipe=recipe,
        primary_score="native_bonafide_output",
        secondary_score="spoof_minus_bonafide",
        dtype="float32",
        seed=recipe["seed"],
    )
    lock_path = root / "measurement-lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        if not same_expert(previous["expert"], expert.identity):
            raise ValueError("Expert settings differ; use a new output directory")
    atomic_json(lock_path, intended_lock)
    # Fixed label-blind prefix and positions; no accuracy or EER enters controls.
    control_ids = [r["id"] for r in rows[: min(64, len(rows))]]
    dataset = Waveforms(control_ids, recipe["audio_root"], recipe["crop"])
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    if any(r["status"] != "valid" for r in items):
        raise ValueError("Control audio failure")
    direct = expert.check_upstream(
        AudioInput(
            items[0]["id"],
            read_audio(Path(recipe["audio_root"]) / (items[0]["id"] + ".flac")),
        ),
        atol=1e-4,
        rtol=1e-4,
    )
    with torch.inference_mode():
        inputs = torch.stack([r["waveform"] for r in items]).to(device)
        _, batched = expert._run(inputs)
        singleton = torch.cat([expert._run(x[None])[1] for x in inputs])
        _, repeated = expert._run(inputs)
        torch.testing.assert_close(batched, singleton, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(batched, repeated, atol=0, rtol=0)
        torch.cuda.synchronize()
    atomic_json(
        root / "gpu-controls.json",
        dict(
            status="PASSED",
            level="real_checkpoint_real_data_GPU",
            requests=len(items),
            upstream=direct,
            max_batch_singleton_logit_error=float((batched - singleton).abs().max()),
            exact_repeat=True,
            gpu_name=torch.cuda.get_device_name(),
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    if phase == "controls":
        return dict(status="CONTROLS_PASSED", expert=job["id"])
    path = root / "per_utterance_predictions.jsonl"
    completed = []
    if path.exists():
        if not resume:
            raise ValueError("Predictions exist; explicit --resume required")
        completed = read_jsonl(path)
        if [r["id"] for r in completed] != [r["id"] for r in rows[: len(completed)]]:
            raise ValueError("Resume protocol prefix mismatch")
    lookup = {r["id"]: r for r in rows}
    dataset = Waveforms(
        [r["id"] for r in rows[len(completed) :]], recipe["audio_root"], recipe["crop"]
    )
    loader = DataLoader(
        dataset,
        batch_size=job["batch_size"],
        num_workers=recipe["workers"],
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        multiprocessing_context="spawn" if recipe["workers"] else None,
    )
    last_progress = 0
    with path.open("a") as out, torch.inference_mode():
        for items in loader:
            if time.monotonic() - started > recipe["budget"]["max_seconds_per_expert"]:
                raise TimeoutError(
                    "Explicit expert wall-time budget reached; resumable prefix retained"
                )
            good = [r for r in items if r["status"] == "valid"]
            outputs = {}
            if good:
                inputs = torch.stack([r["waveform"] for r in good]).to(device)
                _, logits = expert._run(inputs)
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite expert output")
                for item, pair in zip(good, logits.cpu().tolist(), strict=True):
                    outputs[item["id"]] = pair
            for item in items:
                row = dict(lookup[item["id"]])
                row["samples"] = item.get("samples")
                if item["id"] in outputs:
                    pair = outputs[item["id"]]
                    row.update(
                        status="scored",
                        native_bonafide_score=pair[expert.spec["bonafide_index"]],
                        spoof_margin=pair[expert.spec["spoof_index"]]
                        - pair[expert.spec["bonafide_index"]],
                        raw_outputs=pair,
                    )
                else:
                    row.update(
                        status="failed",
                        error=item["error"],
                        native_bonafide_score=None,
                        spoof_margin=None,
                    )
                out.write(json.dumps(row, allow_nan=False) + "\n")
                completed.append(row)
            out.flush()
            if len(completed) - last_progress >= 512 or len(completed) == len(rows):
                elapsed = time.monotonic() - started
                progress = dict(
                    expert=job["id"],
                    completed=len(completed),
                    requested=len(rows),
                    elapsed_seconds=elapsed,
                    requests_per_second=len(completed) / elapsed,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    pid=os.getpid(),
                )
                atomic_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_progress = len(completed)
            if path.stat().st_size > recipe["budget"]["max_output_bytes"]:
                raise RuntimeError("Explicit output budget exceeded")
    report = summarize(recipe, rows, completed)
    report.update(
        expert=job["id"],
        elapsed_seconds=time.monotonic() - started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        provenance=job.get("provenance", expert.spec.get("supervision")),
    )
    atomic_json(root / "result.json", report)
    expert.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/expert-benchmark-20261111.yaml")
    parser.add_argument(
        "--phase",
        choices=["preflight", "controls", "score", "summarize", "run"],
        default="preflight",
    )
    parser.add_argument("--expert")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    recipe, base = load_recipe(args.config)
    if args.phase == "preflight":
        print(json.dumps(preflight(recipe, base), indent=2))
        return 0
    if args.phase == "summarize":
        for job in recipe["jobs"]:
            root = Path(recipe["output"]) / job["id"]
            print(
                job["id"],
                json.dumps(
                    summarize(
                        recipe,
                        protocol_rows(recipe),
                        read_jsonl(root / "per_utterance_predictions.jsonl"),
                    )
                ),
            )
        return 0
    if args.phase != "run":
        job = next(j for j in recipe["jobs"] if j["id"] == args.expert)
        result = worker(recipe, base, job, args.phase, args.resume)
        return 0 if result["status"] in ("COMPLETE", "CONTROLS_PASSED") else 2
    preflight(recipe, base)
    # Independent OS inference workers, at most one expert per selected GPU.
    pending = list(recipe["jobs"])
    active, results = [], []
    while pending or active:
        busy = {j["gpu"] for j, process, log, started in active}
        for job in list(pending):
            if job["gpu"] in busy:
                continue
            root = Path(recipe["output"]) / job["id"]
            root.mkdir(parents=True, exist_ok=True)
            log_path = root / (
                "evaluation-resume.log" if args.resume else "evaluation.log"
            )
            log = log_path.open("w")
            command = [
                sys.executable,
                "-u",
                "-m",
                "scripts.expert_benchmark",
                "--config",
                args.config,
                "--phase",
                "score",
                "--expert",
                job["id"],
            ] + (["--resume"] if args.resume else [])
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(job["gpu"]),
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS=str(recipe["threads"]),
                PYTHONDONTWRITEBYTECODE="1",
            )
            started = time.time()
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env
            )
            atomic_json(
                root / "command-start.json",
                dict(
                    command=command,
                    pid=process.pid,
                    gpu=job["gpu"],
                    log=str(log_path),
                    started_utc=datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                ),
            )
            active.append((job, process, log, started))
            busy.add(job["gpu"])
            pending.remove(job)
            print(
                "started", job["id"], "pid", process.pid, "gpu", job["gpu"], flush=True
            )
        for entry in list(active):
            job, process, log, started = entry
            if process.poll() is None:
                continue
            log.close()
            result = dict(
                expert=job["id"],
                exit_code=process.returncode,
                elapsed_seconds=time.time() - started,
                log=str(Path(recipe["output"]) / job["id"] / "evaluation.log"),
            )
            atomic_json(
                Path(recipe["output"]) / job["id"] / "command-result.json", result
            )
            results.append(result)
            active.remove(entry)
            print("finished", json.dumps(result), flush=True)
        if active:
            time.sleep(2)
    atomic_json(
        Path(recipe["output"]) / "completion.json",
        dict(
            results=results,
            status="COMPLETE"
            if all(r["exit_code"] == 0 for r in results)
            else "FAILED",
        ),
    )
    return 0 if all(r["exit_code"] == 0 for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
