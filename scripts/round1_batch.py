"""Persistent GPU experiments with explicit budgets and development-only selection.

Extract in small resumable parts, validate and pack observations, then discard
only the temporary waveform cache. Test audio is scored after selection.json.
"""

import argparse
import copy
import csv
import dataclasses
import io
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
import yaml

from ontoresponse.common import atomic_json, compute_device, read_jsonl, write_jsonl
from ontoresponse.data.prepare import validate_splits
from ontoresponse.evaluation.metrics import metrics
from ontoresponse.evidence.cache import Cache
from ontoresponse.evidence.dataset import EvidenceBatch, load_batch, origin_weights
from ontoresponse.models.factory import build_model, matched_width, parameter_count
from ontoresponse.ontology.registry import OntologyRegistry, validate_evidence
from ontoresponse.train.development import epoch_schedule
from ontoresponse.train.fit import atomic_torch, forward, restore, set_seed
from ontoresponse.train.pilot import fit_one, predict_batches, subset
from reference.core import ExpertStats, source_weighted_bce
from scripts.prepare_round1 import development_pool


def save_progress(root, **values):
    atomic_json(Path(root) / "progress.json", dict(updated=time.time(), **values))


def run_work_queue(items, devices, action):
    """Pin workers to devices; claim each task once and stop dispatch on error."""
    if not devices:
        raise ValueError("At least one worker device required")
    tasks = Queue()
    for item in items:
        tasks.put(item)
    stopped = threading.Event()

    def worker(device):
        while not stopped.is_set():
            try:
                item = tasks.get_nowait()
            except Empty:
                return
            try:
                action(item, device)
            except BaseException:
                stopped.set()
                raise
            finally:
                tasks.task_done()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(worker, device) for device in devices]
        for future in futures:
            future.result()


def pool_from_config(cfg):
    with tarfile.open(cfg["batch"]["source_map"]) as archive:
        stream = archive.extractfile("ASVspoof2019_LA_VCTK_MetaInfo.tsv")
        rows = csv.DictReader(io.TextIOWrapper(stream), delimiter="\t")
        mapping = {r["ASVspoof_ID"]: r for r in rows}
    return development_pool(cfg["data"]["asv2019"], mapping)


def pilot_rows(pool, count, seed):
    if not 4 <= count <= 1024 or count % 4:
        raise ValueError("Pilot needs 4..1024 origins divisible by four")
    rng = np.random.default_rng(seed)
    groups = sorted({r["source_group"] for r in pool})
    if len(groups) != 2:
        raise ValueError("Revise the two-component development design")
    selected = []
    for group in groups:
        for label in (0, 1):
            rows = sorted(
                (r for r in pool if r["source_group"] == group and r["label"] == label),
                key=lambda r: r["id"],
            )
            selected += [
                rows[i] for i in rng.choice(len(rows), count // 4, replace=False)
            ]
    return selected


def test_rows(cfg, dataset):
    data = cfg["data"]
    rows = []
    if dataset == "asv2019_eval":
        root = Path(data["asv2019"])
        protocol = root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        for line in protocol.read_text().splitlines():
            speaker, ident, _, attack, label = line.split()
            rows.append(
                dict(
                    id=ident,
                    audio=str(root / f"ASVspoof2019_LA_eval/flac/{ident}.flac"),
                    speaker_id="2019-eval:" + speaker,
                    label=int(label == "spoof"),
                    condition="no_new_processing",
                    attack=attack,
                )
            )
    elif dataset == "asv2021_la":
        for line in Path(data["asv2021la_keys"]).read_text().splitlines():
            speaker, ident, codec, transmission, attack, label, trim, phase = (
                line.split()
            )
            if phase != "eval":
                continue
            rows.append(
                dict(
                    id="2021LA:" + ident,
                    audio=str(Path(data["asv2021la"]) / "flac" / (ident + ".flac")),
                    speaker_id="2021-la:" + speaker,
                    label=int(label == "spoof"),
                    condition=codec,
                    transmission=transmission,
                    attack=attack,
                    official_phase=phase,
                    trim=trim,
                )
            )
    elif dataset == "inthewild":
        root = Path(data["inthewild"])
        with (root / "meta.csv").open() as stream:
            for row in csv.DictReader(stream):
                if row["label"] not in ("spoof", "bona-fide", "bonafide"):
                    raise ValueError("Unknown ITW label")
                rows.append(
                    dict(
                        id="itw:" + row["file"],
                        audio=str(root / row["file"]),
                        speaker_id="itw:" + row["speaker"],
                        label=int(row["label"] == "spoof"),
                        condition="no_new_processing",
                    )
                )
    else:
        raise ValueError(dataset)
    for row in rows:
        row.update(
            origin_id=row["id"],
            source_group=row["speaker_id"],
            role="test",
            dataset=dataset,
            source_overlap_status="unknown",
        )
    validate_splits(rows)
    return rows


def check_evidence(records):
    for row in records:
        if not row.get("complete_all_experts") or any(
            m["status"] != "valid"
            for expert in row["experts"].values()
            for m in expert.get("measurements", [])
        ):
            raise RuntimeError(
                "Incomplete evidence; inspect extraction index: " + row["id"]
            )


def cache_directory(cfg, folder, name, scratch_dir=None):
    legacy = Path(folder) / "temporary_cache"
    if scratch_dir is None or legacy.exists():
        return legacy
    return (
        Path(scratch_dir)
        / Path(cfg["output"]).name
        / name
        / Path(folder).name
        / "temporary_cache"
    )


def extraction_policy(batch_profile, experts):
    if batch_profile is None:
        return {}
    data = json.loads(Path(batch_profile).read_text())
    if data.get("status") not in ("passed", "passed_native_batch"):
        raise ValueError("Inference profile has not passed validation")
    policy = {}
    for expert in experts:
        selected = data["selected"][expert["id"]]
        size = selected["batch_size"]
        if type(size) is not int or not 1 <= size <= 64:
            raise ValueError("Invalid profiled inference batch size")
        policy[expert["id"]] = dict(
            batch_size=size, cudnn_allow_tf32=selected["cudnn_allow_tf32"]
        )
    return policy


def extract_parts(
    cfg,
    rows,
    name,
    workers_per_gpu=1,
    scratch_dir=None,
    batch_profile=None,
    dynamic_queue=False,
):
    if not 1 <= workers_per_gpu <= 4:
        raise ValueError("Extraction supports 1..4 workers per configured GPU")
    settings = cfg["batch"]
    policy = extraction_policy(batch_profile, cfg.get("experts", []))
    worker_devices = settings["devices"] * workers_per_gpu
    root = Path(cfg["output"]) / name
    root.mkdir(parents=True, exist_ok=True)
    size = settings["extract_part_size"]
    if not 1 <= size <= 1024:
        raise ValueError("Extraction part exceeds explicit pilot working-set budget")
    if len(rows) > settings["max_dataset_origins"]:
        raise ValueError("Dataset budget exceeded")
    write_jsonl(root / "manifest.jsonl", rows)
    chunks = [rows[i : i + size] for i in range(0, len(rows), size)]
    paths = [root / "parts" / f"{i:05d}" / "evidence.pt" for i in range(len(chunks))]
    lock = threading.Lock()

    def process_part(index, device):
        folder = paths[index].parent
        if paths[index].is_file():
            return
        if shutil.disk_usage(root).free < settings["minimum_free_disk_bytes"]:
            raise RuntimeError(
                "Disk reserve reached; completed evidence remains resumable"
            )
        folder.mkdir(parents=True, exist_ok=True)
        part = chunks[index]
        local = copy.deepcopy(cfg)
        local["output"] = str(folder)
        local["cache"] = str(cache_directory(cfg, folder, name, scratch_dir))
        Path(local["cache"]).parent.mkdir(parents=True, exist_ok=True)
        if (
            shutil.disk_usage(Path(local["cache"]).parent).free
            < settings["minimum_free_disk_bytes"]
        ):
            raise RuntimeError("Scratch disk reserve reached")
        local["runtime"].update(device=device, budget="pilot")
        if policy:
            local["runtime"]["expert_inference"] = policy
        local.pop("pilot", None)
        infer = [
            {k: r[k] for k in ("id", "origin_id", "source_group", "audio")}
            for r in part
        ]
        write_jsonl(folder / "inference_manifest.jsonl", infer)
        (folder / "config.yaml").write_text(yaml.safe_dump(local, sort_keys=False))
        command = [
            sys.executable,
            "-m",
            "ontoresponse",
            "extract",
            "--config",
            str(folder / "config.yaml"),
            "--max-origins",
            str(len(part)),
        ]
        start = time.time()
        with (folder / "extraction.log").open("w") as log:
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=settings["part_timeout_seconds"],
            )
        atomic_json(
            folder / "command.json",
            dict(
                command=command,
                exit_code=proc.returncode,
                elapsed_seconds=time.time() - start,
                device=device,
            ),
        )
        if proc.returncode:
            raise RuntimeError("Extraction failed: " + str(folder))
        extraction = folder / "extraction-0-of-1.jsonl"
        records = read_jsonl(extraction)
        check_evidence(records)
        with lock:
            # Cache verification uses a shared process-local LRU; pack parts serially.
            cache = Cache(local["cache"])
            validate_evidence(local, records, cache)
            batch = load_batch(
                local, extraction, folder / "inference_manifest.jsonl", "inference"
            )
            batch = dataclasses.replace(
                batch,
                labels=torch.tensor([r["label"] for r in part], dtype=torch.float32),
                audit=part,
            )
            atomic_torch(
                paths[index],
                dict(batch=batch, probe_ids=list(OntologyRegistry().probe_ids)),
            )
            Cache._shared_verified.clear()
        # Packed observations contain hidden/logits, identities, masks, quality and row metadata.
        # The full extraction index retains operation/artifact correspondence after validation.
        shutil.rmtree(local["cache"])
        with lock:
            done = sum(p.is_file() for p in paths)
            save_progress(
                root,
                stage="extraction",
                completed_parts=done,
                total_parts=len(paths),
                requested=len(rows),
                last_part=index,
                last_part_seconds=time.time() - start,
                workers_per_gpu=workers_per_gpu,
                scheduling="dynamic" if dynamic_queue else "static",
            )
            print(name, "packed", done, "/", len(paths), "on", device, flush=True)

    if dynamic_queue:
        run_work_queue(
            [i for i, p in enumerate(paths) if not p.is_file()],
            worker_devices,
            process_part,
        )
    else:

        def worker(worker_id, device):
            for index in range(worker_id, len(chunks), len(worker_devices)):
                process_part(index, device)

        with ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
            futures = [
                executor.submit(worker, i, d) for i, d in enumerate(worker_devices)
            ]
            for future in futures:
                future.result()
    return paths


def load_packs(paths):
    batches = [
        torch.load(p, map_location="cpu", weights_only=False)["batch"] for p in paths
    ]
    first = batches[0]
    if any(b.identities != first.identities for b in batches):
        raise ValueError("Mixed expert identities across packed evidence")
    return EvidenceBatch(
        ids=sum((b.ids for b in batches), []),
        hidden=[
            torch.cat([b.hidden[i] for b in batches]) for i in range(len(first.hidden))
        ],
        scores=torch.cat([b.scores for b in batches]),
        valid=torch.cat([b.valid for b in batches]),
        quality=torch.cat([b.quality for b in batches]),
        labels=torch.cat([b.labels for b in batches]),
        origins=sum((b.origins for b in batches), []),
        groups=sum((b.groups for b in batches), []),
        audit=sum((b.audit for b in batches), []),
        identities=first.identities,
    )


def assign_role(batch, indices, role):
    out = subset(batch, torch.tensor(indices, dtype=torch.long))
    return dataclasses.replace(out, audit=[dict(r, role=role) for r in out.audit])


def select_single_expert(batch, experts):
    """Each source group served as a held-out development fold; no test input."""
    if any(r.get("role") not in ("fusion_train", "fusion_val") for r in batch.audit):
        raise ValueError("Single expert selection requires development evidence only")
    groups = sorted(set(batch.groups))
    if len(groups) < 2:
        raise ValueError("Single expert selection needs grouped development folds")
    if not batch.valid[:, :, 0].all():
        raise ValueError("Incomplete original expert coverage during selection")
    labels = batch.labels.numpy()
    memberships = np.asarray(batch.groups)
    scores = batch.scores[:, :, 0].numpy()
    candidates = []
    for index, expert in enumerate(experts):
        fold_eers = [
            metrics(labels[memberships == group], scores[memberships == group, index])[
                "eer"
            ]
            for group in groups
        ]
        candidates.append(
            dict(
                index=index,
                expert=expert["id"],
                fold_eers=fold_eers,
                mean_eer=float(np.mean(fold_eers)),
            )
        )
    chosen = min(candidates, key=lambda r: (r["mean_eer"], r["index"]))
    return dict(
        chosen=chosen,
        candidates=candidates,
        test_used=False,
        rule="minimum mean original-margin EER over held-out development groups; tie expert index",
    )


def refit(cfg, name, seed, train, width, epochs, reg):
    """Fixed epochs selected from development folds; never touch test during fitting."""
    out = Path(cfg["output"]) / "final" / f"{name}-seed{seed}"
    if (out / "model.pt").exists():
        return str(out / "model.pt")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = compute_device(cfg)
    dims = [h.shape[-1] for h in train.hidden]
    model = build_model(dims, reg, name, width, seed).to(device)
    model.expert_identities = train.identities
    stats = ExpertStats.fit(*train.inputs(), role="fusion_train")
    single_selection = None
    if name == "best_single_expert":
        single_selection = select_single_expert(train, cfg["experts"])
        model.chosen_expert.fill_(single_selection["chosen"]["index"])
        atomic_json(out / "expert-selection.json", single_selection)
        opt, epochs = None, 0
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["pilot"]["learning_rate"],
            weight_decay=cfg["pilot"]["weight_decay"],
        )
    history = []
    steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = epoch_schedule(train, seed, epoch)
        losses = []
        for indices in order.split(cfg["pilot"]["batch_size"]):
            part = subset(train, indices)
            opt.zero_grad(set_to_none=True)
            loss = source_weighted_bce(
                forward(model, part, stats, reg),
                part.labels,
                origin_weights(part.origins),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite refit loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            if not torch.isfinite(grad):
                raise RuntimeError("Nonfinite refit gradients")
            opt.step()
            losses.append(float(loss.detach()))
            steps += 1
        history.append(dict(epoch=epoch, steps=steps, train_bce=float(np.mean(losses))))
        write_jsonl(out / "history.jsonl", history)
    model.eval()
    atomic_torch(
        out / "model.pt",
        dict(
            model_name=name,
            dims=dims,
            width=width,
            seed=seed,
            model_state=model.state_dict(),
            stats=stats,
            expert_identities=train.identities,
            expert_specs=cfg["experts"],
            crop=cfg["crop"],
            probe_ids=list(reg.probe_ids),
            probe_families=list(reg.families),
            probe_features=reg.features,
            train_ids=train.ids,
            val_ids=[],
            train_groups=sorted(set(train.groups)),
            val_groups=[],
            epochs=epochs,
            selection_rule=single_selection["rule"]
            if single_selection
            else "fixed median best epoch from development folds; no test used",
            single_expert_selection=single_selection,
            mode="integration_pretrained",
            history=history,
            device=str(device),
        ),
    )
    return str(out / "model.pt")


def train_fold(cfg, stage, fold):
    root = Path(cfg["output"])
    paths = sorted((root / stage / "parts").glob("*/evidence.pt"))
    batch = load_packs(paths)
    groups = sorted(set(batch.groups))
    if len(groups) != 2:
        raise ValueError("Expected two source/speaker development groups")
    train = assign_role(
        batch,
        [i for i, g in enumerate(batch.groups) if g == groups[fold]],
        "fusion_train",
    )
    val = assign_role(
        batch,
        [i for i, g in enumerate(batch.groups) if g != groups[fold]],
        "fusion_val",
    )
    validate_splits(train.audit + val.audit)
    reg = OntologyRegistry()
    dims = [h.shape[-1] for h in train.hidden]
    target = parameter_count(build_model(dims, reg, "typed_response", 64))
    local = copy.deepcopy(cfg)
    local["output"] = str(root / stage / f"fold{fold}")
    local["runtime"]["device"] = cfg["batch"]["devices"][
        fold % len(cfg["batch"]["devices"])
    ]
    local["pilot"] = copy.deepcopy(cfg[stage + "_training"])
    results = []
    for name in cfg["batch"]["models"]:
        local["pilot"]["width"] = (
            matched_width(dims, reg, name, target)
            if name in cfg["batch"]["match_capacity"]
            else 64
        )
        for seed in cfg["batch"]["seeds"]:
            folder = Path(local["output"]) / "runs" / f"{name}-seed{seed}"
            result_path = folder / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                if folder.exists():
                    # Preserve an interrupted attempt, then restart this deterministic bounded run.
                    folder.rename(
                        folder.with_name(folder.name + f"-interrupted-{time.time_ns()}")
                    )
                result = fit_one(local, name, seed, train, val, reg)
            results.append(dict(result, fold=fold))
            print(
                stage,
                "fold",
                fold,
                name,
                seed,
                "val EER",
                result["val_eer"],
                flush=True,
            )
            save_progress(
                Path(local["output"]),
                stage="training",
                completed=len(results),
                total=len(cfg["batch"]["models"]) * len(cfg["batch"]["seeds"]),
            )
    atomic_json(Path(local["output"]) / "results.json", results)


def _score_checkpoint(cfg, dataset, paths, checkpoint, device):
    root = Path(cfg["output"])
    name = Path(checkpoint).parent.name
    out = root / "tests" / dataset / name
    if (out / "report.json").exists():
        return
    cp, model, stats, reg = restore(checkpoint, device)
    destination = out / "per_utterance_predictions.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    labels, scores, conditions = [], [], []
    expert_scores = {e["id"]: [] for e in cfg["experts"]}
    with destination.open("w") as stream:
        for path in paths:
            batch = torch.load(path, map_location="cpu", weights_only=False)["batch"]
            if set(batch.ids) & set(cp["train_ids"]):
                raise ValueError("Test/training ID overlap")
            prediction = predict_batches(model, batch, stats, reg, 128)
            if not prediction.observed.all():
                raise RuntimeError("Missing test coverage")
            for index, (row, logit) in enumerate(
                zip(batch.audit, prediction.logits.tolist())
            ):
                raw_scores = {
                    e["id"]: float(batch.scores[index, i, 0])
                    for i, e in enumerate(cfg["experts"])
                }
                for expert, value in raw_scores.items():
                    expert_scores[expert].append(value)
                item = dict(
                    id=row["id"],
                    label=row["label"],
                    logit=logit,
                    source_group=row["source_group"],
                    condition=row["condition"],
                    status="scored",
                    dataset=dataset,
                    expert_margins=raw_scores,
                )
                stream.write(json.dumps(item) + "\n")
                labels.append(row["label"])
                scores.append(logit)
                conditions.append(row["condition"])
    report = metrics(labels, scores)
    report.update(
        dataset=dataset,
        checkpoint=checkpoint,
        requested=len(labels),
        failed=0,
        device=device,
        calibration="none; ranking metrics from spoof-oriented logits",
        source_independence="unknown; integration_pretrained; historical exposure disclosed",
    )
    report["original_expert_margins"] = {
        e: metrics(labels, s) for e, s in expert_scores.items()
    }
    if dataset == "asv2021_la":
        y, s, c = np.array(labels), np.array(scores), np.array(conditions)
        report["per_codec"] = {k: metrics(y[c == k], s[c == k]) for k in sorted(set(c))}
    atomic_json(out / "report.json", report)
    print("TEST", dataset, name, "EER", report["eer"], flush=True)


def score_dataset(cfg, dataset, paths, checkpoints, parallel_devices=False):
    devices = list(dict.fromkeys(cfg["batch"]["devices"]))
    active_devices = devices if parallel_devices else devices[:1]
    progress_lock = threading.Lock()
    completed = 0

    def score(checkpoint, device):
        nonlocal completed
        _score_checkpoint(cfg, dataset, paths, checkpoint, device)
        with progress_lock:
            completed += 1
            save_progress(
                cfg["output"],
                stage="test_scoring",
                dataset=dataset,
                completed_models=completed,
                total_models=len(checkpoints),
                devices=active_devices,
            )

    save_progress(
        cfg["output"],
        stage="test_scoring",
        dataset=dataset,
        completed_models=0,
        total_models=len(checkpoints),
        devices=active_devices,
    )
    if parallel_devices:
        run_work_queue(checkpoints, devices, score)
    else:
        for checkpoint in checkpoints:
            score(checkpoint, devices[0])


def run_test_phase(
    cfg,
    checkpoints,
    workers_per_gpu=1,
    scratch_dir=None,
    batch_profile=None,
    dynamic_queue=False,
):
    root = Path(cfg["output"])
    for dataset in cfg["batch"]["test_datasets"]:
        rows = test_rows(cfg, dataset)
        save_progress(
            root, stage="test_extraction", dataset=dataset, requested=len(rows)
        )
        paths = extract_parts(
            cfg,
            rows,
            "test_evidence/" + dataset,
            workers_per_gpu,
            scratch_dir,
            batch_profile,
            dynamic_queue,
        )
        score_dataset(cfg, dataset, paths, checkpoints, parallel_devices=dynamic_queue)
    save_progress(
        root,
        stage="completed",
        test_datasets=cfg["batch"]["test_datasets"],
        models=len(checkpoints),
    )


def evaluate_ready_models(
    cfg, workers_per_gpu=1, scratch_dir=None, batch_profile=None, dynamic_queue=False
):
    """Evaluate the saved final ensemble of candidates without entering training."""
    root = Path(cfg["output"])
    if yaml.safe_load((root / "execution_config.yaml").read_text()) != cfg:
        raise ValueError("Evaluation settings differ from the completed training run")
    ready = json.loads((root / "models-ready.json").read_text())
    checkpoints = ready["checkpoints"]
    expected = {
        f"{name}-seed{seed}"
        for name in cfg["batch"]["models"]
        for seed in cfg["batch"]["seeds"]
    }
    if {Path(p).parent.name for p in checkpoints} != expected or any(
        not Path(p).is_file() for p in checkpoints
    ):
        raise ValueError("Missing final checkpoints; test-only mode never retrains")
    if not (root / "selection.json").is_file():
        raise ValueError("Development selection is missing")
    run_test_phase(
        cfg, checkpoints, workers_per_gpu, scratch_dir, batch_profile, dynamic_queue
    )


def run(
    cfg,
    config_path,
    workers_per_gpu=1,
    scratch_dir=None,
    batch_profile=None,
    dynamic_queue=False,
):
    root = Path(cfg["output"])
    b = cfg["batch"]
    if not b.get("full_run_authorized") or cfg["mode"] != "integration_pretrained":
        raise ValueError(
            "Explicit authorized integration full-run configuration required"
        )
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "execution_config.yaml"
    if snapshot.exists() and yaml.safe_load(snapshot.read_text()) != cfg:
        raise ValueError(
            "Use a new output directory after changing experimental settings"
        )
    if not snapshot.exists():
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
    pool = pool_from_config(cfg)
    for row in pool:
        row["role"] = (
            "fusion_train"  # selection roles are assigned independently per fold
        )
    for stage, rows in (
        ("pilot", pilot_rows(pool, b["pilot_origins"], b["split_seed"])),
        ("development", pool),
    ):
        save_progress(root, stage=stage, action="extract", requested=len(rows))
        extract_parts(
            cfg, rows, stage, workers_per_gpu, scratch_dir, batch_profile, dynamic_queue
        )
        jobs = []
        for fold in (0, 1):
            log = (root / stage / f"fold{fold}.log").open("a")
            command = [
                sys.executable,
                "-m",
                "scripts.round1_batch",
                "--config",
                config_path,
                "--train-stage",
                stage,
                "--fold",
                str(fold),
            ]
            jobs.append(
                (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
            )
        for proc, log in jobs:
            code = proc.wait()
            log.close()
            if code:
                raise RuntimeError(f"{stage} training exited {code}; inspect fold logs")
        summaries = sum(
            (
                json.loads((root / stage / f"fold{f}/results.json").read_text())
                for f in (0, 1)
            ),
            [],
        )
        atomic_json(
            root / stage / "comparison.json",
            dict(
                results=summaries,
                test_used=False,
                independent_development_components=2,
                claim_scope="integration_exploration",
            ),
        )
    selection_path = root / "selection.json"
    if not selection_path.exists():
        selected = {}
        for name in b["models"]:
            rows = [r for r in summaries if r["model"] == name]
            selected[name] = dict(
                width=rows[0]["width"],
                epochs=max(1, int(np.median([r["best_epoch"] for r in rows]))),
                mean_development_eer=float(np.mean([r["val_eer"] for r in rows])),
            )
        atomic_json(
            selection_path,
            dict(
                models=selected,
                seeds=b["seeds"],
                test_used=False,
                rule="common epoch count per architecture, median best-BCE epoch over 2 folds x 3 seeds",
            ),
        )
    selected = json.loads(selection_path.read_text())["models"]
    train = load_packs(sorted((root / "development/parts").glob("*/evidence.pt")))
    train = assign_role(train, list(range(len(train.ids))), "fusion_train")
    reg = OntologyRegistry()
    local = copy.deepcopy(cfg)
    local["runtime"]["device"] = b["devices"][0]
    local["pilot"] = cfg["development_training"]
    checkpoints = []
    for name, settings in selected.items():
        for seed in b["seeds"]:
            save_progress(
                root, stage="refit", model=name, seed=seed, epochs=settings["epochs"]
            )
            checkpoints.append(
                refit(
                    local, name, seed, train, settings["width"], settings["epochs"], reg
                )
            )
    del train
    # No test audio or scores are processed until all model settings and final weights exist.
    atomic_json(
        root / "models-ready.json",
        dict(checkpoints=checkpoints, selection=str(selection_path)),
    )
    run_test_phase(
        cfg, checkpoints, workers_per_gpu, scratch_dir, batch_profile, dynamic_queue
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-stage", choices=("pilot", "development"))
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Use models-ready.json; never enter development or refit",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Independent unchanged single-view extractors per GPU",
    )
    parser.add_argument(
        "--scratch-dir",
        help="Fast local temporary cache directory; preserve existing partial caches",
    )
    parser.add_argument(
        "--batch-profile", help="Validated per-expert execution batch profile JSON"
    )
    parser.add_argument(
        "--dynamic-queue",
        action="store_true",
        help="Balance extraction across devices and score different models concurrently",
    )
    args = parser.parse_args()
    if args.test_only and args.train_stage:
        parser.error("--test-only cannot be combined with --train-stage")
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(cfg["runtime"].get("threads", 4))
    if args.train_stage:
        train_fold(cfg, args.train_stage, args.fold)
    else:
        import fcntl

        root = Path(cfg["output"])
        root.mkdir(parents=True, exist_ok=True)
        with (root / "runner.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                if args.test_only:
                    evaluate_ready_models(
                        cfg,
                        args.workers_per_gpu,
                        args.scratch_dir,
                        args.batch_profile,
                        args.dynamic_queue,
                    )
                else:
                    run(
                        cfg,
                        args.config,
                        args.workers_per_gpu,
                        args.scratch_dir,
                        args.batch_profile,
                        args.dynamic_queue,
                    )
            except Exception as error:
                save_progress(root, stage="failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
