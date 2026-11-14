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


def extract_parts(cfg, rows, name):
    settings = cfg["batch"]
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

    def worker(worker_id, device):
        for index in range(worker_id, len(chunks), len(settings["devices"])):
            folder = paths[index].parent
            if paths[index].is_file():
                continue
            if shutil.disk_usage(root).free < settings["minimum_free_disk_bytes"]:
                raise RuntimeError(
                    "Disk reserve reached; completed evidence remains resumable"
                )
            folder.mkdir(parents=True, exist_ok=True)
            part = chunks[index]
            local = copy.deepcopy(cfg)
            local["output"] = str(folder)
            local["cache"] = str(folder / "temporary_cache")
            local["runtime"].update(device=device, budget="pilot")
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
                    labels=torch.tensor(
                        [r["label"] for r in part], dtype=torch.float32
                    ),
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
                )
                print(name, "packed", done, "/", len(paths), "on", device, flush=True)

    with ThreadPoolExecutor(max_workers=len(settings["devices"])) as executor:
        futures = [
            executor.submit(worker, i, d) for i, d in enumerate(settings["devices"])
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


def score_dataset(cfg, dataset, paths, checkpoints):
    root = Path(cfg["output"])
    device = cfg["batch"]["devices"][0]
    for checkpoint in checkpoints:
        name = Path(checkpoint).parent.name
        out = root / "tests" / dataset / name
        if (out / "report.json").exists():
            continue
        cp, model, stats, reg = restore(checkpoint, device)
        destination = out / "per_utterance_predictions.jsonl"
        out.mkdir(parents=True, exist_ok=True)
        labels, scores, conditions = [], [], []
        expert_scores = {e["id"]: [] for e in cfg["experts"]}
        with destination.open("w") as stream:
            for path in paths:
                batch = torch.load(path, map_location="cpu", weights_only=False)[
                    "batch"
                ]
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
            report["per_codec"] = {
                k: metrics(y[c == k], s[c == k]) for k in sorted(set(c))
            }
        atomic_json(out / "report.json", report)
        print("TEST", dataset, name, "EER", report["eer"], flush=True)


def run(cfg, config_path):
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
        extract_parts(cfg, rows, stage)
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
    for dataset in b["test_datasets"]:
        rows = test_rows(cfg, dataset)
        save_progress(
            root, stage="test_extraction", dataset=dataset, requested=len(rows)
        )
        paths = extract_parts(cfg, rows, "test_evidence/" + dataset)
        score_dataset(cfg, dataset, paths, checkpoints)
    save_progress(
        root,
        stage="completed",
        test_datasets=b["test_datasets"],
        models=len(checkpoints),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-stage", choices=("pilot", "development"))
    parser.add_argument("--fold", type=int, choices=(0, 1))
    args = parser.parse_args()
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
                run(cfg, args.config)
            except Exception as error:
                save_progress(root, stage="failed", error=repr(error))
                raise


if __name__ == "__main__":
    main()
