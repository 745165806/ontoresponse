"""Prepare two source-grouped ASV2019 development folds; never read eval data."""

import argparse
import csv
import io
import random
import tarfile
from pathlib import Path

import yaml

from ontoresponse.common import atomic_json, write_jsonl
from ontoresponse.data.prepare import validate_splits


def development_pool(root, mapping):
    """Join target and VC-source speakers before selecting any recordings."""
    root = Path(root)
    parent = {}

    def find(speaker):
        parent.setdefault(speaker, speaker)
        while parent[speaker] != speaker:
            parent[speaker] = parent[parent[speaker]]
            speaker = parent[speaker]
        return speaker

    rows = []
    for split, suffix in (("train", "trn"), ("dev", "trl")):
        protocol = (
            root
            / "ASVspoof2019_LA_cm_protocols"
            / (f"ASVspoof2019.LA.cm.{split}.{suffix}.txt")
        )
        for line in protocol.read_text().splitlines():
            if not line.strip():
                continue
            _, ident, _, _, label = line.split()
            if label not in ("bonafide", "spoof"):
                raise ValueError("Unknown label")
            entry = mapping[ident]
            bona, vc, target = (
                entry["VCTK_ID"],
                entry["VC_source_VCTK_ID"],
                entry["TTS_VC_target_speaker"],
            )
            if label == "bonafide":
                if bona == "-" or vc != "-" or target != "-":
                    raise ValueError("Bonafide source metadata mismatch")
                source, speaker = bona, bona.split("_")[0]
            else:
                if bona != "-" or target == "-":
                    raise ValueError("Spoof source metadata mismatch")
                source, speaker = (vc if vc != "-" else None), target
                if source is None and entry["TTS_text"] == "-":
                    raise ValueError("Unresolved spoof source")
            find(speaker)
            if source:
                a, b = find(speaker), find(source.split("_")[0])
                parent[max(a, b)] = min(a, b)
            rows.append(
                dict(
                    id=ident,
                    origin_id=ident,
                    speaker_id="vctk:" + speaker,
                    audio=str(root / f"ASVspoof2019_LA_{split}/flac/{ident}.flac"),
                    label=int(label == "spoof"),
                    official_split=split,
                    acoustic_source=source,
                    condition="no_new_processing",
                )
            )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate development ID")
    for row in rows:
        row["source_group"] = "vctk-component:" + find(row["speaker_id"][5:])
    return rows


def select_fold(pool, count=64, fold=0, seed=17):
    if count < 4 or count > 64 or count % 4 or fold not in (0, 1):
        raise ValueError("Smoke needs 4..64 origins divisible by 4, fold 0 or 1")
    groups = sorted({r["source_group"] for r in pool})
    if len(groups) != 2:
        raise ValueError("Expected two development components; revise split design")
    rng = random.Random(seed)
    selected = []
    for i, group in enumerate(groups):
        for label in (0, 1):
            candidates = sorted(
                (r for r in pool if r["source_group"] == group and r["label"] == label),
                key=lambda r: r["id"],
            )
            if len(candidates) < count // 4:
                raise ValueError("Insufficient label coverage")
            for row in rng.sample(candidates, count // 4):
                selected.append(
                    dict(row, role="fusion_train" if i == fold else "fusion_val")
                )
    validate_splits(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--work-dir", default="reports/round1-smoke/fold0")
    parser.add_argument("--cache-dir", default="artifacts/round1-smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-origins", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--source-map",
        default="reports/expert-benchmark-20261111/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz",
    )
    args = parser.parse_args()
    root = Path(args.work_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a new empty work directory")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["mode"] != "integration_pretrained":
        raise ValueError("This starter is only for integration_pretrained")
    with tarfile.open(args.source_map) as archive:
        stream = archive.extractfile("ASVspoof2019_LA_VCTK_MetaInfo.tsv")
        entries = list(csv.DictReader(io.TextIOWrapper(stream), delimiter="\t"))
    mapping = {r["ASVspoof_ID"]: r for r in entries}
    if len(mapping) != len(entries):
        raise ValueError("Duplicate source mapping ID")
    pool = development_pool(cfg["data"]["asv2019"], mapping)
    rows = select_fold(pool, args.max_origins, args.fold, args.seed)
    if any(not Path(r["audio"]).is_file() for r in rows):
        raise FileNotFoundError("Selected audio is missing")
    cfg["data"] = {"asv2019": cfg["data"]["asv2019"]}
    cfg["output"], cfg["cache"] = str(root), args.cache_dir
    cfg["runtime"].update(
        device=args.device, budget="smoke", max_origins=len(rows), steps=20
    )
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "audit_manifest.jsonl", rows)
    write_jsonl(
        root / "inference_manifest.jsonl",
        [{k: r[k] for k in ("id", "origin_id", "source_group", "audio")} for r in rows],
    )
    (root / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = {
        "status": "prepared_only",
        "development_pool": len(pool),
        "fold": args.fold,
        "seed": args.seed,
        "summary": validate_splits(rows),
        "config": str(root / "config.yaml"),
        "limitations": [
            "One connected source/speaker group per role; engineering smoke only",
            "Experts have supervised exposure to ASV2019 train/dev",
            "RawNet2 training provenance remains unknown",
            "No test or calibration role; ASV2019 eval excluded",
        ],
    }
    atomic_json(root / "preparation.json", result)
    print(result)


if __name__ == "__main__":
    main()
