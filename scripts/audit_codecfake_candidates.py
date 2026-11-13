"""Audit existing Codecfake Xie train/dev metadata, without fitting/scoring."""

import re
from collections import Counter
from pathlib import Path

import soundfile as sf

from ontoresponse.common import atomic_json


def recording_identity(filename):
    source = re.sub(r"^F\d\d_", "", Path(filename).stem)
    if re.fullmatch(r"p\d+_\d+", source):
        return "VCTK", source, source.split("_")[0]
    if re.fullmatch(r"SSB\d+", source) and len(source) >= 11:
        return "AISHELL3", source, source[:7]
    raise ValueError("Unrecognized original-source identity " + filename)


def run():
    root = Path("${DATA_ROOT}/Codecfake_Xie/extracted")
    labels = root / "label/label"
    inventories = {}
    summaries = {}
    for split in ("train", "dev"):
        path = labels / (split + ".txt")
        rows = []
        missing = []
        identities = set()
        for line in path.read_text().splitlines():
            name, label, codec = line.split()
            if name in identities:
                raise ValueError("Duplicate file " + name)
            identities.add(name)
            if label not in ("real", "fake") or (label == "real") != (codec == "0"):
                raise ValueError("Conflicting label/codec")
            corpus, source, speaker = recording_identity(name)
            rows.append(
                dict(
                    name=name,
                    label=int(label == "fake"),
                    corpus=corpus,
                    source=source,
                    speaker=speaker,
                )
            )
            if not (root / split / name).is_file():
                missing.append(name)
        controls = []
        for corpus in ("VCTK", "AISHELL3"):
            for label in (0, 1):
                row = next(
                    r for r in rows if r["corpus"] == corpus and r["label"] == label
                )
                path_audio = root / split / row["name"]
                if path_audio.is_file():
                    info = sf.info(path_audio)
                    controls.append(
                        dict(
                            name=row["name"],
                            corpus=corpus,
                            label=label,
                            samplerate=info.samplerate,
                            channels=info.channels,
                            frames=info.frames,
                            format=info.format,
                        )
                    )
        inventories[split] = rows
        summaries[split] = {
            "requests": len(rows),
            "labels": dict(Counter(r["label"] for r in rows)),
            "corpus_labels": dict(
                Counter(r["corpus"] + ":" + str(r["label"]) for r in rows)
            ),
            "sources": {
                c: len({r["source"] for r in rows if r["corpus"] == c})
                for c in ("VCTK", "AISHELL3")
            },
            "speakers": {
                c: len({r["speaker"] for r in rows if r["corpus"] == c})
                for c in ("VCTK", "AISHELL3")
            },
            "missing_files": len(missing),
            "missing_examples": missing[:5],
            "audio_format_controls": controls,
        }
        print(split, summaries[split]["requests"], "missing", len(missing), flush=True)
    intersections = {}
    for corpus in ("VCTK", "AISHELL3"):
        parts = {
            split: [r for r in rows if r["corpus"] == corpus]
            for split, rows in inventories.items()
        }
        intersections[corpus] = {
            key: len({r[key] for r in parts["train"]} & {r[key] for r in parts["dev"]})
            for key in ("source", "speaker")
        }
    report = {
        "status": "METADATA_AND_EXISTENCE_AUDIT_ONLY",
        "root": str(root),
        "partitions": summaries,
        "train_dev_overlap": intersections,
        "expert_inference_used": False,
        "training_started": False,
        "limitations": [
            "Speaker pattern inferred from documented source filename conventions; verify against original corpus metadata before certifying split",
            "Cross-role duplicate and upstream training/model-selection source checks still required",
            "A full partition may contain incomplete codec families; exact source-family inventory must be checked before a frozen training recipe",
            "This does not authorize reuse of any test partition for model selection",
        ],
    }
    atomic_json(
        "reports/expert-benchmark-20261111/codecfake-candidate-audit.json", report
    )
    print("overlap", intersections, flush=True)


if __name__ == "__main__":
    run()
