"""Read actual checkpoint/history/manifests, not a status summary."""

import csv
import json
import shutil
from pathlib import Path

import torch
import yaml

from ontoresponse.common import atomic_json, read_jsonl
from docs.history.tools.checksums import sha256
from ontoresponse.data.audio import read_audio
from ontoresponse.transforms.probes import audio_hash

OUT = Path("reports/next-stage-20261110")


def main():
    cfg = yaml.safe_load(Path("configs/local.yaml").read_text())
    out = OUT / "upstream-evidence"
    out.mkdir(parents=True, exist_ok=True)
    old = read_jsonl("reports/implementation/audit_manifest.jsonl")
    root = Path(cfg["data"]["asv2019"])
    protocols = {}
    for split in ("train", "dev", "eval"):
        suffix = "trn" if split == "train" else "trl"
        path = (
            root
            / f"ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{split}.{suffix}.txt"
        )
        for line in path.read_text().splitlines():
            speaker, ident, _, attack, label = line.split()
            protocols[ident] = {
                "speaker": speaker,
                "label": int(label == "spoof"),
                "split": split,
            }
    results = []
    for e in cfg["experts"]:
        cp = Path(e["checkpoint"])
        base = cp.parent.parent
        folder = out / e["id"]
        folder.mkdir(exist_ok=True)
        item = {
            "id": e["id"],
            "checkpoint": str(cp),
            "checkpoint_exists": cp.is_file(),
            "checkpoint_sha256": sha256(cp),
            "research_holdout_pass": False,
            "external_ssl_pretraining_overlap": "unknown"
            if e["kind"] == "ssl_aasist"
            else None,
        }
        state = torch.load(cp, map_location="cpu", weights_only=False, mmap=True)
        if e["state_key"]:
            final = json.loads((base / "finalized.json").read_text())
            config = json.loads((base / "config.yaml").read_text())
            history = list(csv.DictReader((base / "history.csv").open()))
            best = min(
                history, key=lambda r: (float(r["source_val_eer"]), int(r["epoch"]))
            )
            selected = Path(final["selected_checkpoint_ref"])
            item.update(
                selected_epoch=int(best["epoch"]),
                history_epochs=len(history),
                maximum_configured_epochs=config["execution"]["training"]["max_epochs"],
                selection_rule=final["selection_rule"],
                selected_epoch_matches_metadata=int(best["epoch"])
                == final["selected_epoch"]
                == state["epoch"],
                selected_epoch_sha256=sha256(selected),
                checkpoint_matches_selected_epoch=sha256(selected) == sha256(cp),
                early_stopping="no patience-based stopping recorded; history reaches configured 80 epochs; best checkpoint selected post hoc on source_val",
                interruption_record=json.loads((base / "failure.json").read_text())
                if (base / "failure.json").exists()
                else None,
                warning="failure.json also records interruption; it is retained, not represented as early stopping",
                class_index_map=state["class_index_map"],
                training=state["training"],
            )
            supervised = {}
            for role in ("fit", "source_val"):
                p = Path(state[role]["manifest_ref"])
                rows = read_jsonl(p)
                ids = {Path(r["audio_relpath"]).stem for r in rows}
                if not ids <= set(protocols):
                    raise ValueError("Unknown upstream IDs")
                speakers = {protocols[i]["speaker"] for i in ids}
                supervised[role] = {
                    "path": str(p),
                    "sha256": sha256(p),
                    "ids": sorted(ids),
                    "speakers": sorted(speakers),
                    "n": len(rows),
                    "old_exact_overlap": {
                        r: [v["id"] for v in old if v["role"] == r and v["id"] in ids]
                        for r in ("fusion_train", "fusion_val", "calibration")
                    },
                    "old_speaker_overlap": {
                        r: sorted(
                            {v["speaker_id"] for v in old if v["role"] == r} & speakers
                        )
                        for r in ("fusion_train", "fusion_val", "calibration")
                    },
                }
                shutil.copy2(p, folder / (role + ".jsonl"))
            item["supervision"] = supervised
            item["provenance_pass"] = (
                item["selected_epoch_matches_metadata"]
                and item["checkpoint_matches_selected_epoch"]
            )
            for name in (
                "config.yaml",
                "finalized.json",
                "run.json",
                "meta.json",
                "history.csv",
                "metrics.jsonl",
                "train_log.jsonl",
                "failure.json",
                "source_train_job.json",
            ):
                if (base / name).is_file():
                    shutil.copy2(base / name, folder / name)
            for p in (base / "code_snapshot").rglob("*.py"):
                dest = folder / "training-code" / p.relative_to(base / "code_snapshot")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            item["decision"] = (
                "retain only for integration pilot; exclude every upstream source_val speaker from all new fusion roles"
            )
        else:
            item.update(
                provenance_pass=False,
                supervision=None,
                selection_rule=None,
                early_stopping=None,
                checkpoint_contents="plain task model state_dict; no training IDs, validation IDs, seed or selection history",
                decision="QUARANTINED: exclude RawNet2 from new experiments; never infer source history from filename or current training script",
            )
            for name in (
                "main.py",
                "data_utils.py",
                "README.md",
                e["module"],
                e.get("config", ""),
            ):
                p = Path(e["repository"]) / name
                if p.is_file():
                    dest = folder / p.name
                    shutil.copy2(p, dest)
        del state
        for name in (e["module"], "LICENSE"):
            p = Path(e["repository"]) / name
            if p.is_file():
                shutil.copy2(p, folder / p.name)
        results.append(item)
    external = read_jsonl("reports/implementation/external/conditions-audit.jsonl")
    consumed = {
        "schema": "consumed-exploration-1",
        "origins": sorted({r["origin_id"] for r in external}),
        "request_ids": [r["id"] for r in external],
        "source_groups": sorted({r["source_group"] for r in external}),
        "audio_sha256": {
            r["id"]: sha256(r["audio"]) for r in external if Path(r["audio"]).is_file()
        },
        "waveform_sha256": sorted(
            {
                audio_hash(read_audio(r["audio"]))
                for r in external
                if Path(r["audio"]).is_file()
            }
        ),
        "policy": "All eight origins and any derived versions are consumed exploration; no recalibration, threshold/model choice, or fresh test claims.",
    }
    atomic_json(OUT / "consumed-exploration.json", consumed)
    atomic_json(
        OUT / "supervision-audit.json",
        {
            "experts": results,
            "mode_allowed": "integration_pretrained_only",
            "research_holdout_pass": False,
            "old_freezes_preserved": True,
            "remediation": "new ledgers; remove complete source_val speaker groups; quarantine RawNet2; require actual source maps for strict research_holdout",
        },
    )
    print([(r["id"], r["provenance_pass"], r.get("selected_epoch")) for r in results])


if __name__ == "__main__":
    main()
