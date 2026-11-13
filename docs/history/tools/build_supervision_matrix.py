"""Separate known task supervision, model selection and unknown SSL/source exposure."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from ontoresponse.common import atomic_json, read_jsonl
from docs.history.tools.checksums import sha256

OUT = Path("reports/codex-next-stage-20261110")


def main():
    OUT.mkdir(exist_ok=True)
    audit = json.loads(
        Path("reports/next-stage-20261110/supervision-audit.json").read_text()
    )
    pilot = read_jsonl("reports/pilot-20261110/audit_manifest.jsonl")
    old = read_jsonl("reports/implementation/audit_manifest.jsonl")
    matrix = []
    overlaps = []
    for expert in audit["experts"]:
        for stage in [
            "generic_ssl_pretraining",
            "task_fit",
            "checkpoint_selection",
            "early_stopping",
            "task_hyperparameter_selection",
        ]:
            source_role = "fit" if stage == "task_fit" else "source_val"
            known = expert.get("provenance_pass") and stage in (
                "task_fit",
                "checkpoint_selection",
            )
            source = expert["supervision"][source_role] if known else None
            status = (
                "KNOWN"
                if known
                else "NOT_APPLICABLE"
                if (stage == "generic_ssl_pretraining" and expert["id"] != "ssl_aasist")
                or (stage == "early_stopping" and expert.get("provenance_pass"))
                else "UNKNOWN"
            )
            matrix.append(
                {
                    "expert": expert["id"],
                    "stage": stage,
                    "status": status,
                    "n_ids": source["n"] if source else None,
                    "n_speakers": len(source["speakers"]) if source else None,
                    "source_map_status": "UNKNOWN",
                    "manifest_sha256": source["sha256"] if source else None,
                    "selection_rule": expert.get("selection_rule")
                    if stage == "checkpoint_selection"
                    else None,
                    "selected_epoch": expert.get("selected_epoch")
                    if stage == "checkpoint_selection"
                    else None,
                    "new_pilot_included": expert["id"] != "rawnet2",
                    "pretraining_condition": "open_pretraining"
                    if expert["id"] == "ssl_aasist"
                    else None,
                }
            )
            for version, ledger in [
                ("old_consumed_development", old),
                ("new_pilot", pilot),
            ]:
                for role in ["fusion_train", "fusion_val", "calibration", "test"]:
                    part = [r for r in ledger if r["role"] == role]
                    if not part:
                        continue
                    ids = (
                        sorted({r["id"] for r in part} & set(source["ids"]))
                        if source
                        else None
                    )
                    speakers = (
                        sorted(
                            {r["speaker_id"] for r in part} & set(source["speakers"])
                        )
                        if source
                        else None
                    )
                    overlaps.append(
                        {
                            "expert": expert["id"],
                            "supervision_stage": stage,
                            "ledger": version,
                            "role": "confirmation_test" if role == "test" else role,
                            "ID_overlap_status": "KNOWN" if source else status,
                            "ID_overlap_count": len(ids) if ids is not None else None,
                            "overlap_IDs": ids,
                            "speaker_overlap_count": len(speakers)
                            if speakers is not None
                            else None,
                            "overlap_speakers": speakers,
                            "source_overlap_count": None,
                            "source_overlap_status": "UNKNOWN: recording-level official maps not locally available",
                        }
                    )
    atomic_json(OUT / "supervision_matrix.json", matrix)
    fields = list(matrix[0])
    with (OUT / "supervision_matrix.csv").open("w") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(matrix)
    atomic_json(
        OUT / "role_overlap_report.json",
        {
            "rows": overlaps,
            "research_holdout": "BLOCKED",
            "integration_gate": "PASS_KNOWN_TASK_ID_SPEAKER_ONLY",
            "unknown_is_not_zero": True,
            "RawNet2": "excluded",
            "maps": {
                "url": "https://www.asvspoof.org/asvspoof2019/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz",
                "official_index": "https://www.asvspoof.org/index2021.html",
                "download_status": "BLOCKED: shell network unavailable; web confirms official link but does not supply binary archive",
                "download_log": "source-maps/download.log",
            },
        },
    )
    # All recordings of a known speaker are in one component. Unknown source links
    # are explicitly absent, never certified as disjoint and never split to fit ratios.
    groups = defaultdict(list)
    for r in pilot:
        groups[r["speaker_id"]].append(r)
    summaries = [
        {
            "component_id": key,
            "known_edge_type": "speaker_equal",
            "source_edges": "UNKNOWN",
            "n": len(v),
            "roles": dict(Counter(r["role"] for r in v)),
            "labels": dict(Counter(r["label"] for r in v)),
        }
        for key, v in sorted(groups.items())
    ]
    if any(len(g["roles"]) != 1 for g in summaries):
        raise ValueError("Known connected speaker component split across roles")
    atomic_json(
        OUT / "dataset_summary.json",
        {
            "origins": len(pilot),
            "known_components": len(groups),
            "components": summaries,
            "source_graph_complete": False,
            "large_component_policy": "Do not split a connected component to achieve class ratios",
            "scope": "two-expert integration under known speaker/task isolation; not strict source-independent research",
            "ledger_sha256": sha256("reports/pilot-20261110/audit_manifest.jsonl"),
        },
    )
    print(
        "matrix rows",
        len(matrix),
        "overlap rows",
        len(overlaps),
        "known speaker components",
        len(groups),
    )


if __name__ == "__main__":
    main()
