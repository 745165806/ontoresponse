"""Post-freeze reporting only: cost curves, paired source bootstrap, no fitting."""

import csv
import json
from pathlib import Path

from ontoresponse.common import atomic_json, read_jsonl
from ontoresponse.evaluation.metrics import paired_bootstrap


def main():
    root = Path("reports/pilot-20261110")
    out = Path("reports/codex-next-stage-20261110")
    reports = json.loads((root / "test-reports.json").read_text())
    runs = json.loads((root / "training-results.json").read_text())
    cost = []
    for report in reports:
        name = report["model"]
        run = next(
            r for r in runs if r["model"] == name and r["seed"] == report["seed"]
        )
        probes = (
            0
            if name in ("original_mlp", "best_single_expert")
            else 3
            if name.endswith("_p3")
            else 6
        )
        experts = 1 if name == "best_single_expert" else 2
        cost.append(
            {
                "model": name,
                "seed": report["seed"],
                "probe_budget": probes,
                "expert_views_per_request": experts * (1 + probes),
                "test_eer": report["metrics"].get("eer"),
                "actual_fpr": report["metrics"]["actual_fpr"],
                "actual_tpr": report["metrics"]["actual_tpr"],
                "n_requested": report["metrics"]["N_requested"],
                "n_failed": report["metrics"]["N_failed"],
                "parameters": run["parameters"],
                "optimizer_steps": run["steps"],
                "epochs": run["epochs"],
                "stop_reason": run["stop_reason"],
                "same_family_edges_with_3_probes": 0 if probes == 3 else None,
            }
        )
    atomic_json(
        out / "cost-performance.json",
        {
            "rows": cost,
            "actual_extraction_scope": "All methods share full 6-probe cache; expert-view counts are cold execution budgets, not measured GPU speedups",
            "three_probe_caveat": "one probe/family; same-family active edges disappear, so cost and topology changes are not isolated",
        },
    )
    selected = json.loads((root / "selection.json").read_text())["selected_nonontology"]
    paired = []
    for seed in (17, 29, 43):
        a = read_jsonl(
            root
            / "runs"
            / f"original_mlp-seed{seed}"
            / "per_utterance_predictions.jsonl"
        )
        b = read_jsonl(
            root / "runs" / f"{selected}-seed{seed}" / "per_utterance_predictions.jsonl"
        )
        if [r["id"] for r in a] != [r["id"] for r in b]:
            raise ValueError("Unpaired requests")
        common = [
            (x, y) for x, y in zip(a, b) if x["status"] == y["status"] == "scored"
        ]
        paired.append(
            {
                "seed": seed,
                "models": ["original_mlp", selected],
                "N_common": len(common),
                "paired_EER_difference_original_minus_response": paired_bootstrap(
                    [x["label"] for x, y in common],
                    [x["logit"] for x, y in common],
                    [y["logit"] for x, y in common],
                    [x["source_group"] for x, y in common],
                    replicates=2000,
                ),
            }
        )
    atomic_json(
        out / "pilot-paired-statistics.json",
        {
            "seeds": paired,
            "source_group_scope": "known speaker groups only; recording-level links unknown",
            "warning": "three groups / one spoof group; degeneracies and uncertainty are reported; seeds are not extra recordings",
        },
    )
    failure = []
    for r in read_jsonl(root / "extraction-0-of-1.jsonl"):
        for name, expert in r["experts"].items():
            if expert.get("status") != "complete":
                failure.append(
                    {
                        "id": r["id"],
                        "expert": name,
                        "status": expert["status"],
                        "error": expert.get("error"),
                    }
                )
            for view in expert.get("measurements", []):
                if view["status"] != "valid":
                    failure.append({"id": r["id"], "expert": name, **view})
    atomic_json(
        out / "pilot-failures.json",
        {"failures": failure, "count": len(failure), "every_request_retained": True},
    )
    old = json.loads(
        Path("reports/next-stage-20261110/consumed-exploration.json").read_text()
    )
    test = [r for r in read_jsonl(root / "audit_manifest.jsonl") if r["role"] == "test"]
    atomic_json(
        out / "test-consumption.json",
        {
            "old_exploratory_origins": old["origins"],
            "old_descendants": old["request_ids"],
            "new_consumed_pilot_test_origins": [r["origin_id"] for r in test],
            "new_consumed_speakers": sorted({r["speaker_id"] for r in test}),
            "policy": "No future confirmation claim on these origins/descendants; exclude relevant source components when maps are available",
        },
    )
    with (out / "cost-performance.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(cost[0]))
        writer.writeheader()
        writer.writerows(cost)
    print("cost rows", len(cost), "failures", len(failure), "paired seeds", len(paired))


if __name__ == "__main__":
    main()
