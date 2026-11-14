"""Summarize complete fixed-model test reports; never substitute partial-data EER."""

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ontoresponse.common import atomic_json

EXPECTED = {"asv2019_eval": 71237, "asv2021_la": 148176, "inthewild": 31779}


def summarize_tests(root, seeds=(17, 29, 43)):
    root = Path(root)
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    models = {}
    datasets = {}
    config_path = root / "execution_config.yaml"
    config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    part_size = config.get("batch", {}).get("extract_part_size", 256)
    for dataset, expected in EXPECTED.items():
        valid, rejected = [], []
        for path in sorted((root / "tests" / dataset).glob("*/report.json")):
            report = json.loads(path.read_text())
            name, sep, seed = path.parent.name.rpartition("-seed")
            if (
                not sep
                or not seed.isdigit()
                or int(seed) not in seeds
                or report.get("n") != expected
                or report.get("requested") != expected
                or report.get("failed") != 0
                or not isinstance(report.get("eer"), (int, float))
                or not math.isfinite(report["eer"])
                or not 0 <= report["eer"] <= 1
            ):
                rejected.append(str(path))
                continue
            valid.append(path.parent.name)
            models.setdefault(name, {}).setdefault(dataset, {})[int(seed)] = (
                report["eer"] * 100
            )
        datasets[dataset] = dict(
            expected_requests=expected,
            full_model_seed_reports=len(valid),
            incomplete_or_invalid_reports=rejected,
        )
        packs = list((root / "test_evidence" / dataset / "parts").glob("*/evidence.pt"))
        packed_requests = 0
        for pack in packs:
            index = int(pack.parent.name)
            if not 0 <= index * part_size < expected:
                raise ValueError("Unexpected extraction part: " + str(pack))
            packed_requests += min(part_size, expected - index * part_size)
        datasets[dataset]["extraction"] = dict(
            packed_parts=len(packs),
            total_parts=math.ceil(expected / part_size),
            packed_requests=packed_requests,
            last_pack_utc=datetime.fromtimestamp(
                max(p.stat().st_mtime for p in packs), timezone.utc
            ).isoformat()
            if packs
            else None,
        )
    rows = []
    for model, data in sorted(models.items()):
        row = {"model": model, "datasets": {}}
        for dataset, values in data.items():
            if sorted(values) != sorted(seeds):
                continue
            numbers = [values[seed] for seed in seeds]
            row["datasets"][dataset] = dict(
                eer_percent_mean=statistics.mean(numbers),
                eer_percent_seed_std=statistics.stdev(numbers)
                if len(numbers) > 1
                else 0,
                per_seed_percent=values,
                n=EXPECTED[dataset],
                failed=0,
            )
        if row["datasets"]:
            rows.append(row)
    return dict(
        status=progress,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        datasets=datasets,
        models=rows,
        aggregation="EER (%) mean and sample standard deviation across three fixed seeds",
        limitations=[
            "integration_pretrained; experts exposed to ASV2019 development",
            "RawNet2 supervision provenance unknown",
            "Only full requested datasets are summarized; no test-based selection",
        ],
    )


def write_summary(root):
    root = Path(root)
    report = summarize_tests(root)
    atomic_json(root / "TEST_EER.json", report)
    lines = [
        "# 固定模型完整测试 EER",
        "",
        "状态：" + str(report["status"].get("stage", "pending")),
        "",
        "| 数据集 | 已提取录音 / 总录音 | 完整模型×seed报告 |",
        "|---|---:|---:|",
        *[
            f"| {dataset} | {data['extraction']['packed_requests']} / {data['expected_requests']} | {data['full_model_seed_reports']} |"
            for dataset, data in report["datasets"].items()
        ],
        "",
        "仅汇总完整测试集；单位为百分比，均值 ± 三个固定seed的样本标准差。空白表示尚未完成，不是EER为0。",
        "",
        "| 方法 | 2019 LA eval | 2021 LA eval | In-the-Wild |",
        "|---|---:|---:|---:|",
    ]
    for row in report["models"]:
        cells = []
        for dataset in EXPECTED:
            data = row["datasets"].get(dataset)
            cells.append(
                f"{data['eer_percent_mean']:.6f} ± {data['eer_percent_seed_std']:.6f}"
                if data
                else "—"
            )
        lines.append("| " + row["model"] + " | " + " | ".join(cells) + " |")
    if not report["models"]:
        lines += [
            "",
            "尚无完成三个seed、满足全量覆盖的测试结果。开发验证EER不作为此表的测试结果。",
        ]
    lines += [
        "",
        "完整样本数：2019 LA eval 71237；2021 LA eval阶段148176（不混入progress/hidden）；ITW 31779。",
        "",
        "限制：预训练专家已使用2019开发数据，RawNet2监督来源未知；测试结果不用于重新选择模型或探针。",
    ]
    (root / "TEST_EER.md").write_text("\n".join(lines) + "\n")
    return report


def verify_predictions(cfg, dataset, official_metric):
    """Recompute finished reports from every requested prediction, without fitting."""
    import numpy as np

    from ontoresponse.evaluation.metrics import metrics
    from scripts.expert_benchmark import official_eer
    from scripts.round1_batch import test_rows

    root = Path(cfg["output"])
    expected_rows = test_rows(cfg, dataset)
    expected = {row["id"]: row for row in expected_rows}
    if len(expected) != EXPECTED[dataset] or len(expected_rows) != len(expected):
        raise ValueError("Full protocol request count/uniqueness mismatch")
    names = [
        f"{name}-seed{seed}"
        for name in cfg["batch"]["models"]
        for seed in cfg["batch"]["seeds"]
    ]
    missing = [
        name
        for name in names
        if not (root / "tests" / dataset / name / "report.json").exists()
    ]
    if missing:
        return dict(status="pending", dataset=dataset, missing_model_reports=missing)
    expert_ids = [e["id"] for e in cfg["experts"]]
    observations, experts, results = {}, {}, []
    for name in names:
        folder = root / "tests" / dataset / name
        report = json.loads((folder / "report.json").read_text())
        seen, labels, scores = set(), [], []
        raw_scores = {e: [] for e in expert_ids}
        with (folder / "per_utterance_predictions.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                ident = row["id"]
                if ident in seen or ident not in expected:
                    raise ValueError("Duplicate/unexpected prediction ID: " + ident)
                source = expected[ident]
                if (
                    row["status"] != "scored"
                    or row["dataset"] != dataset
                    or any(
                        row[k] != source[k]
                        for k in ("label", "source_group", "condition")
                    )
                ):
                    raise ValueError("Prediction status/protocol mismatch: " + ident)
                seen.add(ident)
                labels.append(row["label"])
                scores.append(row["logit"])
                values = tuple(row["expert_margins"][e] for e in expert_ids)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Nonfinite expert observation")
                if ident in observations and observations[ident] != values:
                    raise ValueError(
                        "Models did not share expert observations: " + ident
                    )
                observations[ident] = values
                for expert, value in zip(expert_ids, values):
                    raw_scores[expert].append(value)
        if seen != set(expected):
            raise ValueError("Missing full-test predictions for " + name)
        if any(
            report[k] != v
            for k, v in dict(n=len(expected), requested=len(expected), failed=0).items()
        ):
            raise ValueError("Incomplete saved report for " + name)
        recomputed = metrics(labels, scores)
        for key in ("eer", "auroc", "spoof_auprc"):
            if abs(recomputed[key] - report[key]) > 1e-12:
                raise ValueError(
                    "Saved metric differs from predictions: " + name + " " + key
                )
        nearest = official_eer(
            dict(official_metric=official_metric), labels, -np.array(scores)
        )
        results.append(
            dict(
                model_seed=name,
                n=len(seen),
                eer_percent=recomputed["eer"] * 100,
                nearest_det_eer_percent=nearest["eer_percent"],
                auroc=recomputed["auroc"],
            )
        )
        if not experts:
            for expert, values in raw_scores.items():
                calculated = metrics(labels, values)
                if (
                    abs(
                        calculated["eer"]
                        - report["original_expert_margins"][expert]["eer"]
                    )
                    > 1e-12
                ):
                    raise ValueError("Saved expert EER differs: " + expert)
                reference = official_eer(
                    dict(official_metric=official_metric), labels, -np.array(values)
                )
                experts[expert] = dict(
                    eer_percent=calculated["eer"] * 100,
                    nearest_det_eer_percent=reference["eer_percent"],
                )
    result = dict(
        status="passed",
        dataset=dataset,
        requested=len(expected),
        failed=0,
        model_seed_reports=len(results),
        all_prediction_ids_and_protocol_fields_match=True,
        same_expert_observations_for_all_models=True,
        max_allowed_metric_error=1e-12,
        original_expert_margins=experts,
        models=results,
        official_metric=str(official_metric),
        note="ROC interpolation remains primary; upstream nearest-DET is a separately named convention. No test-based fitting or score reversal.",
    )
    atomic_json(root / f"prediction-check-{dataset}.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/round1-batch-20261113")
    parser.add_argument("--verify-predictions", choices=EXPECTED)
    parser.add_argument(
        "--official-metric", help="Upstream evaluation.py containing compute_eer"
    )
    args = parser.parse_args()
    if args.verify_predictions:
        if not args.official_metric:
            parser.error("--official-metric is required for prediction verification")
        cfg = yaml.safe_load((Path(args.root) / "execution_config.yaml").read_text())
        print(
            json.dumps(
                verify_predictions(cfg, args.verify_predictions, args.official_metric),
                indent=2,
            )
        )
    else:
        print(json.dumps(write_summary(args.root), indent=2))
