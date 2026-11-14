"""Summarize complete fixed-model test reports; never substitute partial-data EER."""

import argparse
import json
import math
import statistics
from pathlib import Path

from ontoresponse.common import atomic_json

EXPECTED = {"asv2019_eval": 71237, "asv2021_la": 148176, "inthewild": 31779}


def summarize_tests(root, seeds=(17, 29, 43)):
    root = Path(root)
    progress_path = root / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    models = {}
    datasets = {}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/round1-batch-20261113")
    args = parser.parse_args()
    print(json.dumps(write_summary(args.root), indent=2))
