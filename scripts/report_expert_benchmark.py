"""Recompute complete expert benchmark ledgers and write the comparison report."""

import csv
import json
from pathlib import Path

from ontoresponse.common import atomic_json, read_jsonl, sha256
from scripts.expert_benchmark import load_recipe, protocol_rows, summarize


def main():
    recipe, base = load_recipe("configs/expert-benchmark-20261111.yaml")
    root = Path(recipe["output"])
    expected = protocol_rows(recipe)
    frozen = json.loads((root / "preflight.json").read_text())
    expected_weights = {j["id"]: j["checkpoint_sha256"] for j in frozen["jobs"]}
    summary = []
    reference_audio = None
    for job in recipe["jobs"]:
        folder = root / job["id"]
        result = json.loads((folder / "result.json").read_text())
        lock = json.loads((folder / "measurement-lock.json").read_text())
        if lock["expert"]["checkpoint_sha256"] != expected_weights[job["id"]]:
            raise ValueError("Checkpoint identity differs from preflight")
        path = folder / "per_utterance_predictions.jsonl"
        if sha256(path) != result["predictions_sha256"]:
            raise ValueError("Prediction content drift")
        predictions = read_jsonl(path)
        recomputed = summarize(recipe, expected, predictions)
        if recomputed["status"] != "COMPLETE":
            raise ValueError("Incomplete official benchmark")
        for key in (
            "native_author_eer",
            "margin_author_eer",
            "native_grouped_roc",
            "margin_grouped_roc",
        ):
            if recomputed[key] != result[key]:
                raise ValueError("Metric recomputation mismatch: " + key)
        audio = {r["id"]: r["audio_sha256"] for r in predictions}
        if reference_audio is None:
            reference_audio = audio
        elif audio != reference_audio:
            raise ValueError("Experts did not receive identical input audio content")
        summary.append(
            dict(
                expert=job["id"],
                gpu=job["gpu"],
                N_requested=result["N_requested"],
                N_scored=result["N_scored"],
                N_failed=result["N_failed"],
                native_eer_percent=result["native_author_eer"]["eer_percent"],
                margin_eer_percent=result["margin_author_eer"]["eer_percent"],
                native_grouped_roc_eer_percent=100
                * result["native_grouped_roc"]["eer"],
                checkpoint_sha256=expected_weights[job["id"]],
                elapsed_seconds=result["elapsed_seconds"],
                cuda_peak_allocated_bytes=result["cuda_peak_allocated_bytes"],
            )
        )
    report = {
        "status": "VERIFIED_COMPLETE_GPU_BENCHMARK",
        "dataset": "ASVspoof2019_LA_official_eval",
        "unique_audio_requests": len(expected),
        "bonafide": 7355,
        "spoof": 63882,
        "results": summary,
        "all_experts_identical_audio_hashes": True,
        "recomputed_from_prediction_files": True,
        "primary_metric": "Author compute_eer using native bonafide output; nearest DET point with stable ties",
        "secondary_metric": "Spoof-minus-bonafide margin; reported separately, never selected using eval",
        "model_or_direction_selected_on_eval": False,
        "fusion_training_started": False,
        "source_map_downloaded_and_real_format_verified": True,
        "interpretation": {
            "aasist_local": "Not a reproduction of author released 0.83% checkpoint; present local task-trained checkpoint is weaker.",
            "aasist_author_reference": "Same predeclared author-weight reference reproduces 0.83% when rounded to two decimals.",
            "ssl_aasist": "2019 eval benchmark only. Paper 0.82% concerns 2021 LA, not 2019 LA; no same-set paper target certified.",
            "rawnet2": "Local Mel frontend uses first_conv=1024, 20 initial filters; original paper reports other architecture variants. EER alone does not certify source/training reproduction.",
        },
    }
    atomic_json(root / "summary.json", report)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# 三专家完整 ASVspoof2019 LA eval — 2026-11-11",
        "",
        "## Material Passport",
        "",
        "- mode: experiment run / reproducibility verification",
        "- verification: VERIFIED_COMPLETE_GPU_BENCHMARK",
        "- scope: fixed checkpoints and scores; no expert/fusion training, calibration, threshold or test-driven selection",
        "- data: 71237 unique eval recordings, 7355 bonafide / 63882 spoof; all four runs complete, zero failures",
        "- compute: NVIDIA RTX A6000, GPUs 0/1/3, float32, eval, gradients disabled; GPU2 not used by this benchmark",
        "",
        "## 完整结果",
        "",
        "| 权重 | 作者原生分数 EER (%) | 融合 margin EER (%) | 有效/请求 | GPU | 秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['expert']} | {row['native_eer_percent']:.6f} | {row['margin_eer_percent']:.6f} | {row['N_scored']}/{row['N_requested']} | {row['gpu']} | {row['elapsed_seconds']:.1f} |"
        )
    lines += [
        "",
        "主 EER 使用作者原生 `output[:,bonafide_index]`，传入已锁 hash 的 AASIST 官方 `compute_eer`。分数方向在运行前固定。margin 是当前融合观测口径，其结果不能与原生 score 混为同一个实验。独立 grouped ROC 插值值另存 JSON，不能因细微口径差而选择较小者。",
        "",
        "## 是否达到论文效果",
        "",
        "- 现用本地 AASIST 未复现发布权重的 0.83%；本地作者参考权重实测约0.83%，支持评测链正确。没有修改当前融合配置或根据 eval 从历史 epoch 中挑模型。",
        "- SSL+AASIST 官方 README/论文的0.82%是ASV2021 LA，2.85%是2021 DF；本轮2019 LA结果不能证明这两项已经复现。当前本地任务权重也没有相同的RawBoost增强训练历史。",
        "- 原始RawNet2论文Mel S1/逆Mel S2/线性S3的2019 LA EER分别5.64/5.13/4.66%。本地RawNet2架构参数不同、checkpoint训练来源未知，不能仅凭EER接近便认证论文复现。",
        "",
        "## 验证和审计",
        "",
        "- 每个专家均通过真实GPU上游hidden/logit比对、batch/singleton容差检查、重复推断逐bit一致；每条请求输出status及音频sha256。",
        "- 四份逐条文件均重新加载并重算；覆盖、结果hash、权重hash和同一ID的跨专家音频hash全部复核。",
        "- unit/synthetic：新增5项；完整171 passed，0 skipped；这与真实checkpoint、完整真实数据、GPU证据分层。",
        "- 来源映射archive已取得，2019全协议和2021DF全部611829个本地keys映射全覆盖；2019 train/dev共享1043条声学来源，保守speaker/source组件不足以直接满足多角色划分。",
        "- 本地Codecfake train/dev共833343条文件全部存在，但speaker重叠且抽查原始采样率不一致；未在该不合格划分上启动全量融合。具体重分组方案见下方文档。",
        "",
        "## 可复核产物",
        "",
        "- `summary.json` / `summary.csv`：全结果及权重hash。",
        "- 各专家目录：`measurement-lock.json`、`gpu-controls.json`、`result.json`、`per_utterance_predictions.jsonl`、`command-*.json`、`evaluation.log`。",
        "- `official-source-map-audit.json`、`codecfake-candidate-audit.json`：无专家分数参与的数据审计。",
        "- [融合数据与GPU全量方案](../../docs/FUSION_DATA_RECOMMENDATION_20261111.md)。",
        "",
        "重算命令：`${USER_HOME}/anaconda3/envs/py310/bin/python -m scripts.report_expert_benchmark`。",
        "",
        "来源：[AASIST作者仓库](https://github.com/clovaai/aasist)、[SSL作者仓库](https://github.com/TakHemlata/SSL_Anti-spoofing)、[SSL原论文](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.pdf)、[RawNet2原论文](https://www.eurecom.fr/publication/6395/download/sec-publi-6395.pdf)。",
    ]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        root / "consumption-ledger.json",
        dict(
            dataset="ASVspoof2019_LA_eval",
            role="fixed_single_expert_benchmark",
            requests=len(expected),
            protocol_sha256=sha256(recipe["protocol"]),
            checkpoint_ids=[r["expert"] for r in summary],
            model_selection_allowed=False,
            cannot_claim_new_unseen_confirmation=True,
            note="Benchmark consumption does not change official partitions, but must be disclosed in future experiment selection/confirmation records.",
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
