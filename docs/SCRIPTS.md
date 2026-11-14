# 实验入口

主要操作使用python -m ontoresponse，详见[运行实验](RUN_EXPERIMENTS.md)。scripts仅保留实验编排、数据检查和指标工具。

| 任务 | 入口 | 说明 |
|---|---|---|
| 固定模型纯测试 | python -u -m scripts.round1_batch --config ... --test-only --workers-per-gpu 3 | 复用最终权重，GPU0/1独立进程并发，不重新训练 |
| 完整测试EER汇总 | python -m scripts.report_round1_tests --root ... | 仅汇总全量覆盖、三个seed齐全的报告，绝不把部分数据EER作为完整结果 |
| 完整批量实验 | python -u -m scripts.round1_batch --config configs/round1-batch.yaml | 已授权GPU0/1；1024 pilot→完整开发两折/三seed→固定轮数重训→三数据集完整测试；分块保存、失败显式、可重启 |
| 第一轮开发smoke准备 | python -m scripts.prepare_round1 --work-dir ... --device cuda:0 | 2019 train/dev来源连接后选64条，32训练/32验证；--fold 1交换角色，不使用2019 eval或外部测试 |
| 有界基线比较 | python -m scripts.run_bounded --config ... | 普通基线后比较关系模型；用新output |
| 有界pilot | python -m scripts.run_pilot --config ... | --preflight-only检查实际输入/分组/设备/预算，无SHA或文档审核 |
| 完整专家LA eval | python -m scripts.expert_benchmark --config ... --phase ... | preflight / controls / score / summarize / run；需要显式GPU全评测预算 |
| 专家结果重算 | python -m scripts.report_expert_benchmark | 从本地完整逐条分数重算指标；不校验文件摘要；避免覆盖历史报告 |
| 来源连接与分组 | python -m scripts.prepare_connected --config ... --candidates ... | --audit-materialized检查T版本、组隔离和重复音频 |
| 历史环境T生成 | python -m scripts.materialize_development --config ... --base-manifest ... | 不运行专家，保留失败请求 |
| 可选数据方案快照 | python -m scripts.freeze_development --config ... | 默认实时检查；--freeze保存可读配置，不是运行前置条件 |
| 公开导出 | python -m scripts.export_public --output PATH | 默认每文件1MiB，排除私人路径、音频、权重和缓存；只列文件名/大小 |

measurement_controls、profile_multicrop用于实测专家/codec/crop与成本；pilot_summary和recompute_predictions用于旧结果统计与独立指标复算；stage_run只记录命令、退出码、耗时和日志。audit_fusion_candidates、audit_codecfake_candidates检查候选来源，不读取专家分数。

## 历史工具

原audit_bundle、lock_dependencies、audit_supervision、build_supervision_matrix移至[history/tools](history/tools)，不再列为实验步骤。它们只供需要重现旧归档结构的人使用；当前实验不导入这些SHA审计工具。更早的一次性脚本恢复见[历史说明](history/README.md)。历史报告保持原状，不把旧实验写成新结论。
