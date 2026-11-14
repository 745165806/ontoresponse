# 当前状态 — 2026-11-14

任务未完成，已修复故障并恢复GPU任务。2019 train/dev用于开发，最终测试为2019 LA eval、2021 LA eval和In-the-Wild。

| 阶段 | 实际状态 |
|---|---|
| 1024条pilot | 4/4块提取完成；两折×三seed×12方法，共72次比较完成 |
| 50224条完整开发数据 | 197/197块提取完成；另72次开发比较完成，选择配置已保存 |
| 最终重训 | 原先保存3个模型后，因无参数best_single_expert创建AdamW而停止；已修复并恢复，目前8/36个checkpoint |
| 三测试集 | 尚无完整逐句预测或测试EER；不能用开发EER代替 |
| 修复验证 | 10项相关回归测试通过、0 skipped、退出0；Ruff通过 |

[开发验证EER](reports/round1-batch-20261113/DEVELOPMENT_EER.md)：两折×三seed均值，全视图MLP 0.019376%、分数MLP 0.020616%、原始embedding融合0.026072%、同邻接共享关系参数0.026376%、本体typed_response 0.029234%、错误family 0.027870%。当前开发结果未显示本体优于这些强对照；专家已使用2019开发数据且只有两个来源组件，不作为独立泛化证据。

修复：最佳单专家仅根据开发组原始margin的平均EER选择并保存chosen_expert，零优化步骤；神经融合仍使用原有开发选择的训练轮数。配置、专家权重、探针、其余方法选择结果不变，未读取测试分数。失败日志与恢复前状态均保留；不重提取已完成数据。

运行PID见reports/round1-batch-20261113/launch.json，当前阶段以progress.json为准；日志runner.log。10项测试命令/退出码/日志见recovery-tests-20261114.command.json与.log。GPU0/1在重启前空闲，其他GPU任务未改动；当前最终重训使用GPU0，提取阶段使用0/1。

[操作说明](docs/RUN_EXPERIMENTS.md)、[批量报告](reports/round1-batch-20261113/REPORT.md)。历史177项完整回归和64条GPU smoke结果保留，不将其冒称为本次修复后的完整回归。结果监控进程已重新启动，终态后汇总并同步小报告。GitHub仅同步代码、说明和小型报告，音频/权重/观测包留本地；无需SHA或文档封存。
