# 当前状态 — 2026-11-14

正在使用已保存的36个最终模型进行完整测试。已完成最终重训，当前进程为纯测试入口，不进入开发训练或重训。2019 LA eval、2021 LA eval和In-the-Wild分别评测；完整测试EER尚未产生。

| 阶段 | 实际状态 |
|---|---|
| 1024条pilot | 4/4块提取、72次两折/三seed比较完成 |
| 50224条完整开发数据 | 197/197块提取、另72次比较完成，开发选择已固定 |
| 最终模型 | 36/36 checkpoint已保存，models-ready.json记录完整清单 |
| 固定模型测试 | GPU0/1各3个独立进程，复用已完成块；2019启动快照为10/279块已打包，实时进度以progress.json为准；2021/ITW按流程继续 |
| 测试EER | 尚无全量、三个seed齐全的结果；不以开发或部分数据EER替代 |

当前命令：`python -u -m scripts.round1_batch --config configs/round1-batch.yaml --test-only --workers-per-gpu 3 --scratch-dir /tmp/ontoresponse-round1-cache`。PID见launch.json，总进度progress.json，逐数据集进度test_evidence/<dataset>/progress.json，日志runner.log。原无参数基线优化器故障已修复，错误日志保留；恢复后重训成功。临时缓存已改为本地SSD，保留已完成观测包和部分计算。原始专家逐视图推断实现保持不变：多视图batch前向在开发录音出现超过预设容差的差异，未采用；本次仅增加独立进程并发。GPU2/3任务未改动。

[完整测试EER表](reports/round1-batch-20261113/TEST_EER.md)每30秒本地更新，只接受71237/148176/31779条全量覆盖、失败0并齐全三个seed的报告。每个数据集的36份报告完成后自动同步小报告到GitHub；进程结束或失败也更新。原始音频、权重、观测包和逐句大文件留本地。

[开发验证EER](reports/round1-batch-20261113/DEVELOPMENT_EER.md)：本体0.029234%、全视图MLP0.019376%，未显示本体开发增益，不能视为测试结果。专家开发曝光、RawNet2监督未知和两个来源组件的限制保留。

[操作说明](docs/RUN_EXPERIMENTS.md)、[批量报告](reports/round1-batch-20261113/REPORT.md)。固定测试入口与全量覆盖汇总相关回归已通过，命令/退出码见fixed-model-test-regression.command.json和full-test-report-regression.command.json；最新完整回归182 passed、0 skipped、退出0；命令/日志见ssd-test-full-regression.command.json及.log。无需SHA或文档封存。
