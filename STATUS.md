# 当前状态 — 2026-11-14

正在使用已保存的36个最终模型进行完整测试。已完成最终重训，当前进程为纯测试入口，不进入开发训练或重训。2019 LA eval、2021 LA eval和In-the-Wild分别评测；完整测试EER尚未产生。

| 阶段 | 实际状态 |
|---|---|
| 1024条pilot | 4/4块提取、72次两折/三seed比较完成 |
| 50224条完整开发数据 | 197/197块提取、另72次比较完成，开发选择已固定 |
| 最终模型 | 36/36 checkpoint已保存，models-ready.json记录完整清单 |
| 固定模型测试 | GPU0/1各2个独立进程，实际专家batch=32/32/64，复用已完成块；2019启动快照为10/279块已打包，实时进度以progress.json为准；2021/ITW按流程继续 |
| 测试EER | 尚无全量、三个seed齐全的结果；不以开发或部分数据EER替代 |

当前命令：`python -u -m scripts.round1_batch --config configs/round1-batch.yaml --test-only --workers-per-gpu 2 --scratch-dir /tmp/ontoresponse-round1-cache --batch-profile reports/batch-throughput-20261114/selected-policy.json`。PID见launch.json，总进度progress.json，逐数据集进度test_evidence/<dataset>/progress.json，日志runner.log。原无参数基线优化器故障已修复，错误日志保留；恢复后重训成功。临时缓存已改为本地SSD，保留已完成观测包和部分计算。真实推断batch已增至AASIST32、SSL+AASIST32、RawNet2-64；上游权重、crop和FP32不变。与相同batch原生模型输出精确一致；已有开发样本的12个融合模型决策不变。原batch1存在浮点差异，详见[加速报告](reports/batch-throughput-20261114/REPORT.md)，不承诺旧/新EER完全一致。GPU2/3任务未改动。

[完整测试EER表](reports/round1-batch-20261113/TEST_EER.md)每30秒本地更新，只接受71237/148176/31779条全量覆盖、失败0并齐全三个seed的报告。每个数据集的36份报告完成后自动同步小报告到GitHub；进程结束或失败也更新。原始音频、权重、观测包和逐句大文件留本地。

[开发验证EER](reports/round1-batch-20261113/DEVELOPMENT_EER.md)：本体0.029234%、全视图MLP0.019376%，未显示本体开发增益，不能视为测试结果。专家开发曝光、RawNet2监督未知和两个来源组件的限制保留。

[操作说明](docs/RUN_EXPERIMENTS.md)、[批量报告](reports/round1-batch-20261113/REPORT.md)。固定测试入口与全量覆盖汇总相关回归已通过，命令/退出码见fixed-model-test-regression.command.json和full-test-report-regression.command.json；本轮批量相关18项通过、0 skipped，真实音频16条336观测通过；最新完整回归184 passed、0 skipped、退出0，见reports/batch-throughput-20261114/full-regression-final.command.json/.log。无需SHA或文档封存。
