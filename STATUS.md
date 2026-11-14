# 当前状态 — 2026-11-14

正在使用已保存的36个最终模型进行完整测试。已完成最终重训，当前进程为纯测试入口，不进入开发训练或重训。2019 LA eval、2021 LA eval和In-the-Wild分别评测；完整测试EER尚未产生。

| 阶段 | 实际状态 |
|---|---|
| 1024条pilot | 4/4块提取、72次两折/三seed比较完成 |
| 50224条完整开发数据 | 197/197块提取、另72次比较完成，开发选择已固定 |
| 最终模型 | 36/36 checkpoint已保存，models-ready.json记录完整清单 |
| 固定模型测试 | GPU0/1各2个提取进程，通过共享队列动态领取未完成块，实际专家batch=32/32/64；2019动态调度验证快照238/279块，2021/ITW按流程继续；实时进度以progress.json为准 |
| 测试EER | 尚无全量、三个seed齐全的结果；不以开发或部分数据EER替代 |

当前命令：`python -u -m scripts.round1_batch --config configs/round1-batch.yaml --test-only --workers-per-gpu 2 --scratch-dir /tmp/ontoresponse-round1-cache --batch-profile reports/batch-throughput-20261114/selected-policy.json --dynamic-queue`。PID见launch.json，总进度progress.json，逐数据集进度test_evidence/<dataset>/progress.json，日志runner.log。已仅重启本项目调度器，复用已完成观测包和部分缓存；提取任务独占领取，同一录音全部视图保持在同一任务中。后续36个模型按完整checkpoint动态分配GPU0/1，各自在完整测试集计算EER。临时缓存使用本地SSD；每卡2进程，真实推断batch为AASIST32、SSL+AASIST32、RawNet2-64。音频处理/缓存打包期间GPU利用率仍可能下降。上游权重、crop和FP32不变。与相同batch原生模型输出精确一致；已有开发样本的12个融合模型决策不变。原batch1存在浮点差异，详见[加速报告](reports/batch-throughput-20261114/REPORT.md)，不承诺旧/新EER完全一致。GPU2/3任务未改动。

[完整测试EER表](reports/round1-batch-20261113/TEST_EER.md)每30秒本地更新，只接受71237/148176/31779条全量覆盖、失败0并齐全三个seed的报告。每个数据集的36份报告完成后自动同步小报告到GitHub；进程结束或失败也更新。原始音频、权重、观测包和逐句大文件留本地。

[开发验证EER](reports/round1-batch-20261113/DEVELOPMENT_EER.md)：本体0.029234%、全视图MLP0.019376%，未显示本体开发增益，不能视为测试结果。专家开发曝光、RawNet2监督未知和两个来源组件的限制保留。

[操作说明](docs/RUN_EXPERIMENTS.md)、[批量报告](reports/round1-batch-20261113/REPORT.md)。固定测试入口与全量覆盖汇总相关回归已通过，命令/退出码见fixed-model-test-regression.command.json和full-test-report-regression.command.json。批量专家真实音频16条336观测通过。双卡调度相关14项通过；真实checkpoint在16条既有开发证据上同时运行GPU0/1，最大logit差异0。最新完整回归187 passed、0 skipped、退出0，见reports/batch-throughput-20261114/two-gpu-full-regression.command.json/.log；双卡验证见two-gpu-fusion-check.json。全量测试仍运行中，尚无正式测试EER。无需SHA或文档封存。
