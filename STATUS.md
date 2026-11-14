# 当前状态 — 2026-11-14

2019 LA eval已完成71237条录音及36个模型评分，失败0。三个seed的本体融合EER为0.348615% ± 0.125739%，响应MLP为0.216023% ± 0.012425%，同邻接共享关系模型为0.343384% ± 0.076784%；本体未优于这些强对照，保留固定模型继续外测。切换2021 LA后出现音频解码异常，已验证兼容解码并恢复GPU0/1；ITW随后进行。全部测试使用已保存的36个最终模型，不进入训练或据测试EER重新选择。

| 阶段 | 实际状态 |
|---|---|
| 1024条pilot | 4/4块提取、72次两折/三seed比较完成 |
| 50224条完整开发数据 | 197/197块提取、另72次比较完成，开发选择已固定 |
| 最终模型 | 36/36 checkpoint已保存，models-ready.json记录完整清单 |
| 2019 LA eval | 279/279块、71237/71237条、36/36模型完整报告，失败0；EER已发布 |
| 2021 LA eval | 原首4块有389条soundfile解码失败；修复后这4块1024条、21504项观测全部有效，GPU0/1继续后续分块；总量148176条/579块，实时进度见progress.json |
| In-the-Wild | 尚未开始，总量31779条/125块，等待2021阶段完成 |

当前命令：`python -u -m scripts.round1_batch --config configs/round1-batch.yaml --test-only --workers-per-gpu 2 --scratch-dir /tmp/ontoresponse-round1-cache --batch-profile reports/batch-throughput-20261114/selected-policy.json --dynamic-queue`。PID见launch.json，总进度progress.json，逐数据集进度test_evidence/<dataset>/progress.json，日志runner.log。已仅重启本项目调度器，复用已完成观测包和部分缓存；提取任务独占领取，同一录音全部视图保持在同一任务中。后续36个模型按完整checkpoint动态分配GPU0/1，各自在完整测试集计算EER。临时缓存使用本地SSD；每卡2进程，真实推断batch为AASIST32、SSL+AASIST32、RawNet2-64。音频处理/缓存打包期间GPU利用率仍可能下降。上游权重、crop和FP32不变。与相同batch原生模型输出精确一致；已有开发样本的12个融合模型决策不变。原batch1存在浮点差异，详见[加速报告](reports/batch-throughput-20261114/REPORT.md)，不承诺旧/新EER完全一致。GPU2/3任务未改动。

[完整测试EER表](reports/round1-batch-20261113/TEST_EER.md)每30秒本地更新，只接受71237/148176/31779条全量覆盖、失败0并齐全三个seed的报告。每个数据集的36份报告完成后自动同步小报告到GitHub；进程结束或失败也更新。原始音频、权重、观测包和逐句大文件留本地。

2021恢复详情见[解码修复报告](reports/flac-recovery-20261114/REPORT.md)：原有soundfile可正常读取的路径保持原样；仅FLAC解码报错时使用严格FFmpeg，拒绝任何解码错误或与流元数据不符的帧数，不重采样。389条失败录音及64条成功对照共453条与独立SoX完整PCM逐样本一致。16条原失败录音的实际GPU提取336观测、关系/缓存读取均通过。原失败摘要和日志保留，2019结果与所有已完成观测复用，未删测试请求。

[开发验证EER](reports/round1-batch-20261113/DEVELOPMENT_EER.md)：本体0.029234%、全视图MLP0.019376%，未显示本体开发增益，不能视为测试结果。专家开发曝光、RawNet2监督未知和两个来源组件的限制保留。

[操作说明](docs/RUN_EXPERIMENTS.md)、[批量报告](reports/round1-batch-20261113/REPORT.md)。双卡调度14项通过；真实checkpoint在16条既有开发证据上同时运行GPU0/1，最大logit差异0，见reports/batch-throughput-20261114/two-gpu-fusion-check.json。最新完整回归192 passed、0 skipped、退出0，含5项解码兼容及损坏/帧数/采样率保护检查，见reports/flac-recovery-20261114/full-regression.command.json/.log。真实解码、真实GPU提取和完整测试效果分开记录；2021/ITW完整EER尚未生成。无需SHA或文档封存。
