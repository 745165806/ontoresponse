# 当前状态 — 2026-11-13

用户已授权自主执行大批量GPU实验，并恢复ASVspoof2019 LA eval测试。2019 train/dev用于开发；2019 LA eval、2021 LA eval阶段与In-the-Wild仅在设置确定后测试。Codecfake延后。

| 层级 | 实际状态 |
|---|---|
| implemented | 新增分块提取/压缩观测、GPU分批训练和验证、两折选择、固定轮数重训、2021 LA按eval阶段读取、三测试集独立评价 |
| unit/synthetic | 177 passed、0 skipped、退出0；日志reports/round1-batch-20261113/regression.log |
| real-checkpoint / GPU | 三专家原实现兼容性已有验证；本次64条×3专家×7视图共1344个观测有效，失败0 |
| real-data / GPU smoke | 32训练+32验证，original_mlp/response_mlp/typed_response各20步均成功，save/load精确一致；验证EER全0，仅工程结果 |
| running | 后台PID见reports/round1-batch-20261113/launch.json；GPU0/1处理1024条pilot；首512条（两块）已打包，10752个观测有效且跨GPU专家身份一致，随后自动进入完整开发及测试 |
| research | 完整结果尚未产生；不能据smoke宣称本体有效 |

批量配置configs/round1-batch.yaml：先1024条两折/三seed/12模型pilot，再50224条开发录音的两折选择及完整重训；测试71237条2019 LA eval、148176条2021 LA eval、31779条ITW。输入按协议取样，不混合2021 progress/hidden。原始数据、权重和大缓存仅本地保存。

运行/恢复步骤见[运行实验](docs/RUN_EXPERIMENTS.md)，命令、资源和限制见[批量报告](reports/round1-batch-20261113/REPORT.md)。总进度progress.json、提取进度pilot/progress.json、日志runner.log；失败不计作通过，测试尚未开始。

限制：现用专家已使用2019开发数据，RawNet2监督来源未知；保守来源连接仅两个开发组件，结论属于integration_pretrained。三份现用权重不能都声称复现论文；历史专家评测见reports/expert-benchmark-20261111/RESULTS.md。2021 LA/ITW来源独立性及历史消费限制继续披露。取消hash/SHA或文档审核不改变这些实验限制。

常规实验无需文件/源码SHA审计。保留配置、分组、checkpoint、曲线、逐句预测和指标；GitHub仅同步代码、说明及小型重要报告。
