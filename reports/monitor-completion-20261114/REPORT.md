# 定时监测与完整预测复算 — 2026-11-14

目标：完成三个测试集的固定36模型实验。健康GPU调度器持续运行，GPU0/1的batch32/32/64和每卡2worker保持固定；只更新30秒观察进程，PID/命令语义见monitor-start.json。每次报告同时读取实际已打包观测文件，计入末尾短块，显示各数据集已提取录音、块数、完整模型报告数与检查UTC。2021验证快照为28/579块、7168条；ITW等待后续，不将这些部分进度称为完整效果。

2019 LA eval的36份完整预测已逐句复算，71237条协议ID完整唯一、标签/source_group/condition/status正确，所有模型共用相同三专家原始margin；EER、AUROC、spoof-AUPRC与保存结果误差≤1e-12。命令和退出0见2019-prediction-recalculation.command.json/.log，耗时21.37秒。结果见[完整预测复算](../round1-batch-20261113/prediction-check-asv2019_eval.json)。没有读取或生成任何文件hash/SHA。

| 2019方法 | 主表ROC插值EER % | 上游nearest-DET EER % |
|---|---:|---:|
| typed_response（三seed均值） | 0.348615 | 0.348764 |
| response_mlp（三seed均值） | 0.216023 | 0.216781 |
| all_views_mlp（三seed均值） | 0.487071 | 0.485883 |
| relation_tied（三seed均值） | 0.343384 | 0.343889 |
| 原始AASIST margin | 2.080218 | 2.080308 |
| 原始SSL+AASIST margin | 0.611829 | 0.611947 |
| 原始RawNet2 margin | 5.084976 | 5.084675 |

原始专家一栏使用本轮共享观测的spoof-minus-bonafide margin，与历史单专家原生bonafide分数benchmark分别保留；本轮既有batch1观测包复用，新观测采用已记录批量设置。两种EER离散/插值约定有少量差异，主表定义保持不变，方向按既定spoof语义取反后传入上游，不根据测试标签反选方向或指标。

unit/synthetic：新增短末块乱序计数与完整预测检查的7项回归，含缺失、重复ID、错误标签、模型间专家观测不一致及保存指标不符；相关19项通过。完整回归199 passed、0 skipped、退出0，见full-regression.command.json/.log。真实完整数据复算与单元测试分开记录，不将未完成数据集计作通过。

观察进程会在2021/ITW各36份报告齐全后自动复算并发布小型结果；音频、权重、数值观测包和逐句大文件留本地。若进程/复算失败，保留日志并处理，不能以部分结果宣告实验完成。只有三数据集全量评分与预测复算均完成，才能结束当前目标。公共专家2019开发曝光、RawNet2监督来源未知等限制仍保留。
