# 当前状态 — 2026-11-13

最新调整：2019 LA eval已从第一轮评测方案移除，原始数据和历史benchmark保留。测试仅2021 LA eval及In-the-Wild。新增`scripts.prepare_round1`，已执行`python -m scripts.prepare_round1 --work-dir reports/round1-smoke/fold0 --device cuda:0`（py310解释器，退出0）；开发池50224条，准备32训练/32验证，各16真16假、各一个保守来源组件，结果见`reports/round1-smoke/fold0/preparation.json`。神经计算配置为CUDA，但本轮仅准备数据，未启动提取/训练/外测。`pytest -q tests/test_round1_prepare.py tests/test_connected_split.py tests/test_pilot_budget.py`为12 passed、0 skipped、退出0；首次测试命令误指不存在的test_data.py退出4，纠正后通过。GPU查询时0/1空闲、2/3忙，未影响任何进程。入口及后续限制见[运行实验](docs/RUN_EXPERIMENTS.md)。

前一轮数据检查：已用python3读取本地协议/CSV、逐ID检查文件存在（退出0，CPU，无音频解码或GPU运行）：2019 train/dev/eval为25380/24844/71237条，2021 LA完整keys为181566条（eval148176、progress16464、hidden16926），ITW为31779条，路径缺失均0。初次`python`命令因PATH未激活返回127，改用python3完成；未改动环境。Codecfake延后。完整数据说明见[更新后的数据方案](docs/FUSION_DATA_RECOMMENDATION_20261111.md)。

实现整句真假判别：0=bonafide，1=spoof。常规实验已移除文件/源码SHA审计、文档审核、质量封存和预注册文件门槛，优先使用配置、数据划分、checkpoint、曲线和指标。

| 层级 | 实际状态 |
|---|---|
| implemented | 专家提取、观测缓存、普通/关系融合、校准与评价；evaluate无需额外freeze |
| unit/synthetic | 私有与最终公开候选均172 passed、0 skipped |
| real-checkpoint / GPU | 三专家各8条既有录音比对通过；误差均小于3.34e-6 |
| real-data / GPU | 旧融合24条CPU精确一致、GPU误差小于3.82e-6；16源录音、2步融合训练和单录音三专家七视图提取通过 |
| research | 新full融合训练未执行；本体增益尚无充分实验验证 |

命令、退出码、日志与设备见[本轮报告](reports/simplify-experiments-20261113/REPORT.md)及同目录unit-final.command.json、gpu-check.command.json。GPU为A6000物理0；未影响其他GPU任务。本轮只做已消费数据的兼容性与有界运行检查，没有新的确认实验。

ASV2019 LA eval历史完整EER：现用AASIST 2.121052%、SSL+AASIST 0.637284%、RawNet2 5.084675%，作者AASIST参考0.829511%。不能宣称三份现用权重都复现论文。[完整报告](reports/expert-benchmark-20261111/RESULTS.md)。

2019来源重叠、公开专家选模型曝光、RawNet2来源未知及旧pilot独立组不足等限制仍保留。技术审计的删除不改变数据独立性；[数据方案](docs/FUSION_DATA_RECOMMENDATION_20261111.md)尚需落实，新数据不得直接随机逐文件切分。

操作见[运行实验](docs/RUN_EXPERIMENTS.md)。历史状态/工具移至docs/history，旧结果不重写；GitHub仅上传小型代码与报告，不上传音频、权重和缓存。

GitHub main已更新，包含精简代码、实验说明及小型报告；发布回执见reports/simplify-experiments-20261113/publish.json。提交/推送均退出0，远端分支已确认。
