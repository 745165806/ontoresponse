# CLI与运行入口

按[运行实验](RUN_EXPERIMENTS.md)开始。默认配置为configs/local.yaml，默认GPU为cuda:0，默认模型response_mlp。使用python -m ontoresponse --help查看参数。

| 用途 | 命令 |
|---|---|
| 环境检查 | doctor（audit为兼容别名） |
| 数据准备 | prepare-data、prepare-external、make-environments |
| 真实专家/观测 | check-experts、extract、merge |
| 融合训练 | train、ablate |
| 校准与测试 | calibrate、evaluate |
| 单条推断/指标重算/源码导出 | infer、report、export |
| 可选旧接口 | freeze、audit-export、export-ontology |

通用参数：--config、--work-dir、--cache-dir、--device、--manifest、--index、--output。--work-dir覆盖配置的output，便于新建实验；--device控制专家与融合模型计算设备。extract的--output指定索引文件路径。

训练参数：--model、--steps、--seed、--width、--resume。默认最多64源录音、20步；pilot需要显式配置，完整专家benchmark有独立全量推断预算，不能把有界融合实验误称正式全量研究。

评价/推断使用--checkpoint、--audio。evaluate可直接运行，不再强制--lock或人工freeze；若提供旧lock，仍检查请求、模型设置及校准的一致性，不读取文件SHA。模型设置变化使用新实验目录，已评价的历史目录不覆盖。

提取分片使用--shard-index、--shard-count，合并用--indices。可选本体导出支持--shacl；SHACL缺依赖只影响该可选导出，不阻止普通训练。

pilot预检只检查实际输入、分组、设备和预算，不再要求上游文档hash、质量封存或预注册文件。独立脚本见[SCRIPTS.md](SCRIPTS.md)，复杂数据安排见[DEVELOPMENT_V2.md](DEVELOPMENT_V2.md)。
