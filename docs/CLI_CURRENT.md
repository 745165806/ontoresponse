# CLI与运行入口

使用`python -m ontoresponse --help`查看当前完整参数。主CLI保留原命令与参数兼容；默认配置为`configs/local.yaml`，默认模型为`response_mlp`。

| 用途 | 命令 |
|---|---|
| 环境/来源 | `doctor`、`audit` |
| 数据准备 | `prepare-data`、`prepare-external`、`make-environments` |
| 专家/证据 | `check-experts`、`extract`、`merge` |
| 本体 | `export-ontology` |
| 融合训练 | `train`、`ablate` |
| 校准与冻结评价 | `calibrate`、`freeze`、`evaluate` |
| 单条推断/重算/导出 | `infer`、`report`、`audit-export` |

通用路径参数：`--config`、`--manifest`、`--index`、`--output`。训练预算参数：`--max-origins`、`--steps`、`--seed`、`--model`。提取分片：`--shard-index`、`--shard-count`；合并：`--indices`。推断/评价使用`--audio`、`--checkpoint`、`--lock`；本体引擎可用`--shacl`。

## 独立实验入口

```bash
python -m scripts.expert_benchmark --help
python -m scripts.run_pilot --help
python -m scripts.prepare_connected --help
python -m scripts.materialize_development --help
python -m scripts.freeze_development --help
```

完整专家benchmark有独立的明确full-evaluation预算，不能把主CLI的smoke/pilot预算改掉当作全量融合训练。pilot仅在显式enabled且来源、质量、预算和预注册门通过后执行。已有冻结目录不覆盖。

`full_attention`为完整自注意力TransformerEncoder对照；`untyped_union`合并邻接；`relation_tied`保留四通道及逐关系归一化、仅共享矩阵。旧名字继续兼容保存的checkpoint。

数据开发接口见[DEVELOPMENT_V2.md](DEVELOPMENT_V2.md)，全部保留脚本见[SCRIPTS.md](SCRIPTS.md)，最新状态见[STATUS.md](../STATUS.md)。
