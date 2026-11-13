# 运行实验

常规实验只需要配置、数据划分、真实专家权重和运行环境。无需计算文件/源码SHA，无需先审核文档或生成质量封存报告。历史模型和缓存仍可读取。

## 先准备

在项目根目录激活已有py310环境。检查`configs/local.yaml`中的数据、专家源码和权重路径；默认设备为`cuda:0`。`--device`可覆盖配置，指定CUDA但不可用时明确报错。

每个实验使用独立工作目录，例如`reports/my-experiment`；缓存位置来自配置，也可通过`--cache-dir`指定。

```bash
python -m ontoresponse doctor
python -m ontoresponse --help
```

## 用已有数据清单训练与评价

工作目录下的`audit_manifest.jsonl`是数据划分表，沿用旧文件名，不是校验清单。每行包含`id, origin_id, source_group, speaker_id, audio, label, role`；可加`condition`。`role`使用`fusion_train / fusion_val / calibration / test`，同源及派生录音不能跨角色。0是真、1是假。

`inference_manifest.jsonl`仅保留`id, origin_id, source_group, audio`四个字段，保证专家推断不读标签。

```bash
python -m ontoresponse extract --work-dir reports/my-experiment --max-origins 24
python -m ontoresponse train --work-dir reports/my-experiment --model response_mlp --steps 20
python -m ontoresponse calibrate --work-dir reports/my-experiment --checkpoint reports/my-experiment/runs/response_mlp-fixed-seed17/model.pt
python -m ontoresponse evaluate --work-dir reports/my-experiment --checkpoint reports/my-experiment/runs/response_mlp-fixed-seed17/model.pt
```

`evaluate`只读取清单中的test角色；使用独立测试清单/提取索引时传入`--manifest`和`--index`。不需要`--lock`或单独执行`freeze`。旧freeze接口保留供读取历史实验和可选保存评价设置使用。

训练默认遵循smoke预算：最多64条源录音、20步。这是运行检查，不是正式效果验证。pilot仍需显式配置预算；数据来源和分组不足时，应调整实验设计。这里没有自动启动全量训练。

## 快速验证本地链路

没有自定义清单时，可先用`prepare-data --work-dir reports/my-smoke --max-origins 24`生成ASV2019 dev的有界训练/验证/校准子集，再按上述前三步运行。它没有test角色，而且公开专家可能已用过dev选模型，因此只能说明集成链路可运行。

同一缓存比较`original_mlp`、`all_views_mlp`、`response_mlp`、`set_attention`、`untyped_response`与`typed_response`；保持样本、参数信息和训练预算一致。GPU用于专家提取、融合训练和推断。改变模型时用不同模型目录；改变数据/实验设置时用新工作目录。训练恢复使用`train --resume PATH`。

## 保存什么

- 数据划分表和本次配置。
- `model.pt`、`history.jsonl`或`convergence.jsonl`、验证指标。
- `calibration.json`与测试`per_utterance_predictions.jsonl`、`report.json`。
- 必要的命令、退出码、耗时和设备信息。

缓存内部仍使用稳定短标识区分视图，噪声种子和重复音频识别也需要稳定标识；它们不用于文件审计或阻止代码修改。缓存以路径、大小、修改时间和`runtime.cache_revision`区分输入。修改专家权重、预处理或提取实现后，应使用新缓存目录或增加`cache_revision`，避免复用旧观测。

SHA清单、依赖逐文件校验、源码质量封存不是实验前置条件。旧审计工具仅保存在`docs/history/tools/`，历史报告不重写。GitHub导出只列文件名/大小，默认每文件最多1MiB，音频、权重和缓存留本地。
