# 实施与复现

本实现只做整句二分类：0 bonafide，1 spoof。固定专家返回整句 hidden 和 spoof−bonafide margin；已知探针作用于完整当前输入；原始/探针共享 crop、eval、聚合和 fusion_train scaler。原始+响应节点经两层类型关系计算后直接 BCE 输出任意实值 logit。旧 risk/QP 与局部定位未进入实现。

## 本地运行

从项目根目录执行，优先使用已存在的 py310；无需修改上游工程。机器路径集中在 `configs/local.yaml`，探针参数仍以 `configs/probes.yaml` 为唯一来源。以下是历史smoke复现流程，已有output不可覆盖。当前GPU和来源状态见根目录[STATUS.md](../STATUS.md)；旧`gpu-access.json`仅代表当时沙箱条件。

```bash
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse doctor
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse prepare-data --max-origins 24
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse check-experts
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse extract --max-origins 24
${USER_HOME}/anaconda3/envs/py310/bin/python -m scripts.run_bounded
```

`run_bounded` 先完成普通基线，再跑关系模型；每个模型/seed 最多20步、3个固定seed；所有模型共享同一缓存、源版本权重、train/val/calibration和选模规则。记录 fixed-width、与typed参数量接近的matched-width、更大width=128强基线。参数匹配通过维数/参数计数完成，不读标签选宽度。`no_delta_mlp` 是与全视图MLP同信息的绝对观测控制；它与同配置all_views_mlp的计算相同，不能据此声称差分增加信息。

8条外部原录音在预测前固定，生成 no-new-processing / AAC / Opus 共24条当前输入；全部仍保留原origin与speaker/source_group。在训练选择结束后再校准、冻结模型/输入音频hash/阈值/探针及本体，再提取和评价外部音频：

```bash
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse prepare-external --max-origins 8
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse make-environments
${USER_HOME}/anaconda3/envs/py310/bin/python -m scripts.evaluate_bounded
```

完整顺序复现入口 `python -m scripts.reproduce_smoke` 记录每条命令、退出码、耗时和日志。冻结文件不允许在内容不同的情况下覆盖；需要新试验时复制local配置并指定新输出目录，运行各脚本时传入`--config /path/to/new.yaml`。再次训练保存的checkpoint若hash变化，必须新建实验目录，不可覆盖已经看过外部结果的锁。

可独立运行的 `calibrate / freeze / evaluate / infer / report` 参数见 `python -m ontoresponse --help`。例如：

```bash
python -m ontoresponse infer --audio /absolute/audio.wav --checkpoint reports/implementation/runs/typed_response-fixed-seed17/model.pt
python -m ontoresponse report --manifest reports/implementation/runs/typed_response-fixed-seed17/external-evaluation/per_utterance_predictions.jsonl --output /tmp/recomputed-report.json
```

## 实现边界

- 专家接口只有 `AudioInput(id, waveform)`；不接受label、speaker、attack或历史环境字段。auditor从真实checkpoint提取训练/选模来源；加载strict=True。SSL仅在内存中替换通用frontend路径、移除版本不兼容的task配置字段；不改网络参数或上游磁盘文件。先迁移完整模型device/dtype再eval，推断前后检查所有module均为eval，避免上游内部`.train()`分支。
- RawNet2为本机detectionLib中的实际源码/权重组合；与此实现直接输出一致，但其原始训练/选模历史仍未知，不能称官方checkpoint或严格holdout。
- 缓存key含原文件/波形hash、完整操作参数、独立seed规则、checkpoint/源码/frontend hash、crop/聚合、执行代码与schema hash。各专家共享同一probe artifact；相同数值但不同专家保留不同测量key。原子NPZ、完整性hash、重复ID拒绝、失败状态和覆盖率均保留。
- MP3/AAC/Opus调用真实ffmpeg。容器解码长度差时，以同codec/码率/长度的独立脉冲控制验证固定delay；仅在可验证时裁尾/移除delay，并记录前后长度、命令、控制脉冲位置与测得lag。校验失败保持missing，无替代codec。
- `OntologyRegistry`使用同一YAML编译数值边与真实观测来源RDF；Python检查专家版本、crop、score方向、当前录音派生与同probe artifact。SHACL引擎仍需要可用的pyshacl；缺依赖是BLOCKED，不能写通过。
- CPU分片函数和严格合并接口已经实现；真实四卡执行及与单卡的数值对照尚未验证。
- 外部JSONL每条都有status、logit/p或null、固定开发阈值、来源组、条件、checkpoint和registry hash。报完整请求数、有效子集指标、失败政策上下界、每条件EER及条件均权macro EER。bootstrap按source/speaker组，同时抽取所有派生条件；多seed分别报告，不扩大独立样本数。
- ASV2021 DF全量官方包装器强制ID全集/有效状态，依官方方向取负spoof margin；没有执行全量。官方来源archive已取得，但认证新分组及任务监督来源仍须单独审计。ASV5本地只有协议，音频/工具未锁定；不自动下载。

## 当前研究限制

24条ASV2019 dev只用于8条fusion_train、8条fusion_val、8条calibration；8条In-the-Wild只作集成smoke。AASIST/SSL专家训练与该fusion_train的文件ID无重叠，但上游source_val分别与fusion_val、calibration重叠2条。后续取得的官方映射又发现2019 train/dev有1043条共同声学来源。因此该结果只能是integration_pretrained，不是research_holdout。未知处理仅称“融合训练未见AAC/Opus算子”，不声称所有专家/通用SSL彻底未见。新数据安排见[融合数据方案](FUSION_DATA_RECOMMENDATION_20261111.md)。
