# 运行实验

常规实验只需要配置、数据划分、真实专家权重和运行环境。无需计算文件/源码SHA，无需先审核文档或生成质量封存报告。历史模型和缓存仍可读取。

## 第一轮现在怎么启动

用户已恢复2019 LA eval测试。2019 train/dev用于开发，正式测试为2019 LA eval、2021 LA eval阶段和In-the-Wild。先用真实来源分组的64条开发录音检查三专家链路，再扩展pilot和正式训练。

在项目根目录激活环境：

```bash
source ${EXTERNAL_PATH}
conda activate py310
```

新实验首先准备清单（只读2019 train/dev协议；从本地官方映射归并目标与VC来源说话人，不读测试音频或分数）：

```bash
python -m scripts.prepare_round1 --work-dir reports/round1-smoke/fold0 --device cuda:0
```

本机`reports/round1-smoke/fold0`已准备好，跳过上述准备命令；重复准备会拒绝覆盖，另起实验应更换工作目录。默认32条训练、32条验证，各16真/16假；每角色只有一个连接组件，仅供工程smoke。官方来源映射默认使用本地`reports/expert-benchmark-20261111/ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz`，其他机器可通过`--source-map PATH`指定；不自动下载数据。

先确认cuda:0仍空闲，然后依次执行（配置包含AASIST、SSL+AASIST、RawNet2，原始+六探针）：

```bash
python -m ontoresponse extract --config reports/round1-smoke/fold0/config.yaml --max-origins 64
python -m ontoresponse train --config reports/round1-smoke/fold0/config.yaml --model original_mlp --steps 20
python -m ontoresponse train --config reports/round1-smoke/fold0/config.yaml --model response_mlp --steps 20
python -m ontoresponse train --config reports/round1-smoke/fold0/config.yaml --model typed_response --steps 20
```

只提取一次，三个融合模型共享观测；各模型输出在该工作目录的`runs/<model>-fixed-seed17/`。检查提取失败数、loss/验证指标及checkpoint保存。20步、单seed和64条录音不支持方法优越性结论；相同width也不等于参数量匹配，正式对照仍需容量和预算控制。来源未知的RawNet2可用于明确标注的integration smoke，不能据此宣称严格来源独立。

可另用`--fold 1 --work-dir reports/round1-smoke/fold1`准备同一批64条录音的反向角色，分别训练，不能共用已拟合的标准化或融合权重。三个测试集均不参与此开发过程。

已有独立批量入口，见下节；普通CLI仍有smoke/pilot预算，不能把`--steps`调大冒充正式全量训练。本轮不要求calibration，EER直接从logit计算。

## 自主批量实验（已启动）

配置`configs/round1-batch.yaml`明确授权完整运行，使用cuda:0和cuda:1；其余GPU不使用。流程依次为：

1. 1024条录音pilot：两折、seed 17/29/43，12种模型/对照，每个神经模型最多320步。
2. 50224条完整开发录音：按两个来源连接组件进行两折选择；最多30 epoch/15000步，早停只看开发BCE。
3. 根据两折、三seed的最佳开发epoch中位数，为每种架构固定重训轮数，然后使用完整开发池重训。强基线采用接近typed模型的参数量，训练预算相同。
4. 所有设置和最终权重生成后，依次完整评测2019 LA eval（71237）、2021 LA eval阶段（148176）及ITW（31779）。2021另外报告七种codec条件，progress/hidden不混入eval。

运行目录`reports/round1-batch-20261113`。每块最多256条，提取只读无标签清单，校验同专家/同视图后将embedding、margin、mask、质量与对应metadata打包为evidence.pt；删除该块临时波形缓存，保留原始音频。CPU只做准备和指标，神经计算使用CUDA。保留100 GiB磁盘余量；单块3600秒超时。部分块/模型完成后可继续，失败记录保留，不悄悄删除困难样本。

任务已在独立后台进程启动，无需再次运行。查看进度：

```bash
cat reports/round1-batch-20261113/progress.json
tail -n 20 reports/round1-batch-20261113/runner.log
cat reports/round1-batch-20261113/pilot/progress.json
```

若进程已退出，需要继续同一配置时运行：

```bash
python -u -m scripts.round1_batch --config configs/round1-batch.yaml
```

锁文件防止同一目录重复启动；修改配置需新目录。已完成的块和模型不重算；被中断的单模型开发训练保留旧目录后从头运行，尚不支持在该模型的epoch中间恢复。选择保存在selection.json；逐句测试logit和report.json在tests/<dataset>/<model>-seed<seed>/。统计来源独立性未知、专家开发曝光和仅两个开发组件的限制仍存在；这些结果属于集成研究，不自动升级为research_holdout。

## 当前模型的完整测试

36个最终checkpoint已保存于`reports/round1-batch-20261113/final`，清单为models-ready.json。用户指定固定当前模型，在2019 LA eval、2021 LA eval阶段和ITW进行完整测试，入口为：

```bash
python -u -m scripts.round1_batch --config configs/round1-batch.yaml --test-only --workers-per-gpu 2 --scratch-dir /tmp/ontoresponse-round1-cache --batch-profile reports/batch-throughput-20261114/selected-policy.json
```

该任务已启动，勿重复运行。纯测试入口核对训练时配置和完整checkpoint，不进入开发训练或最终重训；GPU0/1各2个进程，AASIST/SSL+AASIST/RawNet2实际推断batch分别32/32/64。每条录音的原始/六探针保持同样裁剪和聚合。批量输出与同batch原生上游精确一致，开发融合决策稳定；与batch1存在已记录的浮点差异，不保证测试EER逐位相同。已完成的观测块复用；临时缓存放本地NVMe SSD（/tmp/ontoresponse-round1-cache），打包后删除临时缓存，最终观测包和预测留项目目录。已有部分缓存可保留原位继续，或停止自己的任务后将其整体迁到SSD；缓存键和数值内容不变。只改变推断batch、并发和临时存储位置，不变更模型权重、crop、探针或阈值。实测吞吐和误差见[批量加速报告](../reports/batch-throughput-20261114/REPORT.md)。

完整EER汇总：`reports/round1-batch-20261113/TEST_EER.md`及.json。它仅接受2019=71237、2021 LA eval=148176、ITW=31779条完整覆盖、失败0的报告；每种方法三个seed齐全才汇总均值与样本标准差。未完成显示空白，不以开发EER或小子集EER代替。每个数据集的36份报告完成即自动同步小型结果到GitHub，最终状态或失败状态也同步。完整逐句logit留在本地tests/<dataset>/<model>-seed<seed>/。

可手动重算汇总：

```bash
python -m scripts.report_round1_tests --root reports/round1-batch-20261113
```

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
