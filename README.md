# OntoResponse v0.5 — 整句语音鉴伪的处理响应融合

已实现真实音频与真实鉴伪权重的有界工程管线：多个冻结专家的整句 embedding / spoof margin，加上已知诊断处理前后的响应，通过类型化关系直接 BCE 训练一条录音一个真假 logit。0=bonafide，1=spoof。

**当前已完成 integration_pretrained smoke 和128-origin有界pilot，尚不满足严格独立的 research_holdout，也不构成本体增益结论。** 不使用误差矩阵、QP加权或局部伪造定位。

## 最新验证：2026-11-11 完整GPU专家评测

已使用 NVIDIA RTX A6000 GPU0/1/3，对完整 ASVspoof2019 LA eval 的71237条录音分别测试，所有请求均有输出、失败0。主指标使用作者原生 bonafide score 和官方 `compute_eer`，分数方向/权重/裁剪在运行前固定。

| checkpoint | EER (%) |
|---|---:|
| 现用 AASIST | 2.121052 |
| 现用 SSL+AASIST | 0.637284 |
| 现用 RawNet2 | 5.084675 |
| AASIST 作者发布参考权重 | 0.829511 |

作者AASIST参考复现官方0.83%，现用本地AASIST尚未达到该水平。SSL论文0.82%对应2021 LA，不能与本轮2019结果直接比较；RawNet2本地配置/来源不同，不能单凭EER认定论文复现。完整unit/synthetic为171 passed、0 skipped，真实GPU/checkpoint/数据证据分别记录。

官方2019/2021来源映射现已取得并核验完整ID覆盖。新审计发现2019 train/dev共享1043条声学来源；本地Codecfake官方train/dev也共享说话人，并有采样率差异。融合full训练尚未启动，近期建议Codecfake中文来源按speaker/source重分组、统一16kHz；ASV5作为后续现代路线，本地尚无其音频。

详见[完整评测与论文对照](reports/expert-benchmark-20261111/RESULTS.md)、[结果汇总](reports/expert-benchmark-20261111/summary.csv)、[融合数据建议与GPU全量方案](docs/FUSION_DATA_RECOMMENDATION_20261111.md)。旧pilot的最新解释为区分能力不足、H1未决，尚不能证明本体增益。

本次2026-11-11提交同步文档与小型汇总证据。171项测试及完整GPU benchmark均在研究工作区执行，执行源码hash见新报告的`preflight.json`；本次没有整体同步其后续runner源码或详细逐条分数。公开汇总的可复核范围见[发布范围](reports/expert-benchmark-20261111/PUBLICATION_SCOPE.md)。

## 可运行入口

在项目根目录使用现有py310，无需改上游代码或下载数据：

```bash
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse doctor
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse prepare-data --max-origins 24
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse check-experts
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse extract --max-origins 24
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse train --model response_mlp
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse train --model typed_response
```

完整有界比较与冻结外测顺序见 [实施说明](docs/IMPLEMENTATION.md)。一键入口为 `python -m scripts.reproduce_smoke --config configs/local.yaml`；已经冻结并评价的试验应保留原锁，新运行使用新output目录。CLI亦支持独立calibrate、freeze、evaluate、ablate、infer、report及缓存分片/严格合并。

## 历史smoke已验证

- AASIST、SSL+AASIST、RawNet2真实权重/源码已固定hash；hidden维数160/160/1024，与直接上游输出一致；identity和无损控制响应为零。
- 24条ASV2019 dev，fusion_train / fusion_val / calibration各8条；六探针共享artifact，504项有效expert-view测量。
- 普通融合先行，完成同信息MLP、typed/untyped/attention、错误关系和删边等78个有界训练运行；包含3个seed、参数匹配和更大容量基线，每个最多20步。
- 8条In-the-Wild录音的原版/AAC/Opus共24个外部请求，在模型/校准/阈值/音频内容冻结后全部有输出；结果与配对统计见 `reports/implementation/external-results.json`。小样本不能支持研究效果声明。
- 单元/合成端到端测试与真实checkpoint、真实数据结果分别记录；完整状态和失败事项见 [STATUS.md](STATUS.md)。

## 历史pilot与后续接口

新增四通道 `relation_tied`、完整SHACL正负例、独立预测重算、冻结保护与仓库外wheel验证。详见 [下一阶段实施](docs/NEXT_STAGE_IMPLEMENTATION.md)。新pilot排除未知来源RawNet2及上游选模speaker；原8条外测已消费，不再用于选择。

`CODEX_NEXT_STAGE.md`已读取；该历史阶段完整测试123项通过、无跳过，后续阶段测试结果见最新状态。该pilot为两个专家、3seed、64/24/16/24分组、100epoch/400step上限，15个可训练配置实际160–400步。原始与响应融合验证EER均0，按当时预设门未进入H2；后续诊断将其解释为 `INSUFFICIENT_DISCRIMINATION`、H1未决。24条测试样本只有3个speaker组/1个spoof组，天花板结果不能证明研究效果；这些确认样本也已消费。

完整分层状态见 [STATUS_NEXT.md](STATUS_NEXT.md)。可分享的脱敏证据包为 `reports/codex-next-stage-20261110/audit_bundle.zip`：解压后在解压目录运行 `python3 audit_recompute.py`，标准库独立重算576条预测，不需要音频/模型。包内源码使用路径模板，真实提取需配置本地资产；私有历史tar不用于分享。当前CLI和依赖安装说明见 [CLI_CURRENT.md](docs/CLI_CURRENT.md)。

## 仍然 BLOCKED

本轮完整GPU专家评测已经完成，2019/2021官方来源映射也已取得；四卡提取/合并一致性仍未验证。任务HPO/通用SSL曝光及RawNet2监督来源仍未认证，现用AASIST质量与新融合分组仍需修正，严格research_holdout尚不满足。ASV2021官方全测、ASV5确认和新的融合full训练未执行。GitHub公开仓库：[745165806/ontoresponse](https://github.com/745165806/ontoresponse)。发布内容为脱敏源码与审计证据，音频、模型权重和缓存另行配置；正式融合方案仍需来源/预算/预注册条件通过后执行。

方法、研究协议和验收要求分别见 [METHOD](docs/METHOD.md)、[EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md)、[ACCEPTANCE](docs/ACCEPTANCE.md)。保留的 `reference/` 为数值核，生产输入来自 `ontoresponse/` 的真实音频提取；tests中的合成fixture从不替代真实专家。
