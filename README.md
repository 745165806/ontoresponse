# OntoResponse v0.5 — 整句语音鉴伪的处理响应融合

已实现真实音频与真实鉴伪权重的有界工程管线：多个冻结专家的整句 embedding / spoof margin，加上已知诊断处理前后的响应，通过类型化关系直接 BCE 训练一条录音一个真假 logit。0=bonafide，1=spoof。

**当前已完成 integration_pretrained smoke 和128-origin有界pilot，尚不满足严格独立的 research_holdout，也不构成本体增益结论。** 不使用误差矩阵、QP加权或局部伪造定位。

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

## 本轮已验证

- AASIST、SSL+AASIST、RawNet2真实权重/源码已固定hash；hidden维数160/160/1024，与直接上游输出一致；identity和无损控制响应为零。
- 24条ASV2019 dev，fusion_train / fusion_val / calibration各8条；六探针共享artifact，504项有效expert-view测量。
- 普通融合先行，完成同信息MLP、typed/untyped/attention、错误关系和删边等78个有界训练运行；包含3个seed、参数匹配和更大容量基线，每个最多20步。
- 8条In-the-Wild录音的原版/AAC/Opus共24个外部请求，在模型/校准/阈值/音频内容冻结后全部有输出；结果与配对统计见 `reports/implementation/external-results.json`。小样本不能支持研究效果声明。
- 单元/合成端到端测试与真实checkpoint、真实数据结果分别记录；完整状态和失败事项见 [STATUS.md](STATUS.md)。

## 当前后续阶段

新增四通道 `relation_tied`、完整SHACL正负例、独立预测重算、冻结保护与仓库外wheel验证。详见 [下一阶段实施](docs/NEXT_STAGE_IMPLEMENTATION.md)。新pilot排除未知来源RawNet2及上游选模speaker；原8条外测已消费，不再用于选择。

`CODEX_NEXT_STAGE.md`已读取；最终完整测试123项通过、无跳过。新pilot为两个专家、3seed、64/24/16/24分组、100epoch/400step上限，15个可训练配置实际160–400步。原始与响应融合验证EER均0，未满足预设H1增益门，按协议未进入H2本体比较。24条新测试样本只有3个speaker组/1个spoof组，天花板结果不能证明研究效果；这些确认样本也已消费。

完整分层状态见 [STATUS_NEXT.md](STATUS_NEXT.md)。可分享的脱敏证据包为 `reports/codex-next-stage-20261110/audit_bundle.zip`：解压后在解压目录运行 `python3 audit_recompute.py`，标准库独立重算576条预测，不需要音频/模型。包内源码使用路径模板，真实提取需配置本地资产；私有历史tar不用于分享。当前CLI和依赖安装说明见 [CLI_CURRENT.md](docs/CLI_CURRENT.md)。

## 仍然 BLOCKED

当前工具会话未暴露GPU设备，未运行四卡验证。RDF/SHACL已通过本轮隔离依赖层补齐真实验证；网络依赖下载限制仍在。旧专家选模与部分fusion_val/calibration重叠已审计，新pilot隔离已知ID及speaker，但官方跨说话人源映射与任务HPO来源仍UNKNOWN；research_holdout阻塞。ASV2021官方全测和ASV5确认尚未执行。GitHub公开仓库：[745165806/ontoresponse](https://github.com/745165806/ontoresponse)。发布内容为脱敏源码与审计证据，音频、模型权重和缓存另行配置。正式协议仅为提案，不自动下载或全量训练。

方法、研究协议和验收要求分别见 [METHOD](docs/METHOD.md)、[EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md)、[ACCEPTANCE](docs/ACCEPTANCE.md)。保留的 `reference/` 为数值核，生产输入来自 `ontoresponse/` 的真实音频提取；tests中的合成fixture从不替代真实专家。
