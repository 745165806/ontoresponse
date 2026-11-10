# OntoResponse 实施计划 — 2026-11-09

本轮自主实施；运行环境优先 `${USER_HOME}/anaconda3/envs/py310/bin/python`。不下载数据、不启动全量训练，不更改上游文件或驱动。

1. P0：检测 CPU/GPU/内存/磁盘/依赖/codec；固定本项目代码内容 hash、上游 commit/源码 hash、checkpoint SHA256、训练/选模来源；准备按 speaker/source 连接组件划分的清单及暴露账本。未知来源写 null，严格研究模式 fail closed。
2. P1：加载真实 AASIST、SSL+AASIST、RawNet2 权重，label-free 文件/张量接口，固定 crop、eval 和 margin 方向；同音频同预处理与直接上游输出比对。缺失/不兼容记录 BLOCKED，不使用随机专家。
3. P2：确定性低通、共享噪声方向、真实 MP3 roundtrip；identity/lossless 控制；原子内容寻址缓存与稳定 source 分片，保存逐请求状态及 before/after 溯源。
4. P3：先实现 score LR/MLP、original MLP、TTA 统计、all-view MLP、response MLP；只在 fusion_train 拟合共享 scaler，用同缓存、源版本权重和相同 step/selection 预算。
5. P4/P5：接入现有经过测试的关系数值核，registry 编译实际拓扑与来源 RDF；typed/untyped/attention、边删除、错误 family 成员、no-response/no-delta/单专家/缺失探针消融，记录参数量和重训。
6. P6：独立校准/开发阈值、冻结模型与协议、逐请求预测、组 bootstrap 和多 seed 汇总；未知处理完整 DAG 暴露审计与外部来源审计 gate。未通过 gate 不读取最终 test 做选择。

每阶段在 STATUS.md 分列 implemented/unit_tested/real_checkpoint_tested/real_data_evaluated/GPU/blocked；记录实际命令、退出码、日志及限制。默认 smoke <=64 origins、<=20 optimizer steps/模型/seed；pilot <=1024 origins。合成测试仅作软件验证。

## 本轮落地结果

- P0—P5有界实施与P6集成外测已执行；详细命令/退出码/日志及层级见STATUS.md。
- 实际24条dev + 8条外部origin；74个unit/synthetic测试通过，3个真实checkpoint通过；78次融合run、3seed、每run≤20步。普通基线先完成，然后关系模型/消融；之后校准与输入/模型冻结，再执行外部3条件共24请求。
- 外部结果与不利固定阈值行为均保留；CI包含0，不作本体增益结论。此后没有根据外部结果改参数、阈值或模型选择。
- research_holdout、完整来源映射、GPU/四卡、RDF依赖/SHACL、ASV2021正式全测与ASV5确认均未冒充完成。后续应在资源与监督隔离满足后使用新目录、新冻结协议和未用于探索的新test；不覆盖已评价的freeze锁。

## 2026-11-10 下一阶段（原冻结目录只读保留）

按本轮用户六项任务推进：先导出/验证旧证据和源码快照；重新审计专家训练、早停与选权重；新增四通道逐关系归一化的relation_tied及复制等价/梯度测试；独立重算旧预测与阈值/角色/方向；补齐本体正负例、完整测试、可复现依赖与仓库外安装；最后只有审计/明确pilot预算通过才启动新pilot并保存收敛曲线。

输出固定到reports/next-stage-20261110与新experiment目录，绝不覆盖reports/implementation中的冻结模型/校准/阈值。8条既有外部origin及所有派生版本登记为consumed_exploration。当前CODEX_NEXT_STAGE.md缺失，已搜索项目及相邻工程并询问路径；在文件/预算和监督审计明确前不启动pilot，不标research_holdout。GPU只诊断执行环境，不创建设备节点、改驱动或结束进程。

## 2026-11-10 阶段收尾

已完成baseline完整导出/逐成员验证、两专家原始监督及选权重核查、RawNet2隔离、relation_tied参数复制等价/梯度测试、6冻结run独立指标重算、真实SHACL及负例、99项完整测试、依赖RECORD锁、仓库外wheel检查。原冻结349文件hash不变。
新pilot预算128 origins/6模型×3seed/最多100epochs与400steps，实际仅准备清单；质量门通过但CODEX_NEXT_STAGE.md缺失，pilot预检退出2。无新真实pilot训练/预测；不以准备好的代码和合成32步优化曲线冒充合规pilot已成立。

## CODEX_NEXT_STAGE.md 到位后的修订（2026-11-10）

已读取新增文档A—I。旧“只缺文件即可直接运行六模型pilot”的计划被替代：
1. 新建脱敏audit_bundle.zip，排除所有原始/派生音频、缓存波形与大checkpoint、凭据和绝对个人路径；统一ID假名、补齐独立重算字段/选择账本，验证每个ZIP成员/hash。既有私有tar保留但不得当成符合A节的分享包。
2. 生成supervision_matrix与role_overlap_report（通用SSL/任务fit/选模分别列；未知源不记零），查本地及官方2019/2021→VCTK/VCC映射，报告连接组件与类别/角色。仅在相应审计范围通过后运行明确标记的两专家pilot，绝不冒充research_holdout。
3. 补齐指标边界测试、AUROC/AUPRC、单调校准/饱和阈值审计；补训练scaler、crop coverage、批量/重复/标签隔离/codec多时长/采样权重诊断。
4. 固定新预算/磁盘与缓存上限、B0/B1/B2两候选；先做H1。只有预注册验证标准支持H1时才进入full_attention、untyped_union、relation_tied、typed及确认消融；H1不支持则停止关系复杂化并保留诊断。
5. 保存STATUS_NEXT、实际命令/退出码、收敛曲线/失败、逐条预测、formal_protocol.yaml（只供审阅，不执行full）。所有旧冻结及8条探索外测保持不变。

## 本轮实际收尾

已读取新增文档、补齐123项完整测试、真实两专家测量控制及16个codec控制、最新wheel仓库外检查、15行监督矩阵和完整未知来源披露。执行显式启用的128-origin integration pilot：6种普通方法×3seed，15个可训练配置160–400步，18条收敛记录；18个模型/校准/阈值全部冻结后，产生432条新测试预测。H1验证天花板未支持响应增益，按预注册规则停止H2，typed/tied/union/attention与确认消融没有新的真实训练结果。

严格来源门仍BLOCKED，未把integration质量门写成research_holdout通过。原8条及新24条测试origin进入消费账本。最终分享ZIP仅含脱敏文本源码/配置/测试/日志/576条预测，实际manifest逐成员核对和仓库外stdlib重算；旧私有tar不上传。GitHub离线提交邮箱745165806@qq.com，实际上传因网络/认证/仓库URL未就绪而BLOCKED。STATUS_NEXT逐阶段列明证据与限制。
