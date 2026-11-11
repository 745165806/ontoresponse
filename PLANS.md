# OntoResponse 实施计划 — 2026-11-09

## 2026-11-11 GitHub文档同步

用户明确要求将相关文档更新到现有公开仓库745165806/ontoresponse。基于远端最新main在独立工作区准备：更新README、STATUS/STATUS_NEXT/PLANS，发布完整专家评测报告、融合数据方案及小型汇总/控制/测试证据。沿用已建立的脱敏发布策略，扫描新文本的私人路径/凭据，不上传音频、模型、数组缓存、来源archive或逐条分数；明确本次文档提交与研究工作区执行版本的区别。更新AUDIT_MANIFEST的真实成员大小/hash，核对本地链接及旧576条标准库重算，再做普通fast-forward push和远端公开读取验证，不强推覆盖远端。

## 2026-11-11 三专家完整 LA eval 验证与融合数据方案

用户本轮明确授权三个独立专家在 ASVspoof2019 LA eval 全集评测，并允许后续全量训练/测试。先完成专家质量门，不用 eval 选 checkpoint、方向、裁剪或训练参数；旧冻结产物保持不变。

1. 核验现用三个 checkpoint、作者原生 score/crop、真实协议全集及论文对应的数据库/架构。AASIST 作者本地发布权重作为事先声明的独立参考，不能用 eval 从历史 epoch 中选优。
2. 新建独立 full-evaluation 预算与断点日志；专家输入只有波形张量。GPU 仅使用当前空闲 0/1/3，先以 <=64 条检查 GPU/batch/upstream 一致性，再跑 71237 条完整 eval，无探针、无训练。
3. 每请求保存 status、原生 bonafide score、spoof margin、音频 hash；失败不丢弃。完整覆盖后用固定作者 compute_eer 及独立 grouped ROC 重算，记录权重/代码/协议 hash、时间、资源和退出码。
4. 比较本地实测与适用论文结果，明确 SSL 原论文主评测为 2021 LA/DF、RawNet2 具体变体及来源限制。未达标不能宣称已经达到论文效果。
5. 形成可执行的 fusion_train/val/calibration/test 数据方案：已参与上游选模的 dev 不能冒充严格独立融合集；2019 eval 本轮作为专家 benchmark 后保留用途账本，不据结果反复调参；2021DF 来源重叠与新独立外测分开。若条件不齐，不强行运行带泄漏的融合全量训练。
6. 分层更新 STATUS.md 和独立 reports/expert-benchmark-20261111，保留实际命令、单元与真实 GPU 测试差别及剩余限制。

实际结果：四个预声明权重各完整71237条，0失败；GPU0/1/3、无训练。主EER为本地AASIST2.121052%、SSL+AASIST0.637284%、RawNet2 5.084675%，作者AASIST参考0.829511%。171 unit/synthetic通过、0skip；真实GPU控制及逐条重算另列。官方映射16MiB已取得并全覆盖2019/2021DF，但train/dev存在1043同声学来源；Codecfake现有官方划分speaker全重叠、采样率不统一。已交付中文source/speaker重分组的full候选方案，不在不合格数据或未认证checkpoint上自动启动融合full训练。完整证据和限制见新报告。

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

## 2026-11-10 区分能力修订（附件目标，独立新目录）

1. 只读复核旧128-origin pilot及8条外测：角色×speaker/source/attack/label，训练曲线、逐条预测、时长/crop覆盖；记录全部已消费origin，不改变旧冻结文件。
2. 增加验证集专用 INSUFFICIENT_DISCRIMINATION 阶段门，分离来源组不足、bootstrap退化、baseline下界与有能力但未支持H1；H1/H2/H3定义和主指标保持。
3. 盘点本地2019/2021 DF metadata和来源映射，建立source/speaker连通分组接口，未知映射fail closed；形成新开发/确认协议及完整T/P闭包账本。不能将旧128条或旧8条重新当新数据。
4. 保留single-crop，新增确定性全句覆盖crop坐标、分批推断和mean hidden/logit；坐标按原输入生成并用于全部对齐视图，记录覆盖和实测成本。
5. 在新来源门通过前只做真实checkpoint测量验证、单元测试和无预测数据审计；新H1三seed先于H2，收敛不足不作为正式效果。禁止全量训练/大型下载。
6. 更新formal_protocol.yaml、STATUS_NEXT.md、STATUS.md及独立reports/discrimination-20261110，记录命令、退出码、资源与明确可运行/阻塞结论。

### 区分能力阶段实际结果

- 已输出role-distributions、18条training-diagnosis、128条coverage、本地2019/DF inventory、全旧数据consumed ledger及H1-reassessment；旧验证baseline下界/每类组数/29.63%退化均说明H1未决，不能判失败。
- 已实现cover全句crop、分批expert聚合、逐请求计时与内存、source/speaker传递连通组和验证专用stage gate。完整unit为132通过，real CPU checkpoint控制8通过；原490冻结文件无变化。
- 已形成formal_protocol.yaml新开发512-origin提案、56条T/P闭包、H1/H2/H3顺序/公平对照/收敛/三seed/计算与存储成本；没有认证源映射时不生成假split或启动新科学训练。
- 下一可做工程：将多T origin_uniform_epoch与按预指定条件macro EER接入pilot runner，增加新协议freeze/preflight；在映射到位后生成完整连通角色、每类组数并执行新H1。当前无源映射/task HPO证据，GPU不可用；无全量训练授权。新真实H1/H2尚未完成，目标保持active。

## 2026-11-10 多环境训练接口（继续附件目标）

上一轮已形成可复核诊断并完成全句测量，属于progress。当前先补齐可独立于缺失来源映射推进的工程：
1. 多T训练每epoch按origin选一个有效版本，使用独立seed的固定顺序并保存抽样hash；各模型/候选同数据日程，标签不参与抽样；记录gradient/lr/实际源数。
2. 显式条件集合的验证宏EER，逐条件类别/请求/失败检查；禁止test用于选指标；best epoch仍按预注册val BCE，家族比较按预定宏EER。
3. H1门加入充分收敛检查；不支持/无区分能力/未收敛时不消费新确认集；新增v2 preregistration锁，冻结配置/代码/分组与源映射证据。
4. 用隔离unit/synthetic缓存执行多T/多条件保存恢复、三seed普通基线与泄漏负例；不将这些作为真实H1结果。原490冻结文件不改，真实新数据实验仍需认证源映射。

### 多环境接口实际结果

已完成上述1–4的软件部分：历史T生成、origin等权抽样、condition macro EER、网格/三seed、收敛和确认消费守卫、完整source宇宙/字节来源预注册。最终160 unit/synthetic通过，新wheel仓库外24条旧fusion_val缓存前向通过；新真实数据科学实验未运行。single/cover配置disabled，source readiness明确缺官方归一化映射和认证角色，旧490文件无变。

下一可做工程为完整source连通角色分配与官方map解析接口；取得官方archive/schema后才能认证真实新split。GPU成本仍不能实测；新H1/H2与source独立性未证实，目标保持active。

## 2026-11-10 完整来源连通划分（继续附件目标）

上一轮为progress：160测试和新版本生成/训练/冻结接口已落地。本轮推进最后一个不依赖新专家推断的准备项：
1. 再确认本地官方来源映射是否到位，保留真实网络/设备检查证据；未知不写complete。
2. 实现完整来源宇宙的排除与组件级角色分配，约束每类独立组下限及严格源录音配额；选择不读取任何专家分数。不拆组件、不逐文件随机fallback。
3. 给出可核对的结构可行性/组Bootstrap退化报告；不足时保留原因，不为了可运行调整预设门限。
4. 连接base ledger → T版本 → 来源审计 → freeze路径，避免只有模块却不能形成data-audit。合成验证单列，不声称新真实H1。
5. 官方archive/schema未知时不猜解析器、不伪造映射，保持新实验disabled；更新状态及完整缺项。

### 来源连通阶段结论

已实现完整组件级角色分配、官方协议coverage约束、组内轮转与metadata精度报告、base→T→data-audit→freeze闭环；166单元/合成测试通过，新wheel仓库外旧验证缓存复核通过。真实metadata的speaker-only乐观上界不作为认证split。

来源archive仍缺，curl DNS失败/web工具无法交付gzip字节；同一外部阻塞连续三轮。当前无需映射即可推进的必要准备均已落地；不能凭未知schema猜官方解析器，也不能启动无认证来源的新H1。完整目标未达成，本轮达到blocked审计条件，等待真实官方映射文件或可用下载路径，不转为complete或paused。
