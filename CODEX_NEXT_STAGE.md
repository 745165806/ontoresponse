# Codex 下一阶段任务：OntoResponse v0.5 → 可审计的研究 pilot

## 总目标

继续现有v0.5项目，不重建项目，不回到OntoRisk，不改成局部定位。

整句输入经真实冻结专家产生embedding/spoof margin；已知探针产生处理响应；类型化关系参与直接BCE分类。下一阶段首先证明数据隔离、对照公平与评价可信，再检验本体增量。

请阅读项目现有AGENTS.md、STATUS.md、docs/METHOD.md、docs/EXPERIMENT_PROTOCOL.md，以及本任务。先更新PLANS.md。状态文件中报告已有74项指定测试、真实专家和小外测；必须核验实际证据，不把摘要当成重新运行。

### 不变的限制

- 不改变一条录音一个真假logit及0=bonafide、1=spoof。
- 不把质量/置信度/拒判作为新的主研究目标。
- 不覆盖已冻结实验、不修改旧外测分数、不按已有8条外测调整阈值/方向/校准。
- 不使用test拟合scaler、选探针、关系、模型、阈值或超参数。
- 不升级系统驱动、不结束别人的进程、不自动下载大型数据/权重。
- 不将缺权重或未知来源替换成随机专家或默认为clean。
- 不把skip、未收集到的测试、配置存在记为通过。
- 保留源版本、origin、speaker/source group的区别。

## A. 首先交付完整的审计证据，而不是再次宣称完成

实现一个可运行的最小审计导出功能（命令名按现有CLI组织，不为凑接口重写核心）。在新输出目录中收集：

1. 源码：ontoresponse/、reference/、scripts/、tests/、ontology/、configs/、docs/及顶层包配置。
2. 实际环境：依赖版本、Python、codec二进制及版本、CPU/GPU可见性；脱敏路径。
3. manifest和内容hash：覆盖源码、配置与审核所需结果。记录旧MANIFEST与新manifest的范围，校验文件存在和hash。
4. 实际日志：完整pytest日志与JUnit、check-experts、extract、selection、calibration、freeze、evaluation、report-recompute。
5. 小规模预测：每条录音/条件/方法/seed的raw_logit、p_calibrated、threshold、prediction、label、origin_id、source_group、speaker_group、status、calibrator_id和model_id。
6. 分组和选择账本：split表、专家fit/选模来源、normalizer fit role、超参数候选、选择理由、校准清单、冻结清单及test-consumption账本。
7. 不包含原始音频、大checkpoint、访问凭据、绝对个人路径；合法必要ID可一致伪名化，但必须保留配对与分组可重算性。

验收：第三方不访问音频或模型，也能从给出的逐条预测复算当前小实验；能够独立验证manifest的可得文件。导出清单应列出缺少的文件，不把manifest中只列了名称的文件当已经交付。

## B. 统一实际状态与依赖

1. 原TASK_FOR_CODEX中“仅reference.demo可用”是历史规格。明确标为historical，生成与实际--help一致的CLI说明，不删除原研究约束。
2. requirements.txt、requirements-reference.txt、pyproject.toml中base/test/ontology依赖范围统一说明；输出每个实际使用环境的lock。旧fairseq独立环境不得任意升级。
3. 当前环境无rdflib/pyshacl时报告缺失；有批准的wheelhouse或网络环境后安装项目依赖并执行完整测试，不用importorskip掩盖必需SHACL验收。
4. SHACL必须有真实完整证据图的正例与负例：before/after专家不一致、probe artifact不一致、来源断裂、score语义/crop不一致都应拒绝。
5. 构建wheel，在repo外目录安装并运行最小CLI及资源读取。验证configs、ontology词表的定位；确有问题再补package data，不能未测先称修复。
6. 无GPU时先审计会话/容器/设备节点及权限，不改驱动。GPU可见后分别执行单卡真实输入与互斥分片合并验证，记录容差；不要求CPU和GPU逐bit相同。

## C. 解除监督来源阻塞

生成supervision_matrix.csv/json，按专家区分：
- 通用SSL无标签预训练；
- 鉴伪监督fit；
- 早停、选checkpoint、超参数选择；
- 已知/未知数据版本与ID/source/speaker集合。

生成role_overlap_report，分别比较fusion_train、fusion_val、calibration、confirmation_test与上述已知集合。不能只比较文件名。

现有问题：报告称已知source_val与fusion_val/calibration各重合2条；RawNet2来源未知。按以下顺序处理：
1. 定位真实重合ID及其全部派生源组。
2. 排除这些源组后，检查是否还有足够的独立组及两类样本。
3. 若足够，使用新划分、新缓存角色和新实验目录；若不足，提议重训有关专家或使用有可核来源的模型。
4. RawNet2若无法认证，可在明确标记的两专家pilot中暂时移除，对所有基线使用同样两专家；不能隐含保留为严格模式第三专家。
5. 通用SSL预训练与任务监督分开披露。无法认证的无标签预训练可以在明确open-pretraining条件下报告；不得将其写成所有来源完全未见。
6. research_holdout只有任务监督与选模隔离、来源审计和角色分组均通过时才允许标为PASS。来源未知是UNKNOWN/BLOCKED，不是零重叠。

利用官方2019/2021到VCTK/VCC的录音映射。报告source/speaker连通组数量、大小、类别和角色；如果少数组跨所有角色，不允许为凑比例拆组。

已使用的8个外部origin及其全部环境/探针版本进入exploratory ledger。后续确认测试使用未用于开发的新锁定样本，按研究所声明的speaker/source独立性隔离。旧8条可作回归，不作新证据。

## D. 新增一个关键公平对照：relation_tied

核查reference/core.py实际仍与报告hash一致。当前typed逐关系求均值后求和；untyped把边合并后只求一次均值。因此两者同时改变关系专用参数与归一化。

新增relation_tied模式：
- 使用typed完全相同的四个有向通道、邻居、mask、逐关系degree归一化；
- 所有通道调用同一个W_rel，但每个通道先各自聚合；
- self_map、残差、LayerNorm、dropout、tokenizer、readout及质量输入与typed一致；
- 与原untyped_union作为不同模式保留，不覆盖旧结果。

增加以下测试：
1. 把typed的所有关系矩阵设成同一W，复制其余参数，在eval/dropout=0时输出等于relation_tied。
2. 不同mask、零邻居、缺失原始观测和全无证据下保持语义。
3. relation_tied保存/恢复一致，参数共享是真共享，不是初值相同的独立矩阵。
4. 原untyped_union仍能复现，报告它与tied不同是设计差异。
5. 对照不同模式的可训练参数与实际聚合方式。

解释规则：typed−tied检验类型专用参数；tied−union检验关系平衡归一化；typed−wrong-family检验对应关系；typed−same-information attention/MLP检验总体结构先验。不要把typed−union单独称为纯语义关系效应。

## E. 修订数值诊断，但不凭小外测重设计模型

1. 记录每专家训练scaler的std分布、1e-5下限命中比例、极端z值、原始/响应norm分布。只用fusion_train决定任何缩放改动。
2. 原始与probe共用同一变换；不同专家适配，不直接比较未对齐坐标。
3. 明确origin的环境版本总权重。在小批量中避免自归一化weights与采样机制破坏预定总体等权；用手算多源、多版本、缺失版本数据测试或采用每origin固定抽一个版本的采样器。
4. 单crop报告每专家实际观察时长与coverage。确定性多crop只作为预注册对照，不引入帧/局部标签。
5. 增加batch大小不变性、重复推断、设备迁移后eval状态、同输入不同行为标签不影响特征、噪声seed与标签/文件名标记独立的检查。
6. 编码延迟与裁尾规则在多时长和控制信号上检查；不能只依据一条音频或label选择最优对齐。
7. 原始与变换视图缓存索引必须可验证；同一probe供全部专家使用。失败和有效零差分严格区分。

## F. 独立重算与阈值审计

编写不调用训练/选择代码的审计脚本，读取已存预测和冻结参数：
- 按条件计算EER/AUROC/AUPRC/NLL/Brier、原始计数、FPR/TPR、coverage。
- 分开raw logits与calibrated probabilities，检查阈值所在空间。
- 主方向固定为高分更spoof。对官方工具明确转换；不在test上猜方向。
- 记录Platt参数a/b及选择来源；正单调校准不应改变无并列时的ROC/EER；若a<=0必须解释开发依据，不隐式翻转test。
- EER阈值只作曲线统计，禁止代替冻结部署阈值。
- 检查完美分隔、反向分隔、常数/并列分数、单类、空集合和NaN状态；记录与官方插值/并列约定的差异。
- 不把4条真实开发录音估计的0% FPR写成1%目标已达标。
- Bootstrap对两模型使用同一source/speaker重采样，所有环境版本一起抽取；三个seed不增加独立录音数。
- 小样本退化重采样透明报告；95%区间包含0时不宣称优越性，也不宣称方法已无效。

## G. 合规pilot：先回答H1，再回答H2

原74测试/20step属于smoke。不要照抄20step当正式预算，也不要直接全量。

在C阶段通过后，生成显式pilot配置：
- 总origin预算不超过现有AGENTS允许的1024，所有环境/探针属于同一origin；
- 按可用独立组设置fusion_train/val/calibration及内部测试，不强行逐条随机凑比例；
- 可在固定评估角色上做嵌套训练子集学习曲线，子集按source/speaker选取；
- 初始hidden_dim从32/64起，3个seed，固定候选数；
- 明确max_epochs、max_steps、early_stop_patience、磁盘与缓存预算，所有模型共享选择规则并有充分收敛机会；
- 默认新阶段只自动执行审计/测试/smoke。pilot和full都需显式预算配置启用，不悄悄突破原限制。

主模型清单先固定并减少：
B0 best_single_expert
B1 original_embedding_score_mlp
B2 all_views或response_mlp（由fusion_val预选，保存两个候选的结果）
B3 full_attention（未实现完整Set Transformer则准确命名）
B4 untyped_union
B5 relation_tied
M typed_response

结构确认清单：wrong-family actual members、三类边删除。信息确认清单：无embedding响应、无score响应、恒等复制视图/同规模无有效变化控制。全部模型获同样family/参数/expert/mask/quality信息及相同缓存。

0/3/6探针做成本性能报告，明确3探针时每家族单例没有same-family边，不能把该比较误作纯成本或纯某关系效应。

H1失败时不要继续增加复杂关系；先诊断证据和专家互补性。H1成立而H2失败时诚实保留响应融合结果，不把本体当作已经证明的贡献。

## H. 正式研究计划与冻结门

pilot后输出供用户审阅的formal_protocol.yaml，不启动未授权full作业：
- 精确训练/选模/校准/阈值/test角色及source哈希；
- 环境T与探针P的完整DAG暴露账本；已在T训练或P探针出现的算子/组合不能伪装完全未知；
- 主对比typed vs fusion_val选出的最强普通响应融合，主指标仍为未知环境macro EER；
- 同时报告原始条件、实际冻结阈值FPR/TPR、NLL/Brier、单条件结果、失败和成本；
- 预先列出所有确认结构消融，使用同一个新的冻结test；
- 独立源/说话人样本量与组数，针对目标EER差异和低误报目标说明精度；
- ASV2021官方全测试和已知重合剔除子集分开报告；未完成来源审计不能声称完全独立；
- 固定随机种子、超参数候选、收敛规则、bootstrap方案与允许的重跑条件。

保持主任务不变：直接整句真假判别。阈值、校准与可靠性只是完整评价，不替代判别目标。

## I. 每阶段的真实交付

在新目录保存：
- STATUS_NEXT.md：implemented / locally_tested / real_checkpoint / real_data / GPU / blocked分别列。
- audit_bundle.zip及hash清单；
- supervision_matrix、role_overlap_report、dataset_summary；
- relation_tied测试和normalization对照报告；
- metrics_recomputed、threshold_audit；
- pilot配置、学习曲线、逐条预测和完整失败列表；
- 正式研究方案及尚缺资源。

不要因不利结果更改主指标或删除方法。没有实际运行证据就写未运行。第一阶段的成功是可独立审计和合规pilot成立，不是强行得到typed第一名。
