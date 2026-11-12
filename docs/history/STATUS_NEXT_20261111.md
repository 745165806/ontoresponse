# 最新：完整GPU专家benchmark与数据建议 — 2026-11-11

三现用专家在完整ASV2019 LA eval（71237条，7355 bonafide/63882 spoof）均完成，失败0：作者原生score EER分别AASIST **2.121052%**、SSL+AASIST **0.637284%**、RawNet2 **5.084675%**。同一流程的作者AASIST发布参考权重为 **0.829511%**，重现官方0.83%。当前本地AASIST未复现；SSL论文主要结果对应2021 LA/DF，RawNet2配置/来源不同，不能将现用三权重一概称为论文复现。

本轮经授权使用沙箱外GPU0/1/3，真实GPU/上游/批量控制通过；四份完整输出均有固定身份与音频hash。完整unit/synthetic171通过、0skip。官方来源archive已经成功取得并按真实格式核验：2019全部ID与DF611829个ID全覆盖；此前网络/缺archive阻塞已解决。新的审计显示2019 train/dev同源1043条、保守连接组件不足；Codecfake官方train/dev也有speaker重叠及不同采样率。

**新的融合全量训练仍未启动**。近期建议Codecfake中文来源按speaker/source重分组、统一16kHz后使用全量；ASV5本地仅协议，作为后续现代路线。监督来源/任务HPO/通用SSL与合格新角色仍需分别认证。最新证据见 RESULTS.md（本地证据：`reports/expert-benchmark-20261111/RESULTS.md`），数据方案见 FUSION_DATA_RECOMMENDATION_20261111.md（本地证据：`docs/FUSION_DATA_RECOMMENDATION_20261111.md`）。以下全部为历史阶段记录，原日志与冻结产物保留。

---

# 历史结论：来源映射阻塞，必要准备已完成 — 2026-11-10

**新真实H1/H2不能启动；完整目标尚未完成。** 同一来源阻塞已连续三轮核验：本地没有官方录音级映射，curl无代理路径DNS失败（退出6），web页面工具不能交付gzip archive字节。新实验配置保持disabled，未写认证base/test角色，未读取新确认结果。

本轮补齐完整来源连通角色分配：共同来源宇宙、消费/专家曝光整组件排除、官方2019/2021协议ID全覆盖检查、origin/class/group配额、最大化有效独立组、组内轮转、求解/超时状态和约束复核。base→历史T→实际波形重复审计→data-audit→freeze闭环已验证。官方archive格式未知，未猜造解析器或声称映射已到位。

- **166 passed、0 skipped**；新wheel与当前源码hash相同，仓库外旧24条fusion_val缓存前向通过。
- 真实2019 eval的71237条无role候选metadata已导出；speaker独立的乐观上界可行，拟议val每类组21/16、test18/12。它不等于真实source独立，未写新角色manifest。
- 旧490冻结文件全部hash不变；GPU仍不可用，没有全量专家重训/大下载/新真实H1训练。元数据不能证明EER功效，拟议32个test bonafide即使IID零错误95%上界仍约8.94%。

完整交付：SOURCE_SPLIT_REPORT.md（本地证据：`reports/connected-split-20261110/SOURCE_SPLIT_REPORT.md`）。阻塞审计：blocked-audit.json（本地证据：`reports/connected-split-20261110/blocked-audit.json`）。继续需要官方来源archive及其可核对格式/完整映射，或恢复可用的下载网络；然后真实解析、覆盖认证、组件划分、精度审计、注册提取及充分收敛H1，H2仍条件执行。

---

# 当前阶段：多环境开发实验接口已验证 — 2026-11-10

**多T训练、条件宏EER、收敛/确认集守卫与预注册锁已实现；真实新H1/H2尚未执行。** 最新证据在 VALIDATION.md（本地证据：`reports/development-runner-20261110/VALIDATION.md`）。此前“多T/多条件runner待接入”的描述属于历史状态，已由本阶段实现替代。

- 新增历史T版本生成：每origin固定hash选训练处理，完整波形对齐、统一WAV容器，失败请求保留；诊断P仍由专家提取管线单独施加。
- 每epoch每origin一个有效T，三seed下同模型/候选共享数据日程；六类基线同候选网格。家族选择用预定AAC/Opus条件macro EER，identity单列。记录源数/梯度/lr/抽样hash，cap-only不算收敛。
- v2冻结检查完整source/speaker连通宇宙和已消费/专家曝光组件；固定代码、配置、checkpoint、音频、probe与本体配置。H1无区分能力或H1/H2未收敛时，停止并且不加载新确认张量。
- 完整unit/synthetic **160 passed，0 skipped**，含六类基线×三seed的合成多版本训练。新wheel在仓库外加载24条旧fusion_val缓存前向通过；没有新真实H1训练或新test读取，GPU仍不可用。
- single/cover两份512-origin新配置保持disabled。就绪检查退出2：缺官方archive/完整解析映射/认证角色清单。严格研究的task HPO/通用SSL曝光问题继续单列；未把integration写成research_holdout。

新运行接口与冻结顺序见 DEVELOPMENT_V2.md（本地证据：`docs/DEVELOPMENT_V2.md`）。原490冻结文件hash无变。本阶段完成工程缺项；下一步仍需完整官方映射解析、连接组件角色分配/精度审计、新缓存及充分收敛H1，不能据软件验证宣布目标完成。

---

# 当前阶段：区分能力诊断与整句覆盖 — 2026-11-10

附件要求已执行诊断、协议修订与真实测量。**旧pilot应解释为 INSUFFICIENT_DISCRIMINATION，H1未决；H2尚未进入。** 原冻结gate与全部490文件hash保持，新增解释单独存档。新三seed H1/H2还没有执行，不能宣称研究目标已完成。

- 验证24条，bonafide/spoof各12条，但speaker组为3/1；speaker内多数标签占比83.3%。基线验证EER三seed均0，无法达到预设0.01改善；按组bootstrap缺类概率8/27。校准与测试亦仅一个spoof speaker。全交叉表与训练曲线索引见诊断报告。
- 新增确定性全句cover crop（尾窗右对齐、原始/探针相同坐标、分批推断、mean embedding/logit），保留原single crop。23/128条原覆盖不完整，cover共155 crop。每请求记录坐标、覆盖、耗时、CPU峰值RSS和GPU分配峰值字段。
- py310完整unit/synthetic：132 passed、0 skipped；两个真实冻结专家×两策略×两个已消费训练音频，共8项real-checkpoint/real-data CPU控制通过。GPU仍不可用：nvidia-smi退出9，torch CUDA不可用。真实GPU级别 BLOCKED。
- `formal_protocol.yaml`新增512-origin开发提案、source/speaker连通分组/消费排除、56条T/P有序闭包、六类H1基线×3seed×4候选×两crop策略与收敛门、后续H2公平对照。全128条pilot、旧8条外测及其关联组件不得称新确认。
- 本地2019 train/dev/eval及2021 DF metadata已盘点；DF有611829条、VCC2018 spoof只有4个speaker，不能硬拆足够独立组。官方录音映射curl退出7，task HPO仍UNKNOWN。未创建假认证新split；新协议仍disabled。当前pilot runner的多T抽样/多条件宏EER选择尚需实现，不能直接运行该新协议。

主交付：DIAGNOSIS.md（本地证据：`reports/discrimination-20261110/DIAGNOSIS.md`）。命令、失败及修正、真实输出、资源/成本与旧文件完整性均在同目录。没有新确认结果、全量重训或大型下载；下一阶段真实科学实验 **BLOCKED（来源与协议接口未齐备）**，代码/测量验证可继续。

---

# OntoResponse v0.5 下一阶段实际交付 — 2026-11-10

已读取并按 `CODEX_NEXT_STAGE.md` A—I 执行，延续现有整句分类工程。标签方向固定为 0=bonafide、1=spoof。没有恢复风险矩阵/QP，没有局部标签，没有启动 full 或专家重训。

**软件与测量审计通过；严格研究来源审计仍 BLOCKED。** 已完成显式配置下的两专家 integration pilot，不能称为 research_holdout。官方跨说话人录音来源映射和任务超参数选择来源未认证，UNKNOWN 不记为零重叠。

| 阶段 | implemented | unit-tested / locally_tested | real-checkpoint-tested | real-data-evaluated | GPU-tested | blocked |
|---|---|---|---|---|---|---|
| A 独立证据导出 | `audit-export`、可分享脱敏 ZIP、逐成员 SHA256/大小/存在性、stdlib 重算；不含音频/权重/缓存/凭据 | 导出/隐私/独立重算测试通过；实际交付验证见 `public-export-verification.json` 与 `public-recompute.command.json` | 保存实际上游一致性日志与权重 hash；不声称交付模型本身 | 旧144条和新432条预测，共576条；保留角色/分组/阈值空间 | 未测 | 私有历史 tar 不属于可分享交付；大型外部资产明确排除 |
| B 依赖/安装/本体 | 统一依赖范围、实际环境/依赖 RECORD 锁、wheel 资源定位、当前 CLI 文档 | **123 passed，0 skipped，0 failures/errors**；真实 SHACL 正例及7种语义/3种拓扑负例；仓库外安装通过 | 已安装包对8条旧真实缓存及原融合 checkpoint 前向通过 | 24条完整证据图、2160条数值边核对通过 | **BLOCKED** | PID1 为沙箱；内核可见4卡，但无可用GPU字符设备，/dev只读/nodev；未改驱动或结束任务 |
| C 来源与角色 | 15行专家监督矩阵、精确ID与speaker角色交集、source UNKNOWN、旧8条探索账本 | 分组、隔离、冻结和预算守卫通过 | AASIST/SSL+AASIST真实上游 hidden/logit、一致crop、identity/lossless通过；RawNet2排除 | 新128条音频实际hash/跨角色波形去重；已知上游fit/选模ID和speaker交集为0 | 未测 | 官方2019/2021→VCTK/VCC映射未取得；task HPO/通用SSL暴露 UNKNOWN；research_holdout=false |
| D relation_tied | typed同四通道/邻居/mask/逐关系均值；仅关系矩阵真共享；保留union | 参数复制、double前向、共享梯度、缺失/空图、save/load和同seed初值通过 | 不改变冻结专家接口 | **本轮真实训练未运行**，H1门停止H2；不把合成测试当效果 | 未测 | 不支持H1时不升级关系复杂度 |
| E 数值与测量 | train-only scaler/crop诊断、origin版本抽样、缓存上限、多时长codec与batch控制 | 手算等权、mask、缓存、边界测试通过 | 两专家batch/separate容差1e-4、重复精确、ID/标签隔离通过；仅CPU迁移 | 16项codec控制全部通过；训练scaler下限命中均0；实际coverage保存 | 未测 | 真实GPU迁移和分片合并数值验证未运行 |
| F 独立评价 | 不调用训练的EER/AUROC/AUPRC/NLL/Brier/ECE/固定点计数，方向与空间审计 | 完美/反向/常数/并列/单类/空/NaN及单调性测试通过 | 冻结权重hash逐一核对 | 旧6run×24请求指标重算一致；新18run×24请求重算；官方并列约定差异明报 | 未测 | 旧概率阈值1.0的float32饱和保留，无有限logit等价；未调整旧阈值 |
| G 有界pilot | 新目录128 origins，64/24/16/24，两个专家，width64，3seeds，最多100epochs/400steps；缓存2GiB/输出1GiB | 显式启用/预算/冻结/收敛/阈值测试通过 | 128条完整提取，896共享音频视图、1792 expert-view | 18个H1配置：3个无训练单专家、15个可训练配置160–400步；432测试预测、失败0；18条收敛记录和PNG | 未测 | H1验证EER两者均0，改善seed=0；按预注册标准不支持H1，**H2与关系消融未运行** |
| H 正式方案 | `formal_protocol.yaml`，角色/暴露DAG/公平对照/统计/重跑/样本量门 | YAML可读；不等于研究审计通过 | 不适用 | 未执行full、ASV2021正式全测或ASV5确认 | 未测 | 方案仅供审阅；来源门和新独立组仍待补齐 |
| I 保存/发布 | 原冻结目录保留；可分享ZIP与离线Git bundle分离 | 旧349文件完整性复核、ZIP manifest真实成员验证；Git bundle实际verify | 原/新冻结权重未修改 | 旧8条及新24条确认origin消费账本保存 | 未测 | GitHub未上传：未连接认证、HTTPS/DNS不可用；邮箱不是仓库URL |

## 运行依据及结果边界

主要证据在 `reports/codex-next-stage-20261110/`，每个 `*.command.json` 记录真实argv、cwd、开始时间、耗时和退出码，`*.log` 为实际输出。旧准备阶段证据保存在 `reports/next-stage-20261110/`，没有被新摘要替代。

- `pilot-execute.command.json`：退出0，约770秒，CPU每进程4线程；`reports/pilot-20261110/completion.json` 记录18个配置/18个评价。没有随机专家或随机分数。
- 模型和超参数候选、H1判据在配置中预先固定。先完成B0/B1/B2，验证集选择六探针 `all_views_mlp`；原始融合和所选响应融合平均验证EER均0，改善seed=0，不支持预设1个百分点/至少2seed改善的门。没有使用test触发停止或选择。
- 新测试集仅24条、3个speaker组，其中只有1个spoof speaker组。18个配置的EER均0、固定FPR=0/TPR=1，是明显天花板，不能证明泛化、本体收益或可靠的1%低误报。配对bootstrap每seed有效1412/2000、退化588；零宽差值区间不是总体等价或零错误证明。
- 所有18个校准/阈值/模型锁完成后才生成新test预测；见 `freeze-order-audit.json`。阈值在float64 calibrated-logit空间，新calibration独立拟合正斜率Platt，新fusion_val选工作点。旧8条没有重新校准、调阈值或选模型。
- `checkpoint-metadata-audit.json` 公开一个元数据单位勘误：旧pilot序列化的 `best_step` 实际是epoch；真实优化步数从原收敛日志重建。原冻结文件不改；未来序列化已修复并通过测试，执行时源码另存 `execution-source/`，不把修复后源码冒充当时版本。
- 0/3/6探针普通融合成本报告已保存；共用完整缓存下的冷执行视图数不是实测GPU加速。3探针每family单例，same-family边消失；本轮没有typed成本/效果曲线。

## 来源结论

AASIST和SSL+AASIST的原fit各25380条、source_val各5654条，history均80epoch，按source_val最小EER/最早epoch选66/7；真实文件hash对应。没有patience早停证据，保留KeyboardInterrupt历史。旧fusion_val和calibration各与source_val重合2条；新pilot排除所有已知上游fit/选模speaker，而非仅删4条。RawNet2只有state_dict，无法认证其历史，已从全部新基线同时移除。

官方录音级源映射未下载成功，本地未发现；完整跨speaker source连通关系未知，不能认定新角色严格来源独立。通用SSL无标签预训练单列open-pretraining；任务HPO来源未知仍是严格研究门阻塞。当前结论仅为已知任务ID/speaker隔离下的工程pilot。

## 交付与下一步

可分享入口：`reports/codex-next-stage-20261110/audit_bundle.zip`。解压后在解压目录执行 `python3 audit_recompute.py`，仅用标准库即可重算576条逐条预测；该镜像使用路径模板，重新提取时需配置本地数据/专家路径。原始音频、探针音频、embedding缓存和模型均不分享。

GitHub离线交付：`deliverables/ontoresponse-github-20261110.bundle`，commit邮箱使用用户提供的 `745165806@qq.com`，认证/实际远端未确认，**没有声称已上传**。当前工程只读空.git由独立脱敏发布副本处理，未重建研究工程。

接下来应取得官方源映射和任务选择来源，设计有足够独立源组、难度适当的新确认集，再审阅正式预算。GPU需要在有设备访问的执行环境重测；不通过修改系统驱动解决当前沙箱限制。本轮已消费的旧8条和新24条及其全部派生版本不能充当新的未见确认集。
