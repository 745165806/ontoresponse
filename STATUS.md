# OntoResponse v0.5 当前状态 — 2026-11-10

**当前以 [STATUS_NEXT.md](STATUS_NEXT.md) 为本轮完整状态。** `CODEX_NEXT_STAGE.md` 已到位并读取，软件/测量审计已通过；严格来源审计仍 BLOCKED，不能标为research_holdout。以下保留准备阶段和上一轮记录，时间先后明确区分。

- implemented：完整脱敏源码/配置/测试/真实日志/576条预测导出，四通道relation_tied真共享及等价验证，监督矩阵/角色交集、独立指标、正式方案草案均已实现。
- unit-tested：最终完整 **123 passed，0 skipped**；SHACL完整正例及语义/拓扑负例通过；最新wheel在仓库外安装并读取资源/真实缓存前向通过。实际命令、退出码、日志见 `reports/codex-next-stage-20261110/`。
- real-checkpoint-tested：新阶段AASIST/SSL+AASIST真实上游一致性、batch/重复/标签隔离控制通过；RawNet2来源无法认证，从新实验全部移除。
- real-data-evaluated：显式128-origin新pilot已完成，64train/24val/16calibration/24test、3seed；18配置中15个可训练配置160–400步，保存完整收敛曲线。先比较原始与普通响应融合，验证EER均0，未满足预注册H1增益门，因此**未运行H2、本体及关系消融真实对比**。432条新测试预测、失败0；旧144条只重算，旧冻结目录保留。
- GPU-tested：**未测试/BLOCKED**。四卡内核可见但当前沙箱无设备访问；未升级驱动或结束他人进程。
- blocked：完整官方录音级源映射、任务HPO来源和严格research_holdout；外部正式确认/full未运行。新test只有3个speaker组/1个spoof组且结果天花板，不能作泛化证明。GitHub仍未上传，认证/远端和网络不可用；邮箱用于离线提交，不能作为仓库URL。

可分享包为 `reports/codex-next-stage-20261110/audit_bundle.zip`，不含音频/权重/缓存/凭据；实际逐成员hash验证与仓库外标准库重算另存日志。旧私有tar含敏感资产，只保留本地历史，不是分享包。原8条外测及新24条测试origin全部登记为已消费。

---

## 2026-11-10 文档到位前的准备阶段历史（非当前状态）

# OntoResponse v0.5 下一阶段准备状态 — 2026-11-10

继续现有项目；没有更改整句任务、恢复风险矩阵/QP或重建工程。本轮软件/测量审计门已通过；**新的真实 pilot 尚未启动**：用户指定的 `CODEX_NEXT_STAGE.md` 在项目和相邻工程搜索中未找到，所需指令无法读取，已询问路径。`reports/pilot-20261110/preflight.json` 的唯一阻塞项是该文档缺失，不以等待时间或自行编写同名文档解除门槛。

## 本轮分阶段结果

| 阶段 | implemented | unit-tested | real-checkpoint-tested | real-data-evaluated | GPU-tested | blocked |
|---|---|---|---|---|---|---|
| 1 完整证据导出 | baseline tar含源码/配置/测试/实际日志/融合checkpoint/缓存/逐条预测；stdlib独立验证工具；新阶段另存tar | archive逐成员存在性/大小/SHA256自验 | bundle含原真实checkpoint比对证据及融合权重 | 6组冻结逐条预测实际存在；不以STATUS代替证据 | 否 | 原大型专家权重/语料在外部资产目录中锁hash，不冒称tar成员 |
| 2 监督来源审计 | 直接读取训练checkpoint、80epoch history、finalized与原始fit/source_val；复制原始证据；新配置隔离完整上游speaker；RawNet2隔离 | 分组/数据角色/预算/冻结保护测试通过 | AASIST/SSL在新清单的真实音频上重新检查通过，hidden/logits/identity/lossless均一致 | 128个新origin清单完成实际音频hash/跨角色波形去重；没有新训练/测试预测 | 否 | 无完整跨speaker源映射及通用SSL暴露认证，**research_holdout仍不通过** |
| 3 relation_tied | 四通道/原邻居/逐关系归一化/全部非关系结构保留，仅关系矩阵别名共享；旧union-untyped保留并明确区别 | 前向double等价、共享梯度=4梯度和、参数复制拒绝不等矩阵、mask/空图、save/load、同seed非关系初值一致 | 结构不改变冻结专家接口；本轮两专家已测 | 新relation_tied只有unit/synthetic验证，未冒称已完成真实训练 | 否 | 等真实pilot解锁 |
| 4 独立重算 | stdlib脚本不调用项目metrics/拟合；校验freeze/方向/角色/概率与阈值空间 | 旧metric角色/方向测试保留；独立实现用真实JSONL逐项核对 | 实际融合checkpoint hash与冻结锁一致 | 6run×24请求=144预测，EER/FPR/TPR/NLL/Brier与旧表差值为0，另给10-bin ECE | 否 | 普通基线阈值p=1及float32饱和警告保留，不重校准/调阈值 |
| 5 测试/本体/依赖/安装 | 修复cwd资源路径；wheel打包配置和SHACL；隔离依赖层锁与离线tar；执行环境定位 | **完整99 passed，0 skipped、0 failures/errors**，含SHACL正例与7种测量负例、3种拓扑负例 | 两专家真实上游比对通过 | 24条旧真实证据图SHACL通过，2160条边与registry数值拓扑一致；仓库外安装后8条真实缓存前向通过 | **未测** | kernel可见4张卡，但沙箱无GPU字符设备，/dev只读/nodev；未改驱动/设备/进程 |
| 6 新pilot | 新目录、128origin、6模型×3seed、100epoch/400steps上限与曲线记录、早停/预算耗尽区分、float64 calibrated-logit工作阈值、完整test锁先行已实现 | 显式预算/RawNet2隔离/缺文档门槛/32步合成优化曲线已测；非真实效果 | 专家检查已过，未启动新提取 | **仅完成分组清单与波形审计，无新真实pilot结果/收敛曲线** | 否 | CODEX_NEXT_STAGE.md缺失；新test仅1个spoof speaker组，限制明确公示 |

## 核心审计发现

- AASIST和SSL+AASIST的fit原始清单各25,380条、source_val各5,654条。history均80epoch；分别按`minimum_source_val_eer_then_earliest_epoch`选择epoch66/7，best.pt与所选epoch文件SHA256相同。另有KeyboardInterrupt的failure.json，保留原证据；不能把它推断为patience早停。
- source_val涉及5个dev speaker。旧fusion_val/calibration各有2条精确ID重叠；新pilot不只是删掉这4条，而是排除全部上游训练/选模speaker。RawNet2只有state_dict，没有这个权重的训练清单/早停/选模记录；当前训练脚本与文件名不能证明其历史，故新配置仅使用两个来源可审计专家。
- `consumed-exploration.json`登记了既有8个In-the-Wild origin、24个派生请求、7个speaker组及音频/波形hash。没有以该集重新拟合校准、阈值或选择新模型。原冻结目录**349个文件与baseline快照逐一核对，缺失0、变化0**。
- 普通基线工作阈值为保存概率的1.0，若干float32概率饱和为1；按原`p>=threshold`的FPR/TPR可独立重算，但不存在简单的有限raw-logit阈值等价解释。新pilot工作点预先改为float64 calibrated-logit空间，仅用新fusion_val选择，旧锁和旧预测均未修改。
- 新清单128 origins：64 train /24 val /16 calibration /24 test。上游精确ID、speaker重叠均0；新test与旧development speaker重叠0；波形跨角色重复0。官方dev有bonafide-only speaker，新test虽然有3个speaker组，却只有1个含spoof的组，因此不能据此主张广泛泛化。所有严格holdout标志仍为false。

## 实际运行证据

本轮解释器为`${USER_HOME}/anaconda3/envs/py310/bin/python`；RDF等附加包使用独立的`PYTHONPATH=${EXTERNAL_PATH}

- baseline导出实际校验**2463个文件**、492,703,163 bytes，SHA256=`3e03ea3f4e41fee4c657f87ab00761e47972140d50d1525b3b7c27cc6ed1aab7`。tar内BUNDLE_MANIFEST与成员集合严格对应；历史manifest缺失文件为0，历史hash差异单列，不伪改原manifest。
- `python -m pytest -q tests --junitxml=reports/next-stage-20261110/unit.xml`：退出0，**99 passed**，0 skipped。依赖的rdflib Dataset.identifier deprecation warnings实际记录，未把skip计通过。见`full-tests.log`及`full-tests.command.json`。
- `python -m scripts.audit_supervision`：退出0，`source-audit.log`/command JSON、`supervision-audit.json`与`upstream-evidence/`保存原始来源证据。
- `python -m ontoresponse check-experts --config configs/pilot-20261110.yaml --manifest reports/pilot-20261110/inference_manifest.jsonl --output reports/next-stage-20261110/real-checkpoints.json`：退出0，两专家真实检查passed，见`checkpoints.log`/command JSON。
- `python3 scripts/recompute_predictions.py --out reports/next-stage-20261110/independent-recomputation.json`：退出0，6run重算与旧表的5项指标差异为0，见`independent-metrics.log`/command JSON。
- 实际旧证据图执行SHACL并核对数值边：`real-evidence-ontology.json`为passed，24个recordings/2160边。
- `pip wheel . --no-deps --no-build-isolation`与仅在`${EXTERNAL_PATH} hash与源码一致。见`wheel-*.command.json`、`outside-install.json`与日志。这是复用既有torch/fairseq底层环境的安装验证，不声称已从空环境下载重建所有二进制依赖。
- `dependency-lock.json`逐文件检查14个本地附加包RECORD，错误0；固定实际版本、Requires-Python、层tar hash和当前有效环境版本。网络下载尝试仍失败（`dependencies-download.log`），实际使用的是已有本地纯Python包，不伪称本次网络下载成功。
- `python -m scripts.run_pilot --config configs/pilot-20261110.yaml --preflight-only`：**退出2，BLOCKED，仅缺下一阶段文档**。质量门`quality-gate.json`已PASS，预算及两专家来源/新分组清单审计通过。没有`experiment-start.json`、新真实提取/训练/测试预测，不能把准备状态说成pilot完成。
- `execution-environment.json`记录PID1为codex-linux-san、namespace、设备挂载、四卡内核信息及nvidia-smi失败；是执行环境设备不可访问，不据此断言物理GPU/驱动不存在。
- `compileall`退出0。各主要命令的时间、完整argv、真实退出码和日志路径均在`reports/next-stage-20261110/*.command.json`中。

所有新产物使用`reports/next-stage-20261110`、`reports/pilot-20261110`、`artifacts/pilot-20261110`及新deliverable；原冻结目录保留。新训练/校准接口也拒绝覆盖已经freeze的run。未下载大数据、重训专家、升级驱动、终止任务或启动full训练。

下一步需要实际读取用户指定的CODEX_NEXT_STAGE.md，确认其约束与现有预算兼容，记录其reviewed SHA256后重新预检。**不自行编造该文档或把缺失指令当作已审核**。具体入口和实现边界见`docs/NEXT_STAGE_IMPLEMENTATION.md`。

---

## 2026-11-09 历史记录（以下为上一轮状态，非本轮证据）

# OntoResponse 实际状态 — 2026-11-09

已实现并跑通真实音频/真实权重的有界 `integration_pretrained` 管线。输出为一条完整录音一项真假logit，0=bonafide，1=spoof，主loss为源版本等权BCE。没有使用OntoRisk误差矩阵/QP，也没有改成局部定位。**正式、独立的research_holdout仍然BLOCKED，不能把本轮smoke解释为研究成功。**

## 分阶段状态

| 阶段 | implemented | unit_tested | real_checkpoint_tested | real_data_evaluated | GPU / blocked |
|---|---|---|---|---|---|
| P0 环境/数据/监督审计 | 资源、依赖、codec、上游commit/源码/权重SHA256、许可证文本、清单分离、角色/来源组检查、源码快照 | 预算/分组/泄漏守卫已测 | 读取真实checkpoint训练/选模记录 | 24条dev清单与上游训练/选模清单精确ID比对；外部metadata版本hash | 本会话未暴露GPU；项目为非Git快照；完整源映射缺失 |
| P1 真实专家 | AASIST、SSL+AASIST、RawNet2冻结接口；同crop/eval/聚合；spoof−bonafide方向；SSL兼容处理；推断只接ID和波形 | label-free接口、资源缺失fail closed | **3专家通过**：hidden=160/160/1024，上游hidden/logits误差均0；identity/lossless均0；检查后继续预测亦精确一致 | 检查使用真实ASV2019 dev音频；不是独立效果评价 | CPU通过；RawNet2上游训练历史未知；无GPU验证 |
| P2 处理/缓存 | 完整音频filter、共享noise方向、真实MP3；独立时序控制；原子内容寻址缓存、完整性校验、源组分片/严格合并、逐请求status | identity/lossless、物理SNR、确定性、mask、缓存变更/损坏/重复、分片与合并 | before/after控制与真实模型通过 | dev：168音频视图/504有效expert-view；外部同样504有效expert-view | 四卡/单卡数值对照未运行 |
| P3 普通融合先行 | score LR/MLP、original MLP、TTA统计、全视图/response/no-delta MLP；train-only scaler、origin等权、保存/恢复 | BCE、mask、save/load、optimizer resume、fit/calibration/threshold隔离 | 使用真实缓存，无随机专家 | 先完成7基线×3seed，再进入P4/P5；8 train/8 val/8 calibration；每模型≤20步 | 超小样本，只是工程集成 |
| P4/P5 本体与消融 | registry→真实数值边/RDF；2层typed、同邻接untyped、全连接attention；错误family实际成员/保度数、三类删边、无响应、单专家、缺探针；参数匹配及更大容量基线 | 更换成员改变计算、重命名不变、mask不经bias泄漏；Python溯源校验 | 真实专家缓存进入分类器；关系参与计算 | 共**78个**有界训练运行、3 seed、相同证据与训练/选择预算；每次精确save/load校验 | RDF语法测试与SHACL引擎受缺失依赖BLOCKED；完整formal调参尚未做 |
| P6 冻结/未知处理/外测 | 独立Platt校准、开发阈值；冻结checkpoint/探针/本体/测试清单/音频内容hash；AAC/Opus处理DAG暴露账本；逐请求预测、独立重算、组bootstrap | role/manifest/audio漂移守卫、全覆盖、阈值、源重叠未知状态 | 实际冻结模型推断与单文件infer通过 | **8个外部origin、7个speaker组、3条件=24请求；两方法×3seed，全覆盖**；每seed 2000配对组bootstrap | 只允许称融合训练未见算子；上游暴露未知；ASV2021全测/映射子集和ASV5未完成 |

## 本轮验证层级与实际命令

所有`python`均指 `${USER_HOME}/anaconda3/envs/py310/bin/python`。

| 层级 | 实际命令 / 操作 | 退出码与证据 |
|---|---|---|
| unit/synthetic | `python -m pytest -q tests/test_core.py tests/test_pipeline.py tests/test_workflow.py --junitxml=reports/implementation/unit.xml` | **退出0，74 passed，0 skipped**；`unit.log`、`unit.xml`；含合成端到端训练/恢复/校准/冻结/评价，绝非真实鉴伪效果 |
| 完整unit集合 | `python -m pytest -q tests --junitxml=reports/implementation/full-unit.xml` | **退出2**：`tests/test_registry.py`收集时缺rdflib；3个旧RDF测试未运行，不算通过；`full-unit.log` |
| Python编译 | `python -m compileall -q ontoresponse reference scripts` | 退出0 |
| 环境/来源 | `python -m ontoresponse doctor` | 退出0，`doctor.log`、`audit.json`、`runtime-lock.json`、`supervision-audit.json`；doctor内nvidia-smi退出9 |
| 数据准备 | `python -m ontoresponse prepare-data --max-origins 24` | 退出0，`prepare.log`、`splits.json`、`audit_manifest.jsonl`、label-free的`inference_manifest.jsonl` |
| real-checkpoint | `python -m ontoresponse check-experts` | 退出0，`real-checkpoints.json`、`check-experts.log`；额外predict→上游检查→predict回归退出0，`real-checkpoints-final.json` |
| real-data提取 | `python -m ontoresponse extract --max-origins 24` | 最终退出0，`extract-final.log`、`extraction-0-of-1.jsonl`；3专家/6探针全部有效 |
| real-data融合 | `python -m scripts.run_bounded` | 退出0，78 runs，42.27s；`comparison-final.log`、`comparison.json`、`selection.json`；普通基线先完成的凭据`p3-completed.json` |
| 本体Python验证/RDF | `python -m ontoresponse export-ontology` | 退出0，`ontology-final.log`、`evidence.ttl`；不是SHACL引擎通过 |
| SHACL引擎 | `python -m ontoresponse export-ontology --shacl --output reports/implementation/evidence-shacl.ttl` | **退出2 BLOCKED**：缺pyshacl，Python验证与RDF导出已完成；`shacl.log` |
| 外部准备/环境 | `python -m ontoresponse prepare-external --max-origins 8`；`python -m ontoresponse make-environments` | 两者退出0，`prepare-external.log`、`make-environments.log`；均为预先固定样本/条件 |
| real-data冻结外测 | `python -m scripts.evaluate_bounded` | 退出0，184.34s，含校准/冻结→外部提取→评价→组bootstrap；`external-evaluation.log`、`external-results.json`；逐条预测在各run的`external-evaluation/` |
| 独立重算 | `python -m ontoresponse report --manifest reports/implementation/runs/typed_response-fixed-seed17/external-evaluation/per_utterance_predictions.jsonl --output reports/implementation/recomputed-typed-seed17.json` | 退出0；与原report全部可重算字段精确一致；`report-recompute.log` |
| 单文件infer | `python -m ontoresponse infer --audio ${DATA_ROOT}/asvspoof2019/LA/ASVspoof2019_LA_dev/flac/LA_D_1047731.flac --checkpoint reports/implementation/runs/typed_response-fixed-seed17/model.pt` | 退出0，`infer.log`；一条输入一项logit/p，未传标签 |
| GPU | nvidia-smi/PyTorch检测、`/dev/nvidia*`检查 | **BLOCKED / 未测试GPU**。用户报告4×A6000，内核有NVIDIA模块，但本会话无设备节点，torch CUDA=False；`gpu-access.json`。未改驱动/结束他人进程 |

完整阶段运行journal见 `reports/implementation/final-run-journal.json`；资源/成本见`cost.json`。CPU每进程4线程；RAM约125GiB、数据盘约3.1TiB空闲。默认24个dev origins+8个外部origins，小于64；每融合run最多20 optimizer steps。单crop、3专家×7视图=21 expert-view/请求；原始缓存观测约37,753 bytes/请求（另计音频artifact/元数据/模型）。没有启动专家重训、全量下载或正式全量实验。

## 小规模真实结果：只说明工程闭环

fusion_val仅8条，多种模型EER都为0；普通基线在3个seed平均验证EER、再以NLL打破平局后，选中width128的`all_views_mlp`，不是按外部结果选择。typed约224,385参数；选中的更强普通基线1,230,337参数，另有接近typed参数量的强对照。

下表每条件只有8个origin（4真4假）；无新增处理、AAC、Opus均保留完整请求。校准与阈值都未使用外部数据。

| 方法 | seeds | AAC/Opus macro EER | no-new-processing EER | 固定开发阈值的外部FPR / TPR | 覆盖 |
|---|---|---|---|---|---|
| val选中普通全视图MLP | 17/29/43 | 各25% | 各25% | 各0% / 75% | 各24/24 |
| typed关系模型 | 17/29/43 | 各0% | 各0% | 0% / 25%、50%、50% | 各24/24 |

**不利结果亦保留：typed的EER较低，但固定工作阈值TPR更差。** 开发阈值仅4条真语音，不能稳定支持1% FPR；配置和报告明确标记`low_fpr_estimate_supported=false`。EER曲线摘要不替代工作阈值结果，不据外测反转分数/改阈值/重校准。

主差值macro EER(typed)−macro EER(MLP)=-25个百分点；每seed配对speaker/source组bootstrap的95%区间均为**[-66.67, 0]个百分点**，包含0。独立组只有7，2000次中1993次可计算、7次退化均明报；没有把3seed×3环境当独立样本。这个结果不支持本体增益、显著性或普遍泛化声明；错误family/untyped在验证集也达相近水平。尚无外部结构消融结论。

## 尚未完成 / BLOCKED

1. **严格监督隔离未满足。** 两个专家的fit为官方train 25,380条，与本次dev样本ID无重叠；其source_val为5,654条，与本次fusion_val/calibration各重叠2条。RawNet2为本机detectionLib真实源码/权重组合，训练/选模历史未知。不能把当前融合称为专家未见holdout。
2. **完整来源映射未锁定。** 当前按speaker/source_group隔离，跨speaker源映射未知；In-the-Wild原始下载版本未知，已固定metadata/实际音频hash，不把未知来源当独立。ASV2019/2021→VCTK/VCC官方录音级映射、ASV2021全测及已知重叠剔除子集未完成。
3. **正式预算未授权/未执行。** 未在更大独立fusion_train/dev/calibration/test开展正式多seed调参；ASV5本地只有协议归档等，完整音频/官方Track1工具未审计。正式效果仍需隔离专家监督、来源映射、预先冻结协议与新测试集。
4. **GPU与SHACL未完成。** 分片/合并只有CPU/单元验证，没有真实四卡完整性/数值对照。临时安装rdflib/pyshacl因网络限制失败（退出1，`dependencies.log`），没有修改系统或现有环境；缺失不能记为skip通过。
5. 仍可扩展正式质量控制、3/6探针预算曲线、逐专家移除、数据集原生导入/映射和正式官方工具封装。ASV2021完整包装器已有，但尚未真实全测；ASV5入口明确BLOCKED。

首次check-experts因准备清单尚未写完退出2，已顺序重跑通过；第一次In-the-Wild准备因实际标签拼写`bona-fide`退出2，按真实metadata接口修正后通过。最初两个MP3视图因长度差明确missing，随后独立脉冲控制确认lag=0且仅尾部多26 samples；记录控制/裁尾后重提取，最终504/504有效。历史失败未伪装成通过；最终证据以final日志/索引为准。

本项目`.git`为只读空占位，无法得到project commit；已使用源码内容hash、对应源码tar快照、配置/权重hash锁定。`reports/implementation/code-snapshots/`保存实际提取版本与当前版本；上游工程与原始数据未改。旧`reports/unit_tests.xml`的36通过是原容器历史结果，不计入本轮。

复现命令、路径配置、缓存/本体语义及范围见 `docs/IMPLEMENTATION.md`。验收逐项状态另见 `reports/implementation/acceptance.json`，资源阻塞见`blocked.json`。
