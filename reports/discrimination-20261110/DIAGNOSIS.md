# 区分能力诊断与新开发协议

## Material Passport

模式：validate / prospective experiment plan；状态：ANALYZED。输入为已消费pilot、旧探索账本、现有checkpoint及本地数据metadata。新实验输出均位于reports/discrimination-20261110；没有新确认预测。真实checkpoint测量为CPU，效能假设尚未验证。

H1、H2、H3定义保留docs/RESEARCH_PLAN.md第14–16行。本次不是把负结果改成正结果，而是修正可识别性结论：旧停止决策保持历史原样；新增解释为INSUFFICIENT_DISCRIMINATION，后续需新预注册开发集。

## 当前角色与混淆

| 角色 | N / 0 / 1 | speaker组（0/1） | speaker内多数标签占比 |
|---|---|---|---|
| calibration | 16 / 8 / 8 | 3 / 1 | 0.812 |
| fusion_train | 64 / 32 / 32 | 6 / 5 | 0.578 |
| fusion_val | 24 / 12 / 12 | 3 / 1 | 0.833 |
| test | 24 / 12 / 12 | 3 / 1 | 0.833 |

完整speaker/source/attack/label交叉表在role-distributions.json。source_group目前就是speaker，并非认证的录音源组。验证、校准、测试均只有一个spoof speaker；多个bonafide-only speaker使身份与标签相关。表中多数标签占比是描述性的组内汇总，不是训练出的分类器或留出预测性能。不能把speaker关联直接当成专家用了身份捷径的因果证据。attack标签本身描述合成方法，与label关联是预期，不准进入forward。

fusion_val基线3seed EER全0，预设绝对改善0.01数学上不可实现。验证3组重采样有8/27=29.63%概率缺spoof类；少组且统计退化。H1-reassessment.json只读fusion_val；原测试结果未参与新规则选择。没有把零宽CI当等价性证明。

## 训练、验证与测试消费

全部18配置均有64条训练/24条验证/16条校准/24条测试；best_single_expert不优化，其余15配置160–400步。所有模型验证EER=0。完整曲线位置/hash、首次/末次train与val BCE、最佳epoch、尾部变化量、停止原因在training-diagnosis.json，原PNG不改。

| 模型 | seed | 步数 | best epoch | stop | val BCE |
|---|---|---|---|---|---|
| best_single_expert | 17 | 0 | 1 | max_epochs | 0.366265 |
| best_single_expert | 29 | 0 | 1 | max_epochs | 0.366265 |
| best_single_expert | 43 | 0 | 1 | max_epochs | 0.366265 |
| original_mlp | 17 | 400 | 100 | max_optimizer_steps | 0.000118052 |
| original_mlp | 29 | 400 | 100 | max_optimizer_steps | 0.000123204 |
| original_mlp | 43 | 400 | 100 | max_optimizer_steps | 0.000195518 |
| response_mlp | 17 | 400 | 100 | max_optimizer_steps | 8.51067e-05 |
| response_mlp | 29 | 160 | 20 | validation_patience | 6.1272e-05 |
| response_mlp | 43 | 356 | 74 | validation_patience | 7.35445e-05 |
| all_views_mlp | 17 | 160 | 10 | validation_patience | 1.04482e-06 |
| all_views_mlp | 29 | 224 | 41 | validation_patience | 6.96199e-06 |
| all_views_mlp | 43 | 328 | 67 | validation_patience | 1.02577e-05 |
| response_mlp_p3 | 17 | 400 | 100 | max_optimizer_steps | 3.98135e-05 |
| response_mlp_p3 | 29 | 200 | 35 | validation_patience | 4.87861e-05 |
| response_mlp_p3 | 43 | 400 | 98 | max_optimizer_steps | 3.01801e-05 |
| all_views_mlp_p3 | 17 | 160 | 18 | validation_patience | 6.95453e-06 |
| all_views_mlp_p3 | 29 | 400 | 99 | max_optimizer_steps | 1.34641e-05 |
| all_views_mlp_p3 | 43 | 400 | 98 | max_optimizer_steps | 2.99341e-05 |

预算上限停止不单独证明收敛；BCE近零的训练曲线不能补救验证天花板或来源组不足。旧pilot未包含score_lr/score_mlp和tta_stats的完整三seed比较，新协议明确补齐，不能把旧18配置冒充六类基线已齐备。

已查看测试清单与角色在test-visibility.json；consumed-recordings.jsonl保守登记所有旧开发记录、8条外测、128条pilot及其身份。旧8条与pilot24条test已看过输出，其余pilot104条也不得重称全新确认。所有同源衍生物保持消费状态。无新测试音频/分数读取或用测试改阈值/模型。

## 本地来源、角色隔离与下一数据协议

local-inventory.json从本地协议实际统计：2019 train/dev/eval为25380/24844/71237条；2021 DF为611829条，来源类asvspoof/VCC2018/VCC2020。DF的corpus字段不等于原始录音映射；VCC2018 spoof仅4个speaker，不能硬拆成每角色>=5独立spoof组。官方映射入口已核验：[ASVspoof2021官方页面](https://www.asvspoof.org/index2021.html)。小metadata包curl退出7，未下载音频。待用官方映射连接源录音、源/目标speaker、派生版本，再打印每角色每类组数。

已审计专家fit/选权重来源继续沿用reports/codex-next-stage-20261110/supervision_matrix.json；公共任务HPO与通用SSL暴露仍UNKNOWN。新数据可用于明确披露的integration研发，不能称research_holdout。

formal_protocol.yaml的新开发提案512源录音：train256/val128/cal64/confirmation64；在完整映射上先排除专家fit/selection/已知HPO及所有已消费组件，再按固定hash、整连通组和每类>=5组分配，类内组轮转选录音。官方2019 eval若转为开发，所选组件及2021衍生物永久退出“未见官方测试”声明。ASV2021的历史codec标签是T，不能充当主动P。没有源映射时不输出假认证split或冻结test ID。

源映射接口ontoresponse/data/connected.py已测试跨speaker传递连接和缺失映射拒绝。此时尚无认证新划分；旧pilot runner只支持单训练版本/单条件选择，不能直接执行新增多T/多条件协议。下一实现应接入origin_uniform_epoch、按条件宏EER验证、协议冻结守卫；然后才提取/三seed运行新H1。

TP-exposure-ledger.json列出56条完整有序T→P组合：实际旧pilot7条（T=identity），拟议训练28条（4T×7views），拟议验证/确认21条（3T×7views）。拟议行不计为实际暴露。执行后还需每条artifact、encoder版本/命令/seed/时延/长度/failure元信息。真假同规则，T历史真值不进入forward。

## 整句覆盖与测量

保留64600采样点（4.0375秒）的原始single crop。新增cover策略顺序覆盖全句，尾窗右对齐；短输入repeat。原始和探针完整波形先处理、长度严格相等，再使用同坐标；每专家只聚合成一个mean embedding与mean margin，不增加标注样本数。分批crop避免同时装入全部GPU输入。缓存identity含策略，旧缓存不覆盖。

128条中23条被旧crop部分覆盖；全体时长1.032–9.905秒；原128 crop在cover策略下为155 crop（约1.211倍expert计算），音频/探针缓存无需因crop重复。

real-multicrop.json：两个真实冻结专家×single/cover×两个已消费fusion_train时长极值，共8项通过；真实identity/lossless近零响应、低通相同坐标、重复精确、整批上游与分批聚合容差1e-4验证。每项保留hidden_dim、权重hash、坐标、覆盖率、耗时。未读取新确认音频；没有GPU测试。runtime.json：torch2.6.0+cu118、CUDA不可用、nvidia-smi退出9。

## H1、H2、H3评价顺序与预算

H1先比较score LR/MLP、原始embedding+score、同探针预算tta_stats、全视图MLP和response MLP，两个crop策略各3seed，4候选/家族。主指标固定为预先指定AAC64/Opus32条件宏EER，identity另报；开发条件及成功规则在新输出前冻结。原H1改善幅度0.01、至少2seed改善、BCE不恶化保持。开发集无法区分则停止、记录原因并新预注册，不换测试指标凑阳性。

优化上限200epoch/2000step，min40epoch、patience20、min_delta1e-6；每epoch记录源等权BCE、EER和预算。cap停止且无预注册收敛证据→INCONCLUSIVE_CONVERGENCE。三个seed不增加独立数据量。各基线共享train-only标准化、metadata、mask、版本抽样和候选预算。

H2仅H1有区分能力且支持后进行：全视图MLP、attention、untyped_union、relation_tied、typed、wrong-family、三类删边。typed/tied唯一改变关系W共享，其逐通道均值和LayerNorm相同；现有等权复制/梯度/缺失/同seed测试通过。union另列，因为改变邻居归一化。参数匹配宽度与更强容量基线同时报告，同信息同调参预算。错误成员固定token后改实际边并重训。无收益与不利结果均保留。

H3在独立来源、冻结确认组与未知融合处理检验，源组配对bootstrap2000，逐seed/逐条件、失败、固定阈值实际FPR/TPR全部报告；不得以注意力或响应大小宣称因果。

预计512源录音、1280当前环境版本、17920 single-crop expert-view调用；H1共144 run、总上限288000 optimizer steps，仍是有界pilot而非正式研究。4.0375秒均长假设波形缓存约2.16GiB，embedding数组按真实hidden维数另算，详见cost-estimate.json；8GiB缓存+2GiB输出只是拟议硬预算，需预检磁盘。CPU两时长的时间外推不等于吞吐量预测，GPU小时数仍null，不能捏造。

## 实际验证、失败与阶段结论

unit-suite-local-deps.command.json：py310，132 passed、0 skipped、退出0。tests含9项新增协议/crop测试与原relation_tied等测试；这些是unit/synthetic，不代表研究效果。real-multicrop.command.json退出0；最终real-multicrop-deterministic.command.json退出0，8项真实checkpoint/真实音频CPU控制通过（显式确定性算法与seed17，内存字段齐备）。

保留失败：首次diagnosis缺PYTHONPATH退出1，明确加项目路径后退出0；首次全测因缺rdflib收集失败退出2，pip受网络限制退出1，使用已有${EXTERNAL_PATH} deterministic algorithms后用原逐bit断言重测8项通过，不放宽容差。官方映射curl退出7；GPU不可用。所有命令、日志、真实退出码存本目录。

原reports/implementation与reports/pilot-20261110的490文件hash逐一核对无变化，见frozen-integrity.json。

**结论：诊断、阶段门、整句crop实现及真实测量可运行；新科学H1/H2实验BLOCKED，尚未执行。** 阻塞包括认证source映射、专家HPO来源（严格研究）、可用设备与多T/多条件训练接口。可以继续实现多T协议runner，但不得据已有小pilot宣称H1失败、本体有效或全句性能改善；没有全量专家重训或大型下载。

统计误读核查11/11：Simpson（条件/组表保留，未合并推断）；生态谬误（speaker汇总不作逐录音因果）；选择偏差（既有有界平衡pilot不代表总体）；collider（未以结果筛人/录音）；基率（平衡样本后验不当部署概率）；均值回归（未按测试差样本重选）；幸存者（旧432请求失败0，新控制8项全部保留）；多重比较（18配置全报、不挑最好测试）；多路径（新协议与旧历史分开）；相关非因果（不做因果语言）；反向因果（未作方向性机制推断）。此核查不替代新来源审计或正式统计功效估计。
