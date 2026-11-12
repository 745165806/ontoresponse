# 多环境开发实验接口验证（2026-11-10）

## Material Passport

类型：软件实现、unit/synthetic与已有真实验证缓存复核；新科学H1/H2未执行。沿用附件目标及上阶段区分能力诊断，没有改H1/H2/H3定义、主指标或已看测试后的成功条件。没有新确认预测，原490冻结文件逐hash无变化。

本阶段补齐epoch级origin等权环境版本抽样、显式AAC/Opus条件宏EER、四候选×三seed家族选模、收敛门和确认集隔离。数据日程由origin/measurement ID、seed和epoch决定，不读取label/attack；与模型随机数状态分离。历史T选择/生成与诊断P接口分开，全部失败请求保留。每epoch日志含有效源数/测量数/梯度范数/lr/日程hash和逐条件指标。

新增pre-extraction preregistration锁：完整源宇宙连通组（包含排除记录）、消费与上游fit/selection来源、role/class组数、条件配对清单、label-free投影、音频字节、专家checkpoint、probe、本体和方法规范。配置或任何证据漂移都会阻止继续。未知source映射不允许退化成speaker-only新实验。两份single/cover配置保持disabled。

v2执行H1不支持、INSUFFICIENT_DISCRIMINATION或INCONCLUSIVE_CONVERGENCE时，不加载calibration/val/test进入确认。H2仅H1支持后运行；若确认获准，只评价已选候选，所有校准/阈值lock完成后再读test张量。旧冻结实验不适用新解释去重算/重选。

## 实际证据与失败项

- `final-complete-tests.command.json`：退出0，160 passed、0 skipped。包含六类普通基线×三seed、每run24步的**合成**多T/多条件训练与save/load；不能据此宣称正式训练效果或真实H1已验证。原relation_tied同归一化等价、梯度和缺失检查继续通过。
- `wheel-build-verified`、`wheel-install-verified`、`wheel-outside-verified`：均退出0；仓库外新wheel加载旧原始融合checkpoint与24条已消费fusion_val缓存，前向与v2条件指标接口通过。没有新专家提取、拟合或新确认读取。依赖仍使用此前已有的/tmp路径；正式复现按pyproject安装。
- `source-readiness.command.json`：预期退出2，缺官方archive、完整解析映射与认证角色清单；`freeze_written=false/test_used=false`。没有模型输出或新确认消耗。
- 初次focused-tests：1 fail/25 pass，正例fixture原本每类两个单类cluster，缺类概率12.5%恰触发预定10%退化门；修正正例为两个mixed-label cluster，未放宽门限，随后38项定向通过。
- `full-tests-final`：1 fail/157 pass；发现损坏ledger在比对hash前被解析，抛JSONDecodeError。已修正为先核对ledger哈希后解析；最终160项通过。
- 初次wheel外检多传了不存在的--unused参数退出2；保留失败日志，以正确参数和/tmp cwd重验后退出0。

## 来源、资源与下一阶段

上阶段角色/label混淆、duration/crop、消费账本、18个真实旧run诊断和56项T/P闭包见 `reports/discrimination-20261110/DIAGNOSIS.md`，不以本阶段unit结果替代。GPU当前仍不可用，真实GPU成本为null。新CPU成本仍只能引用两专家/两时长的小profile，不能推断GPU速度。

新提案512源录音：train256/val128/cal64/test64，共1280当前环境版本和17920 single-crop expert-view调用；H1两crop策略合计144 run，单run最多200epoch/2000step，cap-only不算充分收敛。4.0375秒均长假设wave cache约2.16GiB，evidence arrays约0.011GiB，另加模型、codec与日志；8GiB cache与2GiB输出为拟议硬预算，实验未启动。H2与正式full/专家重训均未授权执行。

**结论：多T/多条件软件接口、版本生成和冻结守卫已可运行；真实新H1仍BLOCKED（缺认证source映射与划分）。** 尚需官方archive解析、完整连接组件角色分配与源组精度审计，再在新缓存运行三seed充分收敛H1。H2取决于H1阶段门。任务保持active，不把软件通过或旧天花板当研究完成。

运行方法与映射接口详见 `docs/DEVELOPMENT_V2.md`。新状态与缺项见本目录requirement-audit.json。
