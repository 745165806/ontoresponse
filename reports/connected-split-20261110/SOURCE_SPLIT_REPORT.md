# 来源连通划分与可行性验证 — 2026-11-10

## Material Passport

范围：source/speaker分组代码、unit/synthetic闭环和真实数据**元数据**可行性诊断。没有新专家推断、真实融合训练或新确认结果。H1/H2/H3仍未被新的真实开发实验验证。

## 本轮实现

完整来源宇宙在角色分配与freeze共用。除记录ID/源录音连接，还连接源/目标speaker和origin；消费记录及已知专家fit/selection所在的整个连通组件排除。完整映射必须覆盖本地2019 train/dev/eval与2021 DF metadata的所有ID，不能把只含选中样本的映射标为complete。官方archive、归一化映射和覆盖协议均固定hash。

新角色分配只接收组件容量、label和预设配额。整数约束保证一个组件最多一个角色、每角色各类origin数量、至少5个类内独立组，以及单组类样本上限。先最大化有效独立组数，再以固定seed/hash破同分；组内源录音配额轮转，避免最低组数通过但数据集中在少数组。整数解与轮转后的约束均独立复核。无解、求解超时、缺所选音频分别报告，不拆组、不按分数挑容易样本。

这是数据角色分配工具，**不是检测的QP/风险矩阵方法**；检测仍为现有BCE整句logit。SciPy/solver版本与实际manifest hash需固定，不声称跨solver版本的最优同分解完全相同。

闭环入口已连接：来源映射→`prepare_connected.py --candidates`→base manifest→历史T版本→`--audit-materialized`→data-audit→freeze。版本审计检查实际跨角色波形重复并生成pilot gate所需数据字段；没有来源映射时甚至不写base ledger。官方archive格式尚未获得，未猜写官方解析器；归一化接口不是已取得真实映射的证据。

## 本地可行性上界（不是认证新划分）

2019 eval 71237条metadata候选可用；真实音频没有用于本诊断。临时只按speaker独立的**乐观上界**有67组，其中bonafide组67、spoof组48。在512-origin提案上，分配得到以下伪角色结构，仅用于核对可行性，未输出实际角色manifest：

| 拟议角色 | 0/1源录音 | 0/1组数 | 0/1描述性Kish组数 |
|---|---|---|---|
| calibration | 32/32 | 14/11 | 13.47/10.89 |
| fusion_train | 128/128 | 14/9 | 13.98/8.99 |
| fusion_val | 64/64 | 21/16 | 20.90/16.00 |
| test | 32/32 | 18/12 | 17.07/11.64 |

真实source链接只会进一步连接这些speaker组，并可能把更多组件连到已曝光数据。因此上界可行**不能批准真实提取**。伪角色没有被写成base/test清单；candidate ledger没有role字段，没有被当成新的确认冻结。全部标签只用于数据配额和审计，没有forward。

每条T/P及所有同源衍生物仍以origin/source组件为单位；统计单位没有因环境数/crop数增长。元数据不能给EER差异的可靠功效；Kish数不是cluster bootstrap CI。拟议test只有32个bonafide origin，即使按不现实的IID零误报计算，单侧95%上界也约8.94%，不能支持1%低FPR。下一步须映射后重新核对实际独立组与开发分数变异；不得在新确认输出后改指标或成功条件。

## 实际命令与验证

- `complete-tests.command.json`：退出0，**166 passed、0 skipped**。新增6个测试覆盖分组/配额/输入顺序、文件多而组少、整组件排除、完整source coverage缺项、以及16个合成origin经T版本/波形审计/freeze的全链。没有用合成分数替代真实专家。
- `focused-tests`首次1失败/32通过：完整闭环fixture沿用了仅作张量标签的aac/opus简写，不是实际算子ID；改为明确的合法合成控制条件，并增加未知条件在写文件前失败检查；随后33个定向测试通过。门限和真实协议条件未改。
- `wheel-build/install/outside-install`均退出0；仓库外wheel与当前源码hash相同，新模块可导入，旧24条fusion_val缓存前向通过，无新拟合/确认数据。
- `candidate-inventory`退出0，71237条已有eval metadata导出到无role候选账本（不是test freeze）；`source-preparation-final`退出2，官方archive和完整归一化映射缺失，不写base角色，不消费新确认结果。
- `mapping-network-check`在无代理路径下退出6（DNS解析失败）；web工具能识别application/x-gzip但不支持读取并交付archive字节，workspace仍无映射包。没有下载任何全量音频。
- 再次核对原490冻结文件全部hash无变化；GPU依旧无法访问，nvidia-smi退出9，未改驱动或其他进程。正式GPU成本仍null，保留上阶段小CPU profile边界。

## 明确结论与剩余阻塞

**软件准备闭环可运行；真实新H1/H2不可运行。** 同一来源阻塞已在诊断、开发runner和本轮来源划分连续三轮核验。现有实现已完成可以独立于真实映射推进的必要准备；下一步必须取得官方映射archive及其可核对格式，完成真实解析/覆盖认证，才能分配认证source组件并提取新数据。未知字段不能靠猜测填齐，旧pilot和8条外测不能重新命名为新确认。

严格research_holdout的task HPO/开放SSL来源风险另行保留，当前只允许明确披露的integration研究。512-origin是有界开发提案，未启动144个H1 runs、H2或任何全量专家重训。H1无区分能力时应报告阻塞，不能强行给阴性/阳性结论。

操作说明见docs/DEVELOPMENT_V2.md，研究规范见formal_protocol.yaml。
