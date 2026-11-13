# 新开发数据的多环境实验接口

当前两份配置 `configs/development-v2-{single,cover}-20261110.yaml` 保持 `enabled: false`。没有认证源映射或新数据实验；旧128条pilot与8条外测不能用作新确认。以下命令已具备实现，但真实新实验仍需数据/监督审计通过。

先做只读就绪检查（允许disabled配置，缺依赖退出2）：

```bash
PYTHONPATH=. ${USER_HOME}/anaconda3/envs/py310/bin/python scripts/freeze_development.py \
  --config configs/development-v2-cover-20261110.yaml \
  --report reports/development-runner-20261110/source-readiness.json
```

官方映射包应置于配置指定路径。归一化映射JSON包含 `mapping_complete: true`、`official_archive_sha256`、以及完整来源宇宙的 `recordings`；每条记录包含 `id, origin_id, speaker_id, linked_speaker_ids, source_ids`。各来源/speaker标识必须先消除跨corpus命名碰撞。该接口尚不包含官方压缩包到归一化映射的解析器；不能手工把缺失映射写为complete。传递连通计算包括已消费/专家训练与选权重的记录，不能只对选中的512条计算。

来源认证后，先用固定hash、类内组轮转形成一行/origin的base ledger，含 `id,origin_id,source_group,speaker_id,label,role,audio`。`source_group`必须等于完整来源宇宙的`connected_groups`结果，整组角色不能跨分割。每角色每类至少5个独立组只是可行性下限，不保证统计功效。512-origin提案并未实际划分。

在启用并冻结正确的有界预算后，版本生成使用：

```bash
PYTHONPATH=. ${USER_HOME}/anaconda3/envs/py310/bin/python scripts/materialize_development.py \
  --config configs/development-v2-cover-20261110.yaml --base-manifest PATH_TO_AUDITED_BASE_LEDGER
```

训练origin生成identity和一个以origin/seed确定的T（lowpass4500、white25或MP3_48）；其余角色按显式条件集合生成。处理整个波形后写统一16kHz WAV FLOAT，保留encoder/seed/长度/失败/哈希。标签不进入T选择或probe seed。生成失败保留请求，不能通过预注册。该脚本不运行专家，不生成P。专家提取随后把所有P应用于当前T音频，并保存共享artifact/测量账本。

生成完整数据审计后，用 `scripts/freeze_development.py --config ... --freeze` 封存配置/代码、完整角色与label-free推断清单、来源归一化映射及官方archive、消费/监督账本、真实专家权重、T音频、probe/ontology配置和formal_protocol规范。freeze前不得存在提取索引或模型输出；已有freeze不能覆盖。`pilot_gate.preflight`还要求独立质量门、来源审计、明确enabled预算与上游文档hash。

执行入口仍为 `python -m scripts.run_pilot --config ...`。v2每家族使用相同width/lr网格、三个seed；选择一个跨seed共用候选。每epoch每origin只抽一个有效T，同一seed下各模型/candidate的数据日程完全一致，不受模型初始化/dropout RNG影响。source-normalized BCE同时用于训练/曲线，best epoch按开发BCE；候选/家族按预定AAC/Opus宏EER，再以BCE和候选ID破同分。identity仅单列报告，不加入主EER。

每条件必须有两类及完整覆盖，否则不选模；不能把不同condition分数池化后冒充宏EER。日志保存每epoch梯度范数、lr、独立源数、测量数、日程hash和逐条件指标。达到200epoch/2000step上限本身不证明收敛。

H1 gate同时检查每条件每类独立组数、基线下界、bootstrap缺类概率和所有所选家族的收敛状态。INSUFFICIENT_DISCRIMINATION或INCONCLUSIVE_CONVERGENCE均不代表H1失败。H2只在H1支持时执行；否则不加载calibration/val/test进行确认，并写 `confirmation-not-run.json`。若允许确认，所有候选选择与全部校准/阈值lock先完成，再读取test张量；只评价每家族已选择的候选，逐条件报告失败和宏EER。

单crop与cover的两套缓存不会覆盖旧缓存。没有真实GPU成本测量；默认CPU配置不冒称GPU验证。实验记录与本轮验证见 `reports/development-runner-20261110`。合成测试仅验证软件接口，尚无新开发数据上的三seed H1/H2结果。

## 完整来源的base ledger准备接口

现已实现从已认证映射分配角色，不再要求手工形成base ledger：

```bash
PYTHONPATH=. ${USER_HOME}/anaconda3/envs/py310/bin/python scripts/prepare_connected.py \
  --config configs/development-v2-cover-20261110.yaml \
  --candidates reports/connected-split-20261110/asv2019-eval-candidates.jsonl
```

候选账本现已从本地2019 eval metadata导出，含71237条、没有role字段；它不是新确认集。只有完整官方映射/覆盖和曝光排除通过后才会生成`base_manifest.jsonl`。求解只接收连通组容量/标签/预设配额；先最大化有效类内独立组，再以hash破同分，组内样本配额轮转。该整数程序只用于数据角色安排，检测模型不变。固定SciPy版本和实际结果hash，求解超时不能写成“已证明无解”。官方2019/2021协议每个ID都必须在归一化来源宇宙中，不能仅映射选中512条。

base准备后按上文materialize命令生成T，再执行：

```bash
PYTHONPATH=. ${USER_HOME}/anaconda3/envs/py310/bin/python scripts/prepare_connected.py \
  --config configs/development-v2-cover-20261110.yaml --audit-materialized
```

此步骤重算来源/角色/配对条件/字节哈希，检查实际波形跨角色重复并写`data-audit.json`，之后才可freeze。该完整闭环已用合成源和合法波形测试，真实源映射仍缺失。`optimistic-feasibility.json`只是speaker独立假设下的metadata上界，不能代替此认证流程；它不写角色manifest，也不授权提取。
