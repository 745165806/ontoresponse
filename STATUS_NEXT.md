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
