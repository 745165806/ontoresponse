# 多条件融合实验

常规操作见[运行实验](RUN_EXPERIMENTS.md)。本页保留v2的实验设计，不要求SHA、文档审核或质量封存；旧完整说明见[历史版本](history/DEVELOPMENT_V2_20261112.md)。

1. 用完整来源映射建立speaker/source连接组；同一组件只能分配给一个角色。映射JSON包含mapping_complete和recordings，每条recording提供id、origin_id、speaker_id、linked_speaker_ids、source_ids。必须覆盖所用官方协议，不能把缺失映射标为complete。
2. 排除专家训练/选模型接触过的来源及已消费探索来源。保留来源说明与样本清单；未知来源不能当独立holdout。来源组合不满足角色/标签/组数要求时，调整设计，不能改成逐文件随机切分。
3. 准备一行/origin的base_manifest，再用materialize_development生成环境T。T选择由origin/seed决定，标签不参与；失败请求保留。专家提取随后执行诊断P，T与P分开。
4. prepare_connected的--audit-materialized实时检查组隔离、条件集合和跨角色重复音频。freeze_development只提供可选可读配置快照；run_pilot无需先创建预注册文件。
5. 标准化只拟合fusion_train。每epoch每origin抽一个有效T，各基线使用同一日程、信息和预算。验证使用明确条件的macro EER，不能池化不同条件分数冒充宏指标。
6. 验证集选择架构/超参，收敛和H1/H2规则按明确配置执行。日志保存BCE、每条件EER、lr、梯度范数和独立源数。达到步数上限不等于收敛，小样本不支持正式增益宣称。
7. 选好模型后用calibration拟合校准、fusion_val选阈值，最后评价test。组数不足或无判别空间时保留未决结果，不以技术校验代替研究证据。

pilot需显式enabled和预算。现有开发配置仍是未启用的候选设计，不会自动下载数据或启动全量训练。完整source universe不足时，普通已分组集成实验仍可通过主CLI运行，但必须披露公开专家接触与来源未知的限制。
