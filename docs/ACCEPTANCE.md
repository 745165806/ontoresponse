# 验收要求

| 编号 | 验证对象 | 必需证据 |
|---|---|---|
| A01 | 整句任务 | 一条输入一项logit，无局部真假label/定位指标 |
| A02 | 专家适配 | 实际checkpoint与上游同预处理输出容差比对 |
| A03 | before/after | 同专家、同当前录音派生、同crop；identity响应近零 |
| A04 | 推断隔离 | 改audit中的label/attack/speaker/env不改变forward输入和输出 |
| A05 | 训练隔离 | test不能改变normalizer、PCA、选模型、阈值或校准 |
| A06 | 视图掩码 | NaN missing与有效0不同；missing不能通过bias泄漏 |
| A07 | 来源与重复 | 真正同测量去重；相同数值但不同专家不去重 |
| A08 | 图语义 | 本体registry→邻接→RDF同hash；同probe边指同artifact |
| A09 | 类型作用 | 删除类型和改变实际成员确实改变计算；纯family重命名不改变 |
| A10 | 神经计算 | BCE反传、save/load、mode/eval、确定性合格 |
| A11 | 强对照 | 相同证据、元信息和调参预算；无类型attention不被刻意弱化 |
| A12 | 全覆盖 | 每条请求都有status，不能悄悄删失败/困难录音 |
| A13 | 未知条件 | 暴露账本检查T、P(T)、专家增强；分类命名准确 |
| A14 | 多卡提取 | 互斥分片、集合完整、合并输出与单卡一致 |
| A15 | 统计 | 源组配对bootstrap与多seed；不把派生录音当独立 |
| A16 | 真伪性 | synthetic、unit、真实checkpoint、真实数据结果明确分开 |
| A17 | 数据版本 | ASV2019/2021映射审计，ASV5用官方Track1工具 |
| A18 | 能力声明 | typed≈untyped就不声称本体有额外检测贡献 |

当前参考模块仅覆盖其中一部分数值单元要求；实际状态见STATUS.md，不将该表作为已完成列表。
