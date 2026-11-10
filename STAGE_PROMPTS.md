# 分阶段给Codex的指令

## 首次启动
读取AGENTS.md、TASK_FOR_CODEX.md及docs。目标改为OntoResponse直接真假判别，不再继续旧风险矩阵/QP。先实际审计环境与资源，运行参考测试，然后完成P1到P3可完成部分。缺权重或数据报告BLOCKED，不造随机输出。只运行有界smoke/pilot，维护STATUS.md。

## 真实专家与响应
按P1/P2实现真实专家embedding/margin、已知探针及缓存。重点验证上游一致性、eval模式、相同crop、codec延迟、同噪声方向和identity响应。输入不含label。提供实际日志而非仅空接口。

## 公平融合
先跑相同缓存的score、embedding、全view/response MLP，再实现typed、untyped和Set Transformer。直接BCE，不加误差矩阵或QP；按METHOD实现元信息和mask。报告每种方法参数量和预算。

## 正式实验
读取EXPERIMENT_PROTOCOL，输出将要锁定的角色划分、来源重合、T/P暴露账本、主对比、未知条件和运行预算。严格模式下专家不能见过fusion角色；资源不符不得冒称严格。冻结协议后才运行指定full预算。

## 独立审计
不要为了得到正结果改主指标。检查元数据泄漏、缺失伪线索、错误关系是否仅重命名、基线是否收到同信息、概率校准是否碰test、失败文件是否被删除。给出能独立重算的结果及剩余限制。
