# 面向实验精简 — 2026-11-13

常规实验不再使用hash/SHA文件审计、源码/文档审核锁或质量封存。保留数据角色隔离、同源分组、真实专家、view对应、缺失mask、训练预算、验证选模、独立校准及公平对照。

## 改动

- 删除实验公共模块的文件SHA/源码汇总函数；移除专家初始化、数据准备、训练恢复、校准、评测、pilot和完整专家benchmark中的文件摘要计算与比较。
- 缓存不再生成/复算数组完整性摘要；保留有限值、结构/视图、重复测量、原子写入和磁盘配额检查。旧缓存和checkpoint仍能读取。内部缓存键、确定性种子和重复音频标识仍可使用散列，但不构成审计清单或文档锁。
- pilot预检只检查输入、角色/分组、设备和预算；v2实时验证来源与条件集合，不要求预注册文件。freeze接口为可选兼容设置快照，evaluate无需手工freeze。
- 训练/评价/推断遵循runtime.device，默认本地GPU为cuda:0；新增work-dir、cache-dir、device、width、resume参数，extract支持单独输出索引。
- 新模型保存可读的probe设置、配置、样本/分组ID和普通run_id；run_id仅用于匹配同次训练的校准，不是内容摘要。
- 四个旧审计工具及根目录旧MANIFEST移入docs/history。公开导出改用PUBLIC_FILES.json，仅含路径/大小和排除原因，默认单文件1MiB，不上传模型/音频/缓存。

## 验证

| 层级 | 结果 | 记录 |
|---|---|---|
| unit/synthetic | 私有与最终公开候选172 passed、0 skipped | unit-final / public-tests.command.json及log |
| 既有指标独立重算 | 576条既有公开预测、36张条件/方法/seed表 | public-recompute.command.json / public-recompute.log |
| real-checkpoint / real-data / GPU | 三专家各8条既有eval录音比对通过，最大输出误差3.34e-6以内 | gpu-check.command.json / gpu-results.json |
| 旧融合兼容性 | 24条既有fusion_val输出CPU精确一致，GPU最大误差3.82e-6以内 | gpu-results.json |
| GPU训练与提取 | 8个train+8个val来源，2步typed_response训练及save/load通过；1条既有录音×3专家×7视图提取和实际关系检查通过 | fusion-smoke / extract-smoke / gpu-results.json |

GPU为NVIDIA RTX A6000物理0，真实专家未训练，其他GPU任务未动。末尾提取阶段记录的峰值分配为204561408字节，不代表整个检查过程的全局峰值。没有新全量训练、独立确认或测试集调参；smoke的EER只说明链路可运行，不代表正式研究效果。

测试必需的pyshacl/rdflib临时依赖目录在会话间已清理，本轮只将小型测试依赖安装到独立${TEMP_ROOT}

旧科研报告保持原状，其中的SHA只说明历史做法。当前实验步骤见[运行实验](../../docs/RUN_EXPERIMENTS.md)。
