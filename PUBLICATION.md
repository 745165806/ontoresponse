# 公开发布范围

当前GitHub同步包含精简后的`ontoresponse/`、`reference/`、`scripts/`、`tests/`，配置/依赖声明、方法/操作文档和小型实验报告。

## 文件策略

- 每个公开文件最多1MiB；超限文本列入AUDIT_MANIFEST的有意排除项。
- 音频、checkpoint、embedding/波形缓存、wheel、Git bundle和压缩归档均保留在本地。
- 私人绝对路径改为路径模板，凭据样式内容移除；配置本机资产后再运行真实提取/评测。
- 原始研究执行版本与公开脱敏字节通过private_source_sha256关联；不能把脱敏源码的hash冒充过往实验版本。

## 结果复核

四个完整专家eval的EER、逐attack汇总、GPU控制、运行命令和退出码公开；四份完整逐条分数保留本地。因此只下载公开汇总不能重算这些完整EER。

此前公开的576条融合预测仍以稳定假名和校准/阈值信息公开；在仓库根目录执行`python3 audit_recompute.py`可用标准库独立重算36张条件/方法/seed表。这不代表新的独立确认实验。

## 存储清理

重复完整备份已改为完整包+差异包，原baseline的2463个文件已实际恢复并校验。删除的是重复容器；checkpoint、分数、缓存、训练日志和独有历史代码内容保留。详情见[清理报告](reports/storage-cleanup-20261112/REPORT.md)。

旧报告记录当时事实。代码/文档精简、归档退役和后续发布分别记录，不修改过去的实验结果或把旧测试写成新研究效果。

本次公开候选代码的测试、GPU兼容性与证据完整性检查见[发布验证报告](reports/repository-update-20261112/REPORT.md)。
