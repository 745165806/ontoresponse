# 清理与完整代码发布 — 2026-11-12

本轮同步当前精简代码、配置、文档与重要小型报告。常用脚本由33个减少到21个，历史状态移入docs/history，核心模块统一格式并清理重复逻辑。完整LA eval结果仍见[专家报告](../expert-benchmark-20261111/RESULTS.md)。

## 验证结果

| 验证层级 | 实际结果 | 证据 |
|---|---|---|
| unit/synthetic | 公开候选代码171 passed、0 skipped | public-unit.command.json / public-unit.log |
| 既有预测独立重算 | 576条旧预测，36张条件/方法/seed表；预测字节未变 | public-recompute.command.json / public-recompute.log |
| real-checkpoint / real-data / GPU | 三专家各8条已消费eval录音，输出误差均小于3.34e-6；旧融合24条已消费val录音CPU精确一致、GPU误差小于3.82e-6 | gpu-parity.json / public-gpu-parity.command.json |
| 静态检查 | Ruff检查与格式检查、源码/文档git diff --check通过；公开路径脱敏引起的一处脚本换行不改变AST；历史日志与上游README保留原始空白 | 发布前检查 |
| 原有科研证据 | 5790个旧文件hash不变；仅2个经过验证的冗余归档容器按清单退役，无计划外变化 | ../storage-cleanup-20261112/evidence-integrity.json |

GPU检查使用NVIDIA RTX A6000物理GPU0；atol/rtol均为1e-4。仅复查既有checkpoint和已消费录音，没有新训练、选模型、阈值拟合或新的确认实验。公开源码的字节hash与原始执行版本分开记录。

## 删除与保留

冗余归档清理净释放536859358字节（约511.99MiB）。旧baseline的2463个文件已实际恢复并逐成员核对大小和SHA256；13个独有历史文件保存在小型差异包，其他文件复用保留的完整包。旧Git bundle对应提交已包含在远端main历史。详情见[存储清理报告](../storage-cleanup-20261112/REPORT.md)。

## GitHub范围

每个公开文件最多1MiB；AUDIT_MANIFEST记录实际交付大小、SHA256与有意排除项。音频、权重、观测缓存、wheel、压缩归档和超限日志/完整benchmark逐条分数保留本地。当前Git树移除早期上传的超限文本，保留Git历史。

完整专家EER仅上传汇总和控制记录，因此仅凭公开汇总不能重算这些EER。旧576条融合预测仍可用根目录audit_recompute.py独立重算。详见[发布范围](../../PUBLICATION.md)。远端推送与读取校验回执另保存在本地，不把发布验证当成新的研究结果。
