# 历史资料

当前状态只维护根目录[STATUS.md](../../STATUS.md)，当前实施计划见[PLANS.md](../../PLANS.md)。本目录保留2026-11-11及以前的原始累积记录：

- [STATUS_20261111.md](STATUS_20261111.md)：原完整状态日志。
- [STATUS_NEXT_20261111.md](STATUS_NEXT_20261111.md)：原后续阶段日志。
- [PLANS_20261111.md](PLANS_20261111.md)：原实施计划。

以上内容逐字节保留，内部相对路径按当时项目根目录理解，不表示某个历史阻塞至今仍成立。真实实验及冻结证据仍在原`reports/`、`artifacts/`和`deliverables/`路径，清理不改写这些产物。

## 精简前源码

`reports/cleanup-20261112/source-before.tar.gz`包含当时源码、测试、配置与文档，`before.json`保存其SHA256和既有证据清单。核心代码字节hash变化后，旧实验身份仍对应旧快照，不用新源码hash覆盖旧锁。

移出的12个一次性脚本：

```text
audit_split_feasibility.py    check_outside_install.py
check_v2_install.py          diagnose_discrimination.py
h1_diagnostics.py           metric_conventions.py
report_connected_split.py   report_development_runner.py
report_discrimination.py    revise_protocol.py
prepare_github.py           stage_benchmark_docs.py
```

这些工具写死旧输出/安装/发布目录，不是当前运行接口。查看原文可使用：

```bash
tar -xOf reports/cleanup-20261112/source-before.tar.gz scripts/report_connected_split.py
```

`reports/cleanup-20261112/removed.json`列出实际删除对象、大小、原因和被移出脚本的原始hash。构建输出与字节码缓存可由安装/运行重新生成。
