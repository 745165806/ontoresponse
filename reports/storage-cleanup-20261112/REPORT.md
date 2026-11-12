# 冗余备份清理 — 2026-11-12

本次释放约 **511.99 MiB**。真实checkpoint、观测缓存、逐条分数、训练日志和独有历史内容保留。

## 已退役文件

- 旧baseline完整容器：约469.88MiB，改为基于较新完整包的差异快照，差异包约394.6KiB。
- 旧GitHub离线bundle：约42.50MiB；其main/HEAD提交已是远端公开main历史的祖先，连通检查通过。

保留的完整包：`deliverables/ontoresponse-v0.5-next-stage-20261110.tar.gz`。

差异包：`deliverables/ontoresponse-v0.5-baseline-20261110.delta.tar.gz`。

旧baseline共有2463个文件，其中13个历史差异文件保存在差异包，其余文件从保留的完整包复用。已经实际恢复至临时完整归档并由原`audit_bundle.py verify`逐成员校验全部大小/SHA256，通过后才删除旧容器和临时恢复文件。

**保留的是原始成员内容及原BUNDLE_MANIFEST，不声称恢复后的gzip容器能重现已退役容器的压缩字节hash。** 原容器hash与删除理由另存retention.json；旧阶段记录只说明当时的事实，不改写历史校验结果。

## 恢复旧快照

```bash
python3 reports/storage-cleanup-20261112/snapshot_delta.py restore \
  --base deliverables/ontoresponse-v0.5-next-stage-20261110.tar.gz \
  --source deliverables/ontoresponse-v0.5-baseline-20261110.delta.tar.gz \
  --output /tmp/restored-baseline.tar.gz
python3 scripts/audit_bundle.py verify /tmp/restored-baseline.tar.gz
```

恢复工具不extractall、不覆盖已有输出，并核对保留包以及每个成员的hash。

## GitHub发布范围

同步精简后的代码、配置、文档和重要文本报告。每个公开文件上限1MiB，超过上限的训练大日志、完整benchmark逐条分数，以及音频、checkpoint、缓存、wheel、压缩归档均不发布；这些实际科研资产保留在本地。公开manifest记录实际交付字节与有意排除项。

## 命令与资源

CPU归档/文件/Git操作，无新训练、GPU推断或新的科学实验。delta-create、delta-restore、restored-snapshot-verify均退出0，命令/耗时/输出见同名command.json与log。完整删除清单和回收字节见retention.json。
