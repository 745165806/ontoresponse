# 当前状态 — 2026-11-12

主任务是一条录音一个真假logit。真实专家、探针、观测缓存、强基线、类型化关系和冻结评价已实现；**新融合full训练未执行，本体收益尚未得到充分实验验证**。

## 当前证据

| 层级 | 状态 | 证据 |
|---|---|---|
| unit/synthetic | 本轮回归171 passed、0 skipped | reports/cleanup-20261112/unit-final.log |
| real-checkpoint | AASIST / SSL+AASIST / RawNet2与上游输出控制通过 | reports/expert-benchmark-20261111/各专家/gpu-controls.json |
| real-data / GPU | 完整LA eval每模型71237条，失败0；A6000 GPU0/1/3 | [完整评测](reports/expert-benchmark-20261111/RESULTS.md) |
| 整理后兼容性 | 24条旧融合输出精确一致，三专家GPU抽查/仓库外安装通过；当时5792个证据文件hash不变 | [整理报告](reports/cleanup-20261112/REPORT.md) |
| 研究状态 | integration_pretrained；严格research_holdout尚不满足 | [数据方案](docs/FUSION_DATA_RECOMMENDATION_20261111.md) |

作者原生score EER：现用AASIST **2.121052%**、SSL+AASIST **0.637284%**、RawNet2 **5.084675%**；作者AASIST参考 **0.829511%**。现用AASIST未达发布权重水平；SSL论文2021结果与RawNet2变体不能作为本轮2019的直接复现判据。

## 当前限制

- 2019/2021官方来源映射已经取得；2019 train/dev共享1043条声学来源，保守source/target-speaker连接后组件不足以直接满足多角色要求。
- Codecfake现有train/dev共享说话人，抽查真假音频采样率不同；需要新分组、统一预处理和完整重复审计。
- 任务HPO、通用SSL曝光和RawNet2训练来源仍需认证；ASV2021正式全测与ASV5确认未执行。
- 旧pilot存在零EER天花板和独立组不足，解释为INSUFFICIENT_DISCRIMINATION、H1未决；H2未进入。
- 排版/导入整理会改变源码字节hash。旧缓存、预注册与质量锁仍对应旧源码，不改写旧身份来绕过守卫。

## 本轮精简

常用脚本33→21；12个一次性历史脚本已保存原文快照后移出常用目录。清理build、egg-info、字节码和工具缓存；统一代码格式/导入，合并重复EER计算，数据准备入口使用main。原模型类、state_dict键、探针/本体配置与既有实验产物保留。

源码快照及实际删除清单：`reports/cleanup-20261112/source-before.tar.gz`、`removed.json`。本轮命令、退出码、测试/安装/GPU兼容性及受保护文件hash检查均存于同目录。没有新训练或新确认实验。

## 历史与发布

- 旧完整状态：[STATUS_20261111.md](docs/history/STATUS_20261111.md)、[STATUS_NEXT_20261111.md](docs/history/STATUS_NEXT_20261111.md)。
- 旧计划：[PLANS_20261111.md](docs/history/PLANS_20261111.md)；脚本恢复：[history/README](docs/history/README.md)。
- 已发布的文档提交：[45dc09f](https://github.com/745165806/ontoresponse/commit/45dc09f387a2aa24b89b22b2a73dd252e0ee5e10)。上述提交标识整理前的公开文档；用户现已授权同步当前完整代码与小型报告。

## 后续存储清理与发布

两个冗余容器已完成验证后退役：旧baseline改为可恢复的差异快照，旧离线Git bundle已由远端公开历史覆盖；释放约512MiB。原baseline2463个成员实际恢复校验通过，其他checkpoint、分数、缓存和日志保留。详情见[存储清理报告](reports/storage-cleanup-20261112/REPORT.md)。本轮公开导出按每文件1MiB上限脱敏，同步当前精简源码与重要小型报告。公开候选代码171项测试通过、0 skipped，旧576条预测重算及三专家/融合GPU兼容性通过；命令、退出码、资源与限制见[发布验证报告](reports/repository-update-20261112/REPORT.md)。
