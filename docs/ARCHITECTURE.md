# 代码阅读指南

系统对一条录音输出一个真假 logit，0=bonafide、1=spoof。冻结专家提供观测，融合模型学习如何使用原始表征、诊断响应和类型化关系。

```mermaid
flowchart LR
    A[当前录音] --> B[原始视图和六个探针]
    B --> C[冻结专家 hidden / margin]
    C --> D[观测缓存和有效性 mask]
    D --> E[训练集标准化和响应 token]
    E --> F[融合模型]
    F --> G[整句 logit]
    G --> H[独立校准与固定阈值]
```

## 建议阅读顺序

| 阶段 | 主要代码 | 职责 |
|---|---|---|
| 入口 | [cli.py](../ontoresponse/cli.py) | 解析参数，调用数据、提取、训练与评价模块 |
| 音频 | [data/audio.py](../ontoresponse/data/audio.py) | 16kHz输入检查、固定/全句覆盖crop、质量特征 |
| 专家 | [experts/base.py](../ontoresponse/experts/base.py) | 加载真实任务权重、关闭梯度和增强、统一整句hidden及分数方向 |
| 探针 | [transforms/probes.py](../ontoresponse/transforms/probes.py) | 低通、共享方向噪声、真实codec和时间对齐 |
| 证据 | [evidence/extract.py](../ontoresponse/evidence/extract.py)、[dataset.py](../ontoresponse/evidence/dataset.py) | 共享视图、缓存身份、来源检查及张量组装 |
| 数值核心 | [reference/core.py](../reference/core.py) | 标准化、响应token、关系传播、mask池化、源等权BCE |
| 模型选择 | [models/factory.py](../ontoresponse/models/factory.py) | 普通基线、typed/tied/attention和受控消融 |
| 本体 | [ontology/registry.py](../ontoresponse/ontology/registry.py) | 同一registry生成数值邻接和来源RDF，校验真实观测对应 |
| 训练 | [train/fit.py](../ontoresponse/train/fit.py)、[pilot.py](../ontoresponse/train/pilot.py) | smoke与pilot预算、选模、恢复、收敛及阶段门 |
| 评价 | [evaluation/](../ontoresponse/evaluation/) | 固定方向EER、校准/阈值、全请求覆盖和源组配对统计 |

`reference/`虽然沿用早期名称，当前确实是生产模型复用的数值核心；保留其模块路径也保证已有checkpoint中的`ExpertStats`仍能加载。它不是可以删掉的演示副本。

## 张量与关系

- `hidden[i]`：`[batch, 1+K, expert_dim_i]`。
- `scores`、`valid`：`[batch, M, 1+K]`；视图0是当前未施加探针的输入。
- 原始节点：每专家一个，共M个；响应节点：每专家每探针一个，共MK个。三专家、六探针时为21节点。
- 原始与探针使用同一个train-only scaler，然后计算hidden/score差分；缺失响应必须同时缺少有效before或after，不能作为零变化。
- 图的四个有向通道对应：原始→响应、响应→原始、同探针跨专家、同家族同专家。两层传播后分别池化原始/响应节点，输出logit。

专家推断不接收标签、attack或历史环境；标签只参与训练损失和评价。探针参数合法可见。图约束测量之间的对应，不规定某种响应必然为真/假。

## 配置与可重复性

`configs/local.yaml`指定本机资产和历史smoke路径；`configs/probes.yaml`与`configs/ontology.yaml`是工作区测量定义。`ontoresponse/_resources/`保存wheel所需的同内容资源，这两份是打包接口要求，测试会检查一致性。

原始模块路径、模型名字、state_dict键和注册表配置保持兼容。源码排版会改变字节级`code_hash`，因此旧实验应使用当时源码快照；不能把旧缓存/质量锁的hash改写成新版本。精简前源码保存在`reports/cleanup-20261112/source-before.tar.gz`。

方法细节见[METHOD.md](METHOD.md)，实验纪律见[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)，操作入口见[SCRIPTS.md](SCRIPTS.md)。
