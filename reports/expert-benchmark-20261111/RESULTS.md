# 三专家完整 ASVspoof2019 LA eval — 2026-11-11

## Material Passport

- mode: experiment run / reproducibility verification
- verification: VERIFIED_COMPLETE_GPU_BENCHMARK
- scope: fixed checkpoints and scores; no expert/fusion training, calibration, threshold or test-driven selection
- data: 71237 unique eval recordings, 7355 bonafide / 63882 spoof; all four runs complete, zero failures
- compute: NVIDIA RTX A6000, GPUs 0/1/3, float32, eval, gradients disabled; GPU2 not used by this benchmark

## 完整结果

| 权重 | 作者原生分数 EER (%) | 融合 margin EER (%) | 有效/请求 | GPU | 秒 |
|---|---:|---:|---:|---:|---:|
| aasist | 2.121052 | 2.080308 | 71237/71237 | 0 | 407.7 |
| ssl_aasist | 0.637284 | 0.611947 | 71237/71237 | 1 | 964.9 |
| rawnet2 | 5.084675 | 5.084675 | 71237/71237 | 3 | 253.1 |
| aasist_author_reference | 0.829511 | 0.815669 | 71237/71237 | 0 | 352.2 |

主 EER 使用作者原生 `output[:,bonafide_index]`，传入已锁 hash 的 AASIST 官方 `compute_eer`。分数方向在运行前固定。margin 是当前融合观测口径，其结果不能与原生 score 混为同一个实验。独立 grouped ROC 插值值另存 JSON，不能因细微口径差而选择较小者。

## 是否达到论文效果

- 现用本地 AASIST 未复现发布权重的 0.83%；本地作者参考权重实测约0.83%，支持评测链正确。没有修改当前融合配置或根据 eval 从历史 epoch 中挑模型。
- SSL+AASIST 官方 README/论文的0.82%是ASV2021 LA，2.85%是2021 DF；本轮2019 LA结果不能证明这两项已经复现。当前本地任务权重也没有相同的RawBoost增强训练历史。
- 原始RawNet2论文Mel S1/逆Mel S2/线性S3的2019 LA EER分别5.64/5.13/4.66%。本地RawNet2架构参数不同、checkpoint训练来源未知，不能仅凭EER接近便认证论文复现。

## 验证和审计

- 每个专家均通过真实GPU上游hidden/logit比对、batch/singleton容差检查、重复推断逐bit一致；每条请求输出status及音频sha256。
- 四份逐条文件均重新加载并重算；覆盖、结果hash、权重hash和同一ID的跨专家音频hash全部复核。
- unit/synthetic：新增5项；完整171 passed，0 skipped；这与真实checkpoint、完整真实数据、GPU证据分层。
- 来源映射archive已取得，2019全协议和2021DF全部611829个本地keys映射全覆盖；2019 train/dev共享1043条声学来源，保守speaker/source组件不足以直接满足多角色划分。
- 本地Codecfake train/dev共833343条文件全部存在，但speaker重叠且抽查原始采样率不一致；未在该不合格划分上启动全量融合。具体重分组方案见下方文档。

## 公开汇总与本地完整证据

- `summary.json` / `summary.csv`：全结果及权重hash。
- 各专家公开目录包含小型控制、结果和命令记录；完整逐条分数保存在本地，不随公开仓库分发。
- `official-source-map-audit.json`、`codecfake-candidate-audit.json`：无专家分数参与的数据审计。
- [融合数据与GPU全量方案](../../docs/FUSION_DATA_RECOMMENDATION_20261111.md)。

本地完整分数齐备时的重算命令：`${USER_HOME}/anaconda3/envs/py310/bin/python -m scripts.report_expert_benchmark`。

来源：[AASIST作者仓库](https://github.com/clovaai/aasist)、[SSL作者仓库](https://github.com/TakHemlata/SSL_Anti-spoofing)、[SSL原论文](https://www.isca-archive.org/odyssey_2022/tak22_odyssey.pdf)、[RawNet2原论文](https://www.eurecom.fr/publication/6395/download/sec-publi-6395.pdf)。

公开仓库包含当前精简源码与小型报告；完整benchmark分数未上传，仅凭汇总不能重算这些EER。此前576条融合预测可用根目录audit_recompute.py独立重算。见[发布范围](../../PUBLICATION.md)。
