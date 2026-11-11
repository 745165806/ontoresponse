# 融合数据建议与全量运行条件 — 2026-11-11

## Material Passport

- mode: experiment run / metadata validation
- objective: 三专家完整 ASVspoof2019 LA eval；确定后续融合训练/选择/校准/测试数据。
- evidence: `reports/expert-benchmark-20261111/` 的真实 GPU 分数及官方/本地 metadata 审计。
- scope: 用户已授权完整评测及后续全量训练/测试；神经模型推断和训练使用 GPU。
- full_fusion_training: NOT_STARTED；本轮先完成独立专家质量门，以下新划分是可审阅方案，不冒称已冻结或完成训练。

## 1. 推荐选择

第一轮使用**本地现有 Codecfake（Xie）的 AISHELL3 中文来源部分**，将官方 train/dev 合并为开发候选池，再按真实说话人与原始录音来源整体重分配 `fusion_train / fusion_val / calibration / test`。不沿用官方的逐句 train/dev 作为说话人独立划分，也不使用任何 test 分数确定新划分。

选择此路线的理由：现有 AASIST/SSL+AASIST 的已知鉴伪监督来自 ASV2019 英语 VCTK；中文来源可减少直接监督来源重叠，并避免一开始混入与 ASV2019 共源的 VCTK 部分。通用 SSL 预训练与任务 HPO 来源仍单独记 unknown，不能因此宣称完整 research_holdout。RawNet2 的监督来源未知，不能靠本轮 EER 自动补齐认证。

本地已逐文件检查 Codecfake train 740747 条、dev 92596 条，均存在。按文件命名解析的候选中文部分共 554806 条目标录音（79258 条原始源录音及六种 codec 重建）；218 个说话人在官方 train/dev 中全部重叠。说话人命名规则还需与 AISHELL3 原始 metadata 核验。

一个**未冻结、未运行**的可重复候选划分为：按 SHA256(`OntoResponse-Codecfake-CN-full-v1-17:` + speaker) 排序，前 152 个说话人训练，之后各 22 个用于选择、校准、测试。这里只用身份 metadata，无专家分数参与。

| 角色 | 说话人 | 目标录音数 | bonafide | spoof |
|---|---:|---:|---:|---:|
| fusion_train | 152 | 387023 | 55289 | 331734 |
| fusion_val | 22 | 52808 | 7544 | 45264 |
| calibration | 22 | 58254 | 8322 | 49932 |
| test | 22 | 56721 | 8103 | 48618 |

同一个 source 的 real 和各 fake 文件必须同组，但它们各自是具有自身真假标签的目标录音，不能把真假衍生混成一个 origin 标签；每条目标录音的所有历史 T 版本、原始/P 观测继续归于该目标 origin。

完整中文池的三专家×七视图约 1165 万 expert-view；即使不新增历史 T，仍显著大于旧 pilot。正式启动前应使用本轮实测吞吐和真实时长/crop 数确定 GPU 小时与缓存配额，并使用批量提取与 minibatch 融合训练。现有 smoke CLI 不应通过去掉预算检查来冒充 full runner。

## 2. 实施与评价顺序

1. 独立专家质量/来源：固定 checkpoint 和 native-score 口径。本地 AASIST 与作者发布权重分开报告，不在 eval 的历史 epoch 中选最优。
2. 数据审计：认证中文 speaker/source 标识；检查所有 real/fake 衍生及其跨角色音频重复；隔离已消费数据。metadata 符合不等于音频重复审计通过。
3. 统一音频处理：本地抽查 real 为 44100/48000 Hz、fake 为 16000 Hz。按相同、与标签无关的规则重采样为 16 kHz 单声道、统一无损容器，保留输入/output hash，不直接把它们送入只接受 16 kHz 的现有 extractor。
4. 预先固定 T/P、crop、三 seed、候选网格、源等权策略、收敛与存储预算；scaler 仅 fusion_train 拟合。专家与融合神经计算全部在 CUDA，CPU 仅作数据读取和指标统计。
5. 先完整训练原始分数/embedding 基线和同信息 probe/response 基线，验证额外观测 H1；H1 有区分能力且支持后再做 typed、relation_tied、full_attention 和错误成员关系 H2。
6. 每角色保留 no-new-processing 和预定 T 条件，未知处理的评价主张以完整暴露 DAG 为准。Codecfake 本身的 neural-codec 标签与系统 MP3 诊断 P 是不同操作。
7. 只在 fusion_val 做模型选择；独立 calibration 做 Platt；阈值在开发角色固定；所有模型锁完后读取 test。报告请求覆盖、每条件 EER、条件宏 EER、固定点 FPR/TPR、三 seed 和源组配对统计。

此方案检验的是 codec 重建鉴伪和处理响应融合，不能单凭结果声称对所有真实 TTS/VC 泛化。新生成器可用 Codecfake C7/ALM 的来源隔离子集确认；In-the-Wild 保留为真实传播域外测，剔除旧实验已经消费的源/说话人，并分别报告完整官方与适用子集。

## 3. 各数据集的用途

| 数据 | 建议用途 | 本地/协议限制 |
|---|---|---|
| ASV2019 LA train/dev | 现有专家复现、常规集成对照 | 已参与现用专家训练/选择；不能当专家未见过的严格融合数据 |
| ASV2019 LA eval | 本轮完整独立专家 benchmark | 71237 条全部已消费为 benchmark；不得按本轮分数选 checkpoint/crop/拓扑，不能冒称从未看过的新确认集 |
| Codecfake Xie 中文来源 | 近期融合 full 候选，重新按 speaker/source 分组 | 音频已在本地；需统一采样率、认证 speaker 与跨角色字节/波形重复审计 |
| ASVspoof5 Track1 | 下一阶段更现代的主训练/开发/确认路线 | 本地仅 protocols、README/LICENSE，无训练/开发音频；不自动下载全量。遵循官方协议与工具，dev 中独立留 calibration |
| ASV2021 DF | 冻结后的外部 codec/传播压力测试 | 官方映射本轮已取得，覆盖611829条本地 keys；须处理2019/VCTK/VCC源重叠，保持官方eval/progress/hidden区别 |
| In-the-Wild | 真实传播域外部测试 | 保留旧已消费源账本；不一边外测一边反复选融合模型 |
| WaveFake / VCTK-based CodecFake | 补充训练/外测 | 注意原始 VCTK/LJSpeech 的同源派生，不能把文件级随机划分当来源独立 |

## 4. 本轮已解决与新发现

官方 `ASVspoof2019_2021_VCTK_VCC_MetaInfo.tar.gz`（约16 MiB）已从 ASVspoof 官网下载成功并校验真实 TSV 格式。2019 train/dev/eval 官方 ID 全覆盖，2021 DF 本地 611829 个 ID 也全覆盖。本轮没有改写旧的 unknown/blocked 历史日志。

审计显示 2019 train/dev **共享1043条声学源录音**；连接 VC 源说话人与目标说话人的保守分组下，train/dev 共同只有两个组件，eval 另有两个组件。相应地，不能把默认 speaker-only 分割直接升级为满足本项目多角色独立组要求的认证划分。这也是不建议继续围绕旧24条或128条零EER pilot扩大结论的具体依据。

该映射的 TTS 行给出输入文本而不是原始声学录音 ID；审计没有把文本伪造为已知音频来源。获得 archive 解决了下载阻塞，但不会自动认证所有专家训练/HPO、通用SSL曝光或新融合划分。

来源：
- [ASVspoof2019 官方数据及映射入口](https://www.asvspoof.org/index2019.html)
- [ASVspoof2021 官方 keys/映射与评测入口](https://www.asvspoof.org/index2021.html)
- [Codecfake Xie 作者仓库](https://github.com/xieyuankun/Codecfake)
- [ASVspoof5 官方工具](https://github.com/asvspoof-challenge/asvspoof5)

实测依据：`official-source-map-audit.json`、`codecfake-candidate-audit.json`、各专家 `measurement-lock.json` / `result.json` / `per_utterance_predictions.jsonl`。
