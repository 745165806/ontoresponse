# 融合数据建议与全量运行条件

## 当前第一轮方案 — 2026-11-13用户指定

用户最新要求：仅使用本地ASVspoof2019 LA train/dev开发，ASVspoof2021 LA eval和In-the-Wild测试。2019 LA eval已移出本轮训练、选择及测试范围；保留原始音频和历史专家benchmark。下方2026-11-11的Codecfake方案保留为后续扩展，不再是第一轮优先路线。本轮为方案调整及本地文件检查，尚未训练新的融合模型。

| 数据 | 协议条数 | 第一轮用途 |
|---|---:|---|
| 2019 LA train | 25380 | 与dev组成融合开发候选池，按来源重新分组 |
| 2019 LA dev | 24844 | 与train统一处理来源重叠，用于开发选择 |
| 2021 LA eval阶段 | 148176 | 固定模型后的信道鲁棒性测试 |
| 2021 LA progress / hidden阶段 | 16464 / 16926 | 如评价则单独报告，不混入eval EER |
| In-the-Wild | 31779 | 固定模型后的真实传播域外部测试 |

本地三个2019协议、2021完整keys及ITW meta.csv中的音频路径全部存在（缺失0）；这里只验证文件存在，没有重新解码全部音频。2019 dev/eval文件夹各有24986/71933个FLAC，多于协议条数，因此按协议ID读取，不能直接遍历目录作为测试集。ITW含54个说话人、19963条真音频和11816条伪音频。

2021 LA音频：`/media/dell/data/fakedata/asvspoof2019/LA/ASVspoof2021_LA_eval/flac`；完整keys：`/media/dell/data/fakedata/asvspoof2021/LA-keys-full/keys/LA/CM/trial_metadata.txt`。共181566行，第8列区分eval/progress/hidden。现有默认配置只接入2021 DF，LA仍需独立入口和来源核查；旧来源映射报告覆盖2019和2021 DF，不能用来证明2021 LA来源独立。[官方LA数据与keys](https://www.asvspoof.org/index2021.html)。

### 开发划分与结论范围

- 2019 train/dev已知共享1043条声学来源，不能直接把官方两个角色当成来源独立的fusion_train/fusion_val。先合并开发池，真实来源和派生整体同组；不使用eval或外部测试分数选择划分。
- 现有保守目标/VC来源说话人连接仅产生两个开发组件。候选做法是两折分组开发：轮流训练/验证，各折标准化只用其训练组；用开发折选择有限超参及固定训练轮数后，可在整个开发池重训最终融合模型。只有两个独立组，开发稳定性与置信区间解释均有限；已实现两组内分层抽样的64条smoke准备入口，可交换两折；正式两折优化和最终全量重训仍未实现。
- 第一轮主指标为EER，可直接使用未校准logit，不强行划独立calibration角色。若以后报告校准概率或固定工作点，需要另行确定开发侧校准/阈值方案，不能从测试集选部署阈值。
- 现用专家已在2019 train/dev训练或选权重，RawNet2来源仍未知，因此本轮属于integration_pretrained集成研究，不能声称专家监督未见的严格holdout。三份checkpoint先固定，不能按两个测试集的效果换权重。
- 2021 LA与2019的来源关系需单独核查；它承担信道压力测试，不预先称为完全独立语料。ITW应披露此前已消费样本，主表完整数据与剔除已消费来源/说话人的适用子集分开报告；不以成绩挑子集。

### 启动入口

使用`python -m scripts.prepare_round1`生成仅含2019 train/dev的64条smoke清单和三专家GPU配置；默认32条训练、32条验证，两个来源组件分别占一个角色。`--fold 1`可交换角色，需指定新的`--work-dir`；共享缓存，但各折分别拟合标准化。真实数据清单已生成在`reports/round1-smoke/fold0`，尚未提取或训练。具体命令见[运行实验](RUN_EXPERIMENTS.md)。

### 比较与执行顺序

1. 先在2019开发数据跑有界GPU pilot，不在两个测试集做方法探索。原始+六探针的观测缓存由所有方法共用，专家冻结，同专家裁剪和聚合一致。
2. 对照单专家、score_mlp、original_mlp、all_views_mlp、response_mlp、set_attention、relation_tied与typed_response；所有强基线获得同样的探针参数、专家身份及mask，匹配容量和选择预算。
3. 至少三个固定seed，训练和选择完成后仅在2021 LA eval、ITW分别报告EER与样本/失败数；2021另给各信道EER。两个测试集不合并为一个主要EER。
4. 用正确/错误关系成员和删边实验判断类型化关系贡献，保留无增益结果。探针P照常使用；第一张标准测试表不额外施加环境T，人工T压力测试另表，避免混淆官方条件。
5. 全量前补齐分批读取缓存、分批验证、断点恢复和资源估算；普通CLI仍是smoke/pilot，旧pilot仍阻止来源未知RawNet2，不能通过去掉预算或隐藏来源限制宣称三专家正式实验已就绪。

保留配置、分组表、checkpoint、训练曲线、逐句预测和指标即可，不新增SHA/hash审计。神经训练与推断使用GPU，CPU用于数据准备和指标统计。

## 以下为2026-11-11历史候选方案（Codecfake延后）

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
3. 统一音频处理：本地抽查 real 为 44100/48000 Hz、fake 为 16000 Hz。按相同、与标签无关的规则重采样为 16 kHz 单声道、统一无损容器，记录输入/output路径、采样率、长度和处理参数，不直接把它们送入只接受 16 kHz 的现有 extractor。
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
