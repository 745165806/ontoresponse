# 已核验的第一方资料与实现入口

核验日期：2026-11-09。以下为检索到的可用资料，不是穷尽查新；preprint不自动视为已发表。正式部署必须固定commit、权重hash、许可证和数据版本，页面可访问不等于已成功下载运行。

| ID | 资料 | 本项目使用范围 |
|---|---|---|
| S01 | [AASIST官方仓库](https://github.com/clovaai/aasist), [模型源码](https://raw.githubusercontent.com/clovaai/aasist/main/models/AASIST.py) | 作者代码明确返回last_hidden与output。 |
| S02 | [SSL Anti-spoofing官方仓库](https://github.com/TakHemlata/SSL_Anti-spoofing), [源码](https://raw.githubusercontent.com/TakHemlata/SSL_Anti-spoofing/main/model.py) | last_hidden在分类层前存在；共享AASIST后端；注意device/dtype切换时内部train()行为。 |
| S03 | [ASVspoof2021官方基线](https://github.com/asvspoof-challenge/2021), [RawNet2源码](https://raw.githubusercontent.com/asvspoof-challenge/2021/main/LA/Baseline-RawNet2/model.py) | fc1_gru后的整句表征与分类输出；以实际选用DF/LA版本和checkpoint为准。 |
| S04 | [ASVspoof2019](https://www.asvspoof.org/index2019.html), [ASVspoof2021](https://www.asvspoof.org/index2021.html) | 官方数据、协议、keys与2019/2021到VCTK/VCC的映射。 |
| S05 | [In-the-Wild作者入口](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | 真实传播来源的独立外测。 |
| S06 | [ASVspoof5官方代码](https://github.com/asvspoof-challenge/asvspoof5), [官方数据记录](https://zenodo.org/records/14498691) | 现代Track1扩展；Zenodo本次直接open遇到429，官方索引仍可核实入口，下载另行核验。 |
| S07 | [RawBoost原论文](https://arxiv.org/abs/2111.04433), [作者实现](https://github.com/TakHemlata/RawBoost-antispoofing) | 说明数据增强已有基础；不能把常规噪声/滤波当新颖性。 |
| S08 | [RAPTOR公开稿](https://arxiv.org/abs/2603.06164), [HTML v1](https://arxiv.org/html/2603.06164v1) | 使用融合并讨论TTA扰动不确定性；本项目不主张首先使用测试时探针。 |
| S09 | [Adaptive reverse perturbation network for audio deepfake detection](https://www.sciencedirect.com/science/article/pii/S0925231225021381) | Neurocomputing 658, 2025, 131466；已有扰动/差异分析。此次核验摘要，不推断未读正文细节。 |
| S10 | [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) | 按关系类型传播及参数共享的技术基础，不是本项目发明。 |
| S11 | [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html), [作者代码](https://github.com/juho-lee/set_transformer) | 同信息无本体强对照。 |
| S12 | [PROV-O](https://www.w3.org/TR/prov-o/), [SHACL](https://www.w3.org/TR/shacl/) | 来源本体和结构校验，不能替代统计训练。 |
| S13 | [OpenAI官方Codex工作流](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex), [AGENTS.md说明](https://developers.openai.com/codex/guides/agents-md) | 项目指令、阶段计划和验收组织。 |

## 查新仍须完成
投稿前重点查：multi-expert transformation-response classification、audio test-time augmentation feature fusion、typed relational multi-view fusion、semantic augmentation relations。比较同输入、同预算与关系消融。当前方法是合理的候选研究路线，不宣称首创所有组件或必然可发表。
