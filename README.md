# OntoResponse

用冻结鉴伪专家的原始表征、诊断处理响应和类型化关系，输出一条录音的真假logit：**0=bonafide，1=spoof**。

当前已完成真实音频工程闭环、smoke/pilot和三个专家的完整GPU评测。新融合full训练尚未启动；现有结果不能证明本体带来额外检测增益。

## 从哪里开始

- 理解代码：[架构与数据流](docs/ARCHITECTURE.md)。
- 运行工具：[脚本入口](docs/SCRIPTS.md)和[CLI](docs/CLI_CURRENT.md)。
- 查看状态：[STATUS.md](STATUS.md)；当前计划：[PLANS.md](PLANS.md)。
- 实验设计：[METHOD](docs/METHOD.md)、[EXPERIMENT_PROTOCOL](docs/EXPERIMENT_PROTOCOL.md)、[ACCEPTANCE](docs/ACCEPTANCE.md)。

```text
ontoresponse/   音频、专家、证据缓存、本体、融合训练与评价
reference/      生产模型复用的数值核心与兼容checkpoint类
configs/        资产路径、探针、本体和明确预算
scripts/        当前可用的运行、审计与复现入口
tests/          单元、合成与接口回归测试
docs/           方法、操作说明及历史文档
reports/        已记录的命令、分数、控制和阶段报告
artifacts/      真实观测缓存
deliverables/   历史完整包与去重后的差异快照
```

## 运行

优先使用现有py310环境，避免升级已核验的fairseq/专家依赖。机器路径见`configs/local.yaml`；入口与参数可安全查看：

```bash
${USER_HOME}/anaconda3/envs/py310/bin/python -m ontoresponse --help
${USER_HOME}/anaconda3/envs/py310/bin/python -m scripts.expert_benchmark --help
```

历史smoke/pilot已有冻结输出；新实验先复制配置并指定新的output，按[实施说明](docs/IMPLEMENTATION.md)检查来源和预算。完整专家评测使用明确的GPU预算；四卡提取/合并一致性尚未验证。

## 已完成的完整专家评测

ASVspoof2019 LA eval：71237条录音，7355 bonafide、63882 spoof；每个权重失败0。下表采用作者原生score与官方EER计算口径。

| 权重 | EER (%) |
|---|---:|
| 现用 AASIST | 2.121052 |
| 现用 SSL+AASIST | 0.637284 |
| 现用 RawNet2 | 5.084675 |
| 作者发布 AASIST 参考 | 0.829511 |

现用AASIST未达到发布权重水平。SSL论文0.82%对应2021 LA，RawNet2本地配置/来源也不同，不能宣称三个现用权重都已复现论文。详见[完整报告](reports/expert-benchmark-20261111/RESULTS.md)。

后续数据建议见[融合数据方案](docs/FUSION_DATA_RECOMMENDATION_20261111.md)：近期可用Codecfake中文来源，但必须按speaker/source重分组并统一采样率；ASV5为后续现代路线，本地尚无其音频。

## 开发与历史记录

```bash
python -m ruff check ontoresponse reference scripts tests
python -m ruff format --check ontoresponse reference scripts tests
python -m pytest -q
```

完整测试还需要`pyproject.toml`中的`test`可选依赖；不能将缺依赖或skip算作通过。测试使用合成fixture，真实checkpoint/数据/GPU验证另行记录。

2026-11-12整理前的状态、计划与一次性脚本恢复说明在[docs/history](docs/history/README.md)。原实验输出和冻结证据保留，生成的构建文件与字节码缓存已清理。GitHub仓库：[745165806/ontoresponse](https://github.com/745165806/ontoresponse)。

公开上传默认每文件不超过1MiB，包含当前源码与小型报告；完整benchmark逐条分数、音频、模型、缓存与归档留在本地。旧576条公开融合预测可独立重算，新benchmark公开汇总不包含完整分数。发布范围见[PUBLICATION.md](PUBLICATION.md)。
