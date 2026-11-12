# 历史实施说明

本文保留2026-11-10准备阶段的复现资料；当前能力、GPU和来源映射状态统一见[STATUS.md](../STATUS.md)，常用入口见[SCRIPTS.md](SCRIPTS.md)。

# 2026-11-10 审计与新 pilot 入口

## 文档到位后的最终状态

`CODEX_NEXT_STAGE.md`已读取并锁hash，不再缺失。当前交付与证据见根目录 `STATUS_NEXT.md`；以下原准备阶段说明保留为历史，不能覆盖这一更新。

可分享审计包：`reports/codex-next-stage-20261110/audit_bundle.zip`，标准库入口为解压目录的 `python3 audit_recompute.py`，576条逐请求预测。ZIP的SHA256针对实际脱敏交付字节；private_source_sha256仅用来关联本地原文件，不能拿来校验脱敏文件。音频、权重、数组缓存、私有tar与凭据均排除，所有实际成员严格对应AUDIT_MANIFEST。

已完成123项完整测试、真实两专家批量/重复/ID隔离控制、16项codec控制、SHACL完整图及负例、最新wheel仓库外安装和真实缓存前向。实际环境/依赖锁、命令退出码与日志在新报告目录；旧fairseq及系统驱动未升级。

显式128-origin两专家pilot执行完成：6种普通配置×3seed，15个可训练配置实际160–400优化步，保存JSONL/PNG收敛曲线；3个单专家配置不训练融合参数。原始与响应融合验证EER均0，不支持预注册H1；按规则停止H2，因此本轮没有typed/relation_tied等关系模型的真实效果结果。模型/校准/阈值先全部冻结，然后432条新测试预测全部成功。新测试组极少且天花板，仅为integration_pretrained；官方source映射与task HPO来源UNKNOWN使research_holdout仍BLOCKED。

`formal_protocol.yaml`仅为供审阅的方案，不启动full。旧8条外测与新24条测试origin进入消费账本。GitHub仅准备脱敏离线Git bundle，网络/认证/实际仓库尚未就绪，未上传。

---

## 文档到位前的准备阶段历史

沿用OntoResponse v0.5：整句embedding/margin + 已知探针响应 + 类型关系 → 整句logit。旧冻结目录与已消费外测不可用于新选模/阈值/校准。

## 实际证据与独立导出

原`deliverables/ontoresponse-v0.5-baseline-20261110.tar.gz`已于2026-11-12去重，独有内容保存在同名`.delta.tar.gz`，公共成员复用保留的`ontoresponse-v0.5-next-stage-20261110.tar.gz`。恢复后2463个原始文件的大小/SHA256已全部验证；不声称重建gzip容器与退役容器的压缩字节完全相同。原清单与校验记录仍保留，恢复方法见[存储清理报告](../reports/storage-cleanup-20261112/REPORT.md)。

```bash
python3 reports/storage-cleanup-20261112/snapshot_delta.py restore --base deliverables/ontoresponse-v0.5-next-stage-20261110.tar.gz --source deliverables/ontoresponse-v0.5-baseline-20261110.delta.tar.gz --output /tmp/restored-baseline.tar.gz
python3 scripts/audit_bundle.py verify /tmp/restored-baseline.tar.gz
python3 scripts/recompute_predictions.py --root reports/implementation --out /tmp/independent-metrics.json
```

原始大型专家权重与完整语料不在tar内重分发；具体文件的真实存在性与hash见来源审计及外部资产清单。tar中含融合checkpoint、已有真实观测及原始训练/选模清单副本，支持独立重算。曾经运行失败的日志仍保留。

## relation_tied

`RelationTiedFusion`复用typed构造的全部节点、邻接、两层计算、逐关系mask归一化、self-map、残差、GELU/Dropout/LayerNorm和池化/分类头。唯一更改是每层4个关系Linear引用同一个Parameter；同seed下非关系参数初值也一致。

`copy_tied_to_typed`将共享矩阵复制到typed的4个独立矩阵；反向复制要求4个矩阵先相等，否则拒绝，避免state_dict共享别名静默采用最后一个矩阵。测试覆盖前向double等价、缺失/全缺失、共享梯度等于4个独立梯度之和、参数量差、保存加载及同seed初值。旧`untyped_response`是union邻接再归一化，数学上不同，继续保留且不把它当relation_tied。

## 来源审计与消费集

`scripts/audit_supervision.py`实际读取checkpoint、80个epoch的history、finalized选权重记录及原始fit/source_val清单，并把原始证据复制到`reports/next-stage-20261110/upstream-evidence/`。AASIST/SSL分别选epoch66/7，与best.pt实际hash一致；选法为source_val EER最小且最早epoch。另有中断failure.json，原样保留，不把它说成“早停”。训练代码没有patience早停，history达80epoch。

旧融合val/calibration各2条ID参与过专家选模；说话人层面影响更大。因此新清单排除全部上游训练/选模speaker，而非只删除4条ID。RawNet2只有state_dict和当前训练脚本，不能据此推断这个权重的训练/选模历史，已从新pilot配置排除。

当前8条In-the-Wild及其所有派生版本登记为`consumed_exploration`，仅允许保留与审计，禁止重新校准、调阈值或新选模。原development样本也不作为新pilot test speaker组。新划分保留origin/speaker组和waveform跨角色去重检查。

## 独立预测重算与阈值空间

`scripts/recompute_predictions.py`仅用stdlib重新计算6个已冻结run的整句EER（按score ties分组、线性ROC交点）、固定规则FPR/TPR、NLL、Brier及10-bin ECE，同时检查角色、checkpoint/校准锁、预测覆盖和概率变换。没有导入项目的metric或拟合函数，没有改变原结果。

旧普通基线阈值为p=1.0，float32概率在若干有限logit处已饱和。旧FPR/TPR依保存的`p>=threshold`仍可重算，但它不是有限的raw-logit阈值。新pilot规则预先固定为float64 calibrated-logit空间比较，记录space、comparison及角色；概率只用于概率指标。这是数值实现修正，不用旧8条外测重新选择阈值。

## 依赖、本体和仓库外安装

利用已有纯Python依赖目录创建隔离层，不修改系统/现有py310。`dependency-lock.json`记录每个依赖的版本、Requires-Python、RECORD逐文件hash校验、实际运行环境版本及离线依赖tar SHA256。原下载尝试失败仍明报；本地层来源未被伪称为本次下载的官方wheel。

完整测试包含SHACL正例、错误专家/录音/crop/score/probe-run/缺失属性/错误跨专家probe等负例；Python进一步核对registry编译的全部真实数值边。实际24条旧证据图也执行SHACL。RDF/SHACL现在有真实通过证据，不再沿用旧的“只解析语法”。

wheel显式包含探针、本体配置和SHACL资源；资源定位不依赖cwd。隔离venv在`/tmp`安装wheel，从仓库外加载registry、运行旧真实缓存的融合前向、relation_tied及SHACL验证；输入/模型均不重新拟合。路径/命令/退出码见outside-install JSON与command journal。

## 新 pilot 预算与启动门槛

`configs/pilot-20261110.yaml`明确：128 origins（64 fusion_train /24 fusion_val /16 calibration /24 test），2个来源可审计专家，6模型×3seed，每模型batch16、最多100epoch/400优化步、至少40epoch后才允许15epoch patience停止；每family仅1组固定超参。保存逐epoch优化loss、eval train BCE、val BCE/EER和实际step数及收敛PNG。预算耗尽与patience停止分开报告，不能声称必然收敛。

顺序：original_mlp → response_mlp → 同信息all-view/attention → relation_tied/typed。每run按fusion_val BCE选checkpoint，基线family按3seed平均val EER再BCE选；全部模型/校准/开发阈值锁定后才读取test tensors评分。新试验不重写已有run/freeze；旧训练和校准入口也加入冻结保护。

该pilot只允许标`integration_pretrained` / `integration_pilot_only`。跨speaker的完整源映射与通用SSL训练暴露未认证，不能标research_holdout。严格排除旧组后，dev新test包含3个speaker组但仅1个spoof speaker组，审计明报，不用于证明广泛泛化或让typed必然胜出。

```bash
PYTHONPATH=${EXTERNAL_PATH} ${USER_HOME}/anaconda3/envs/py310/bin/python -m scripts.run_pilot --config configs/pilot-20261110.yaml --preflight-only
# 只有preflight=PASS_INTEGRATION_PILOT且所需文档已读取并记录hash，才允许去掉--preflight-only。
```

目前用户要求读取的`CODEX_NEXT_STAGE.md`未找到，也未得到其内容，故门槛不会因等待时间自动通过。准备新清单不等于已运行pilot；真实新训练与新test预测尚未启动。GPU内核元数据列出4张物理卡，但本执行沙箱无可用GPU字符设备且/dev只读/nodev；只做环境定位，不改驱动/设备节点/进程。
