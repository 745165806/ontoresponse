# Codex主任务：把OntoResponse实施为可复现实验

## 0. 任务
读取AGENTS.md、docs/METHOD.md、docs/EXPERIMENT_PROTOCOL.md、docs/ACCEPTANCE.md。
**主目标是一条录音一个真假logit**。使用原始embedding/score及已知探针响应，按三类证据关系计算并直接BCE训练。不要继续实现OntoRisk的PSD风险矩阵或QP作为主方法。
reference/只实现张量级参考，不能宣称已有真实音频管线。请实际编写并测试所需模块，分阶段提交STATUS与可运行命令。

## 1. 推荐项目结构
```
src/ontoresponse/
  cli.py  schemas.py
  data/{asv2019,asv2021df,inthewild,asv5,split,audit,audio}.py
  experts/{base,aasist,ssl_aasist,rawnet2,registry}.py
  transforms/{environment,probes,filtering,noise,codec,alignment}.py
  evidence/{extract,cache,merge,normalizer,response}.py
  ontology/{registry,compile,rdf,validate}.py
  models/{tokenizer,typed_relation,untyped_relation,mlp,set_attention}.py
  train/{fit_expert,fit_fusion,calibrate,select_threshold}.py
  evaluation/{official,metrics,bootstrap,ablation,report}.py
```
优先纯PyTorch；固定21节点不需要复杂图数据库。

## P0 环境、来源与研究协议
实际检测GPU/CPU/RAM/磁盘/依赖/二进制可用性；记录repo commit、checkpoint sha、许可证、上游训练和选权重来源。
不升级系统驱动、不终止其他用户进程。路径配置化，未知填null。
分开integration_pretrained和research_holdout；按协议审计源组、说话人和处理DAG。旧fairseq用隔离环境和文件接口。
锁定真正合法的未知环境条件，不把探针已经见过的算子/组合当未知。
验收：audit.json含资源、缺失、划分与来源报告；blocked明确。

## P1 真实专家接口
```
predict_utterances(audio_batch_without_labels) -> {
 ids, embeddings, spoof_margins, valid, metadata
}
```
AASIST取实际last_hidden；SSL+AASIST用out_layer输入hook；RawNet2取fc1_gru输出。检查上游class mapping，margin统一为logit_spoof-logit_bonafide（logprob差同样可用）。不在test上寻找方向。
hook不能改变原logits。基础SSL预训练+随机鉴伪头不算专家。权重缺失必须BLOCKED，不能返回随机值。
所有专家eval、禁用随机增强；审计SSL内部device/dtype移动导致train()的分支，必要时做有上游一致性测试的修复。
同专家before/after采用相同确定性crop及整句聚合。推断不接label、attack、source或env真值。
验收：至少少量真实音频上原实现/适配器logits在明确容差内一致，hidden维数由真实模型确认。

## P2 探针与观测缓存
实现configs/probes.yaml。filter/noise先行；codec必须调用实际支持的编码器roundtrip，缺失不能静默替代。
同一当前录音同家族的噪声实例使用共享基础噪声方向，随SNR缩放；同一probe artifact供所有专家共同读取。seed不依赖标签。
处理完整音频后对齐到当前输入时间轴，再使用固定crop；保留长度、时延和重编码记录。
探针不需要历史干净录音。原始输入和probe输出都保存embedding/margin，不只保存最终概率。
缓存key包括audio、checkpoint、代码、操作、种子、crop与aggregation、schema hash。原子写入，断点续跑，严查重复/缺失ID。
验收：identity与lossless roundtrip响应近零；有效零变化与missing不同；双卡/多分片同一ID无漂移；不同标签不影响seed或特征生成。

## P3 普通融合先行
从同一缓存实现score LR/MLP、原始embedding+score MLP、TTA统计、全视图MLP、显式response MLP。
每专家normalizer只拟合fusion_train，原始和probe用同一transform。质量也train-only。
原始node与response node使用METHOD公式。不同专家投影到共同维数，不直接比较未经适配的embedding坐标。
真假标签只进入BCE；每origin环境版本总权重统一。超参在fusion_val选取，calibration独立。
验收：真实小规模融合可训练/恢复/保存；无probe和有probe结果可比较；不以单次小样本提升当结论。

## P4 本体和主模型
建立完整记录/操作/运行/观测/响应本体。使用PROV-O来源词；SHACL+Python检查before/after同专家、同当前录音派生链、相同crop和score语义。
registry是本体与数值拓扑的共同来源，不能另写互不相关的RDF图。
采用reference/core.py定义的原始+响应节点和四种有向通道（三种语义），2层关系消息传递，原始/响应分别池化后输出logit。
主loss=BCE，不实现误差矩阵为主，不限制输出为原专家概率凸组合。
不要硬规定某个分数变化支持fake或real；仅约束观测对应关系。
验收：更换关系成员会改变真正的边集合；纯重命名family不改变分组；mask节点不会经bias进入聚合；无证据返回review/p=null。

## P5 强对照和结构消融
实现同信息Set Transformer或全连接attention、同邻接无类型模型、参数规模匹配和更强容量基线。
保留probe参数、family、expert ID和mask给所有强基线。
实现删各类边、无响应、无显式delta、随机family成员（保持组大小/节点度）、单专家+探针、缺失probe。
错误关系固定node内容，重新训练；只重命名family不是有效消融。错误关系只用于受控ablation，不能通过正常语义校验后伪装为真实图。
所有方法共享缓存、源权重、训练环境、选择数据和调参预算。

## P6 评价与外部研究
主指标整句未知环境macro EER；同时官方数据结果、no-new-processing、固定dev阈值实际FPR/TPR、AUROC/AUPRC、校准指标及计算成本。
先选择模型再校准/阈值，test锁定。ASV2021DF官方全测与已知源重叠剔除子集均报告，未知来源不能默认独立。ASV5是后续确认，不自动下载全量。
每条请求必须有status，失败不静默删除。配对bootstrap按source/speaker组；三个训练seed起步。错误关系提升相当或无增益均保留，不修改主问题凑阳性。
验收：独立脚本可从per_utterance_predictions重算表格，含实际样本数、失败、版本、选择规则和置信区间。

## 3. 硬件实施
检测到四卡时，冻结专家提取采用互斥utterance分片；同一录音全部环境/probe随同source分片，避免重复。可按专家顺序遍历缓存，单卡只装当前专家。
小融合模型通常单卡/CPU即可；只有重训大专家才用DDP。
三专家×(1+六探针)=21次expert-view/输入，crop数与环境版本另乘。存储按实际Σd_i估算，先profile再全量。
默认smoke≤64源录音与20step，pilot≤1024源录音；正式下载/训练需配置显式full预算。

## 4. 历史目标CLI（原始任务规格，非当前能力清单）

本节及“当前只有reference.demo”的描述保留为最初任务背景。当前已实现入口以实际`python -m ontoresponse --help`、README及docs/CLI_CURRENT.md为准；不将旧文字当作当前运行状态。
```
python -m ontoresponse doctor --config ...
python -m ontoresponse prepare-data --config ...
python -m ontoresponse audit --config ...
python -m ontoresponse extract --config ... --max-origins 64
python -m ontoresponse train --config ... --model response_mlp
python -m ontoresponse train --config ... --model typed_response
python -m ontoresponse calibrate --config ...
python -m ontoresponse evaluate --config ...
python -m ontoresponse ablate --config ...
python -m ontoresponse infer --audio ... --model ...
```
不要在README把未实现命令标成可运行。当前只有python -m reference.demo与pytest是已实现入口。

## 5. 最后交付
代码、配置、数据/权重来源锁、运行环境、split账本、测试、真实checkpoint验证日志、合成smoke与真实结果分开的报告。
STATUS分别列implemented/unit_tested/real_checkpoint_tested/real_data_evaluated/blocked。禁止把参考模型合成loss下降解释为研究成功。
