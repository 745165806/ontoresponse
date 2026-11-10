# 方法规格：直接判别的本体关系融合

## 1. 输入及两个处理层
一条待测完整录音x，标签y∈{0,1}。系统可能收到x=T_env(x0)，但测试时只能看到x，不能得到x0或T_env真值。
系统主动执行已知探针P_k，产生x_k=P_k(x)，k=1..K；x_0在后文token公式里记为当前观测x，而非历史干净录音。为避免符号混淆，代码中使用view_index=0表示当前未探针输入。
探针及其参数可作为特征；历史环境、attack、dataset、speaker、path仅作审计。

## 2. 专家输出
冻结专家f_i；每个视图输出整句h_ik∈R^{d_i}与spoof方向margin s_ik。输入接口不接标签。
多个固定crop必须相同起点/同一聚合策略用于原图与探针。建议对完整音频先处理，再按固定当前输入时间轴取crop；codec延迟校正要记录，不能按标签挑对齐。
AASIST直接返回last_hidden/output；SSL+AASIST取out_layer输入；RawNet2取fc1_gru输出。不得把无鉴伪训练的基础SSL加随机头当专家。

## 3. 缩放与响应
逐专家mu_i,sigma_i只在fusion_train原始视图、有效样本上拟合；所有探针共用同一变换：
  htilde_ik=(h_ik-mu_i)/max(sigma_i,eps)
分数同理。质量数据的缩放也只用fusion_train。
  delta_h_ik=htilde_ik-htilde_i0
  delta_s_ik=stilde_ik-stilde_i0
没有独立标准化before/after，否则差分含义改变。

原始token输入：
  a_i = MLP_i_original([htilde_i0, stilde_i0])
响应token输入：
  r_ik=MLP_i_response([delta_h_ik, stilde_i0, stilde_ik, delta_s_ik,
                         log1p(||delta_h_ik||_2), known_probe_features_k])

保留两个绝对分数是明确选择；差分是二者的确定性函数，不能声称它增加信息。强MLP同样应拿到绝对值与差分。原始h + delta_h保留探针h的信息；baseline也应有相同view h与参数。
不同专家各用适配器映射到d=64起点。同专家差分前坐标一致；不直接对不同模型原始h做余弦相似度。
加入可学习expert ID和node-kind embedding；操作参数输入包括真实家族类别及按物理单位固定缩放的数值（不是测试域标签）。

## 4. 证据图：M+MK个数值节点
M=3,K=6时共21节点：3个原始a_i，18个响应r_ik。操作、录音、运行等本体对象作为元数据，可不作为神经节点。
节点次序：a_0..a_(M-1)，然后r_00..r_0(K-1),r_10..。

三种语义/四种有向实现关系：
R0 anchor_to_response：a_i→r_ik；
R1 response_to_anchor：r_ik→a_i；与R0是同一“before/after来源对应”的两种传播方向；
R2 same_probe_cross_expert：r_ik↔r_jk，i≠j；必须指同一个实际probe artifact，不只是同类操作；
R3 same_family_within_expert：r_ik↔r_il，k≠l且family(k)=family(l)。

前后对应是可验证事实，不是“二者一致则为真”。共享架构不是主要边类型，作为登记与后续消融即可。

## 5. 两层关系计算
  m_v = W_self z_v + Σ_r W_r mean_{u∈N_r(v),valid(u)} z_u
  z'_v = mask_v * LayerNorm(z_v + Dropout(GELU(m_v)))
无邻居的聚合为0，缺失节点不得通过bias、LayerNorm或另一条边重新注入值。
两层提供有限关系组合，不默认叠更多层。不需要大型图数据库或PyG。
RDF关系与数值A[r,target,source]必须由同一registry编译、版本和hash一致。

readout：分别masked-mean汇总原始节点和响应节点，拼接实测质量与覆盖比例，MLP输出任意实值logit l(x)。
  p_uncalibrated=sigmoid(l)
最终概率可独立温度/Platt校准。主预测不限制在各专家原始概率范围内。

## 6. 训练目标
主目标只有源样本归一的BCEWithLogits：
  L = Σ_g mean_{t∈versions(g)} BCE(l(x_g,t),y_g) / number_of_origins
其中versions是某条目标录音的环境处理版本，不是原始录音与其所有真假衍生标签混成同一个标签。source_group用于分割/统计，origin_id用于版本权重，两者区分。
实际可每epoch每origin抽一个环境版达到等权。类采样/损失权重/先验必须记录，最终校准不能宣称任意业务后验。
固定专家不反传；适配器、关系层和分类头联合训练。统一训练环境分配给所有基线。
不强迫所有视图embedding/分数相同；处理可能破坏线索。来源标签保持不代表可观察证据必然充分。

## 7. 本体和语义范围
类：Recording、DetectorModel、ExpertRun、ExpertObservation、ProbeOperation、ProbeRun、ResponseObservation、Decision。
ProbeOperation子类：BandwidthProbe、NoiseProbe、LossyCodingProbe。
关系：generatedBy、usesModel、usesProbe、hasBefore、hasAfter、respondsTo、derivedFrom。

约束示例：
- before与after来自相同专家版本；
- after输入应派生自before对应当前录音，probe run和参数可追踪；
- embedding维数、score方向与聚合策略兼容；
- 同探针跨模型关系指同一音频artifact；
- 源复制不成为新增证据；空值不是零响应。

本体是测量与操作的形式模式；神经网络学习真假作用，不把OWL推理当概率计算。当前图拓扑主要是静态的，因此主张归纳偏置而非新信息。是否有用由同信息强对照验证。

## 8. 消融及公平性
必须共享相同原始缓存、train-only scaler、家族/操作参数、专家身份与mask。
- 去响应：只原始a_i。
- 去显式差分：只原始和probe绝对观测；不能宣称差分本身增加信息。
- 同邻接无类型：所有边用共享W。
- 全连接无类型/Set Transformer：不依赖手工对应的强对照。
- 随机成员：固定真实token/参数，改变same-family实际成员，保持每组大小/节点度；重新训练。
仅把family名称0/1/2重命名不改变同类关系，是无效随机对照。
固定节点后改变成员，哪怕抽象无属性图同构，也已经改变了证据对应；不需伪称图论上必然非同构。

## 9. 缺失与解释
响应必须同时有before/after。全部原始专家无有效观测则返回review、p=null。部分探针缺失保留mask与覆盖；主结果不能以拒判掩盖错误，必须报告全请求数量与失败数。
解释仅为输入输出溯源、关系消融、证据删除后的输出变化。删除测试是模型敏感性，不证明现实因果。

## 10. 推断不可泄漏的内容
推断forward仅接张量和固定registry，不接labels、speaker_id、attack、dataset、environment_truth或路径。
已知probe参数是合法条件；元数据seed不得依赖标签/attack。
模型eval()和禁用随机频域增强；特别审查SSL代码中device/dtype迁移会调用train()的路径。
