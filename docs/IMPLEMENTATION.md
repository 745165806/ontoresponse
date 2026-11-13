# 实施与复现

当前操作以[运行实验](RUN_EXPERIMENTS.md)和[CLI](CLI_CURRENT.md)为准。此前完整实施说明存于[历史文档](history/IMPLEMENTATION_20261112.md)。2026-11-13起，取消文件/源码SHA审计、文档审核和质量封存前置步骤。

整句任务、真实任务专家、统一crop/eval/聚合、fusion_train标准化、诊断响应和类型化关系没有改变。标签只进入训练损失和评价，同源及派生录音同组；test不参与模型、阈值、校准或探针选择。

训练保存配置、checkpoint、曲线和验证指标；校准只使用calibration，阈值只使用fusion_val。evaluate直接使用选定模型和校准设置，保存逐条预测与指标，不需要额外freeze。训练/推断遵循runtime.device，默认cuda:0。

缓存使用原子NPZ、真实观测和有效mask。缓存名称内部需要稳定标识，但不扫描文件/源码摘要，不保存数组完整性hash。输入文件与权重采用路径/大小/修改时间区分缓存；实现变化用新缓存或cache_revision。既有缓存仍可读取。改变缓存格式前先运行回归测试。

本体的主要作用是同一registry生成数值关系；实际边集合、before/after专家、probe artifact、分组及mask仍检查。RDF/SHACL可选导出保留，不是普通训练的依赖门。

新full融合训练仍未执行。现有2019来源重叠、RawNet2来源未知和旧pilot独立组不足等研究限制不会因移除技术审计而消失。数据方案见[FUSION_DATA_RECOMMENDATION](FUSION_DATA_RECOMMENDATION_20261111.md)。
