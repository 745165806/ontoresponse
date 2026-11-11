# 2026-11-11 文档更新的发布范围

本次公开提交同步最新实验说明、融合数据建议与汇总证据；不改变已有发布源码、旧融合预测及其历史实验身份。

包含：README、STATUS、STATUS_NEXT、PLANS、融合数据方案；四个checkpoint的完整eval汇总及逐attack指标、GPU控制、命令/退出码、171项单元测试记录和来源/候选数据审计。

这些真实运行在研究工作区完成。执行源码和benchmark runner的SHA256分别为：

- research source: `7b79db93c0213f8d3b5181eb368f5261ceb71df2d1aabbb32d63223079ae8e1b`
- benchmark runner: `22757ba4e45bcc1915343ef8cd52083cef6bacb969800380c612953a356cf6e5`

本次未整体同步这些后续runner源码。因此171项测试是该研究执行版本的验证记录，不能据此宣称公开旧源码已包含全部新接口。

不包含：原始或探针音频、checkpoint、数组/embedding缓存、官方来源映射archive、四份完整逐条benchmark分数及详细运行日志。JSON里的checkpoint、协议、原始预测SHA256用于标识实际私有/外部输入输出；AUDIT_MANIFEST的SHA256才是本次公开文件的字节校验。

新benchmark EER已在研究工作区按官方compute_eer及独立ROC口径复核；仅凭本次汇总不能重新计算这些EER。根目录audit_recompute.py仍可用标准库重算此前公开的576条融合预测；该旧功能不覆盖新benchmark。

私人路径使用`${PROJECT_ROOT}`、`${EXPERT_ROOT}`、`${DATA_ROOT}`、`${USER_HOME}`、`${EXTERNAL_PATH}`、`${TEMP_ROOT}`模板；控制样本使用稳定假名。某些历史链接指向未包含的本地证据时，以“本地证据”标注其路径。

科学结论范围：现用AASIST未达到作者发布权重效果；SSL的2021论文数值不能当2019对照；RawNet2配置/训练来源仍需认证；新融合full训练未执行，也未宣称本体收益。
