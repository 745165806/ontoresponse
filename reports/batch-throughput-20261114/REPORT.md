# 实际推断batch加速 — 2026-11-14

用户要求增大GPU0/1实际推断batch，提高测试效率。已部署AASIST=32、SSL+AASIST=32、RawNet2=64，单卡2个独立提取进程；模型权重、探针、裁剪、FP32计算和开发选择不变。原有完成的观测包复用。

## 实测推断环节

在已消费的16条开发录音×7视图（112个无标签输入）上测量，不使用测试EER选择配置。以下是单个专家推断吞吐，不是全流程加速承诺；codec、缓存、加载和打包另有开销。

| 专家 | batch=1 视图/秒 | 采用batch | 批量视图/秒 | 推断加速 | 峰值已分配显存 GiB |
|---|---:|---:|---:|---:|---:|
| aasist | 163.11 | 32 | 238.93 | 1.46× | 5.73 |
| ssl_aasist | 55.49 | 32 | 97.02 | 1.75× | 4.00 |
| rawnet2 | 162.45 | 64 | 1217.76 | 7.50× | 1.00 |

## 正确性与数值范围

此前将batch1逐元素1e-4作为工程门槛，原生模型的batch形状会产生浮点差异（部分graph pooling放大差异），因此该门槛不能用于声称批量适配实现错误。本轮改为核对同batch原生模型，并明确保留相对batch1的误差。没有为了加速改变权重或裁剪，也不宣称旧/新batch逐元素相同。

- 三专家的批量适配输出与相同batch的原始上游hidden/logit逐元素一致，实测最大误差均0。
- 16条开发录音上的本体、全视图MLP、响应MLP、共享关系模型，各3seed共12模型，192个预测决策均未变化；这是数值稳定性检查，不是独立鉴伪效果结论。
- 与原batch1相比，本体logit最大绝对差异0.007047，响应MLP最大0.124100；不能据此保证测试EER完全不变。详细差异见profile.json。
- 实际音频→探针→批量专家→缓存：16条开发录音、336个观测全部有效，0失败，关系/视图对应及读取验证通过。
- 首批4个256条正式测试块已完成批量提取和打包，均256/256原始观测齐全，所有probe有效；当时每块总耗时约130秒，含既有波形缓存命中，不能将它直接当作全量冷启动吞吐。

## 记录

gpu-profile.command.json/log保存原始吞吐测量及初次后处理因缓存路径错误的退出1；路径修复后的native-batch-validation.command.json/log退出0。profile.json保留所有batch/TF32试验及误差；selected-policy.json固定当前执行配置，不按测试分数选择。batching-tests.command.json/log为18 passed、0 skipped；full-regression-final.command.json/log为184 passed、0 skipped、退出0。初次完整回归的唯一失败是旧API签名测试未包含新增的keyword-only batch_size；已更新该无标签接口契约并通过完整回归。real-batched-extraction.command.json/log及real-extraction-validation.json记录实际GPU与音频链路。

当前完整测试的命令、PID和执行batch见../round1-batch-20261113/launch.json；批量缓存的inference_policy与分块config.yaml均记录实际设置。既有batch1块保留，新生成块用该批量配置，所有融合方法仍共用相同观测；不使用随机分数替代任何专家。

近期冷分块命令日志（旧100–111，新132–141）：旧每块中位数261.54秒（6进程），新99.98秒（4进程）；按并发折算的近似吞吐提升1.74倍。样本不同，包含实际I/O/codec/加载且不保证未来全程同倍率；见end-to-end-timing.json。
