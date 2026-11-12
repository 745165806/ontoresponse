# 精简与兼容性验证 — 2026-11-12

本轮整理代码与文件，保留整句真假判别目标和实验身份。没有新训练、全量提取或新的确认实验。

## 实际修改

- 常用脚本 **33→21**：12个写死历史阶段/安装/发布目录的一次性脚本移出常用目录，原文可从精简前快照恢复。清单与原hash在`removed.json`。
- 删除可再生成的build、egg-info、pytest/ruff和字节码缓存，更新忽略规则。打包验证重新生成的临时文件也已清理。
- 清理49项未使用导入，统一90个Python文件的导入/排版规则；85个文件的可执行AST在忽略导入和文档字符串后相同。
- 剩余5个有明确结构修改的文件：CLI改用普通json导入；重复EER合并；两个数据准备入口封装main；测试清理无用赋值。EER/配对bootstrap共15项旧/新结果比较完全相同。
- STATUS从204行缩为当前状态，STATUS_NEXT保留兼容指针，PLANS只维护当前任务；旧原文逐字节保存在docs/history。
- 新增架构图、模块职责和常用脚本索引；工作区配置与wheel资源副本保留，因为它们是既有打包/配置接口要求。

## 验证结果

| 层级 | 结果 | 证据 |
|---|---|---|
| unit/synthetic | **171 passed，0 skipped** | unit-final.log / unit-final.command.json |
| 风格与导入 | lint和format检查退出0 | lint.log / format.log |
| 数学实现 | 9个EER fixture和6个源组bootstrap旧/新对比完全相同 | source-parity.json |
| 真实融合checkpoint | 24条已消费fusion_val，CPU整理前后逐bit相同 | gpu-parity.json |
| 真实专家/GPU | AASIST、SSL+AASIST、RawNet2各8条已消费录音；GPU0 A6000；最大差3.33786011e-06 | gpu-parity.log / gpu-parity.json |
| 融合/GPU | 同24条，CPU基准与GPU最大差3.81469727e-06；容差1e-4 | gpu-parity.json |
| 仓库外安装 | wheel安装到独立/tmp目录；源码hash/四项资源相同，旧checkpoint输出精确一致 | installed-verification.json |
| 来源与冻结证据 | **5792个原有文件大小/SHA256全部不变**，测量/预算配置未改 | integrity.json |
| CLI和文档 | 主CLI帮助与旧版完全一致，两个准备入口导入无副作用，活动文档链接有效 | integrity.json |

这些GPU样本仅用于代码兼容性检查，不是新的EER或泛化证据。既有完整EER报告未重写。真实控制、合成测试、安装测试分层记录。

## 保留与恢复

精简前源码hash：`7b79db93c0213f8d3b5181eb368f5261ceb71df2d1aabbb32d63223079ae8e1b`。

整理后源码hash：`5199b439a2da4b575be7d02f8d1de3dc0713920d7892a156397a609949b66941`。

原始源码/配置/测试/文档快照为`source-before.tar.gz`，其hash和原受保护文件清单在`before.json`。核验脚本副本在`verification/`。历史实验的旧源码hash、模型、缓存、预注册及质量锁均保留，不能把它们改写成整理后的hash来绕过守卫；继续旧实验应使用对应快照，新实验应使用新目录并重新审计。

初次`python -m build`退出1：环境未安装build工具，原build目录曾被误识别成同名namespace。已使用现有pip的离线`wheel --no-build-isolation --no-deps --no-index`完成构建，未安装/升级环境依赖；失败日志保留。初次导入排序和两个无用局部赋值提示也已修正，最终检查均通过。

本轮是本地整理，未改写已发布GitHub提交。研究限制与当前完整专家结果仍见根目录STATUS和原benchmark报告。
