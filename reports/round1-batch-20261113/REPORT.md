# 第一轮GPU批量实验 — 2026-11-13

状态：原任务失败后已修复并恢复运行。用户已授权自主执行大批量GPU实验并恢复2019 LA eval测试。此报告记录启动状态，实时进度以同目录progress.json为准，不将运行中任务写为完成。

## 2026-11-14 2019完整结果及2021解码恢复

2019 LA eval已完成71237条、279块、36个模型报告，全部失败0；[TEST_EER.md](TEST_EER.md)给出12个方法的三个seed结果。本体融合0.348615% ± 0.125739%，响应MLP0.216023% ± 0.012425%，共享关系模型0.343384% ± 0.076784%；本体未显示优于强对照的结果，不据此重新调参。全部模型仍使用开发阶段固定设置。

2021首4块中389条被soundfile解码失败，完整性检查使任务退出。严格FFmpeg与独立SoX对这389条和64条成功对照的PCM逐样本一致；已增加仅FLAC失败时的严格解码兼容路径，检查原采样率和完整帧数，保留错误记录。16条原失败输入的336个真实GPU观测通过，完整回归192 passed、0 skipped、退出0。已保留原日志、2019结果和有效缓存，恢复GPU0/1动态队列纯测试任务；2021及ITW完整EER仍待完成。详情见[解码恢复](../flac-recovery-20261114/REPORT.md)，当前batch32/32/64、每卡2worker和动态队列配置见launch.json；下方配置为历史阶段记录。

## 2026-11-14固定当前模型测试

36/36个最终模型已保存。用户要求固定当前模型，在2019 LA eval、2021 LA eval和ITW测试；已使用`--test-only --workers-per-gpu 3`启动纯测试流程。GPU0/1各3个独立进程；测量机械盘I/O占用约91%，故临时缓存迁到NVMe SSD（/tmp/ontoresponse-round1-cache），保留缓存内容及已完成块，最终观测/预测仍存项目目录；保留原有逐视图专家推断；不进入开发或重训。已有完整观测包复用。

多视图batch前向先在已消费开发录音验证，AASIST hidden最大差异0.00111845超过预设atol/rtol=1e-4，因此未采用，专家实现已原样恢复。操作变更与拒绝原因见test-execution-change.json，不进行test-based选择或权重修改。

TEST_EER.md/json仅接受正式样本数、失败0和三个固定seed齐全的测试报告。每个完整数据集产生后自动汇总及同步小报告，尚无完整测试结果时显示空白。完整覆盖及seed汇总的回归测试已通过；逐句大文件只保留本地。

## 2026-11-14状态更新

原批量任务在10:28（Asia/Shanghai）停止，未完成：1024条pilot和50224条完整开发数据均提取完毕，两阶段各72次比较完毕；最终重训因参数为空的best_single_expert误创建AdamW而失败，三个测试集尚未开始。失败日志和原状态已保留为*-before-recovery-20261114文件。

现已修复：无参数基线只在开发组选择专家，不创建优化器，保存chosen_expert，实际优化epoch=0；神经模型训练设置保持原有selection.json。补充测试覆盖无优化器、开发数据限定、checkpoint恢复与复用；10项相关测试通过、0 skipped，Ruff通过。recovery-tests-20261114.command.json和.log记录实际命令与退出0。

已在10:58（Asia/Shanghai）恢复GPU后台任务，复用全部观测和已完成模型，PID见launch.json；监控也已恢复。当前总进度以progress.json为准。原报告下方的启动描述与177项回归属于2026-11-13历史阶段。

开发验证结果已汇总在[DEVELOPMENT_EER.md](DEVELOPMENT_EER.md)，是两折三seed验证均值，不能当三测试集EER。本体0.029234%，全视图MLP0.019376%，未显示本体开发增益。

## 已完成（历史启动阶段）

- 64条真实开发录音，按两个来源组件分为32训练/32验证，各自16真16假；没有读取新测试分数。
- 三专家各448个有效观测，合计1344个，失败0；提取退出0，50.61秒（含既有缓存命中，不能据此线性估计全量吞吐）。
- original_mlp（90945参数）、response_mlp（266369）、typed_response（224385），GPU0各20步，验证EER均0，save/load精确一致；三个命令均退出0。参数量尚未匹配，且只有32条验证，不能判断方法增益。
- 既有64条真实观测打包后读取，所有hidden、score、mask、label、quality、专家身份及ID精确一致，见pack-check.json；只复用开发smoke，没有新增测试曝光。
- 177项测试通过、0 skipped，pytest退出0，35.08秒；regression.command.json、regression.log记录实际命令与日志。

smoke完整记录见../round1-smoke/{extraction,original_mlp,response_mlp,typed_response}.command.json和summary.json。

## 正在运行

启动命令：`${EXTERNAL_PATH} -u -m scripts.round1_batch --config configs/round1-batch.yaml`。PID、启动UTC及日志见launch.json。独立后台进程和文件日志在当前会话结束后继续运行。

- GPU：启动前0为276MiB/近0%负载，1为13MiB/0%负载；使用RTX A6000物理0/1，其余GPU任务未修改或终止。
- 磁盘：启动前约3.1TiB可用；预留100GiB；分块最多256条，临时波形缓存打包后删除，保留原音频与数值观测。
- 每块3600秒超时，失败保留日志和请求；没有随机分数或静默删样本。
- 先1024条pilot，两折×三seed×12对照；再完整50224条开发数据，两折选择、固定开发轮数后完整重训。
- 12对照：最佳单专家、score_mlp、original_mlp、all_views_mlp、response_mlp、set_attention、relation_tied、typed_response、wrong_family及三种删边。四种强基线采用接近typed模型的参数量；所有强基线共用观测及合法元信息。
- 最后评测2019 LA eval 71237条、2021 LA eval阶段148176条、ITW 31779条；三个测试集不用于选择。2021另外报告分codec EER。

## 输出及限制

每块command.json/extraction.log记录退出码与耗时；evidence.pt包含真实hidden/margin、mask、专家身份和对应metadata。训练产生model.pt、convergence.jsonl、result.json。selection.json记录开发选择，models-ready.json生成后才处理测试音频。tests/<dataset>/<model>-seed<seed>/保存逐句logit与EER等指标。

常规CLI的smoke/pilot限制未删除，完整批量有单独显式预算；无需SHA、源码封存或质量文档审批。断点继续会复用完整块和完整模型，中断的单个开发模型保留旧目录后从头运行，尚不支持该模型epoch中间恢复。

专家已在2019开发数据训练/选择，RawNet2来源未知，只有两个独立开发组件；结果属于集成研究，不宣称严格独立或因果解释。固定首64600采样点crop对原始/所有探针保持一致，属于既定裁剪策略。历史2019 eval专家结果及旧ITW消费已存在，外测曝光限制不会因重跑消失。
