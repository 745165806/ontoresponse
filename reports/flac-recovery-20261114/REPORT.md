# 2021 LA解码兼容与测试恢复 — 2026-11-14

2019 LA eval已完成71237条和全部36个模型评分，失败0，完整EER见[测试表](../round1-batch-20261113/TEST_EER.md)。随后2021首4个256条分块中，389条在soundfile读取阶段发生FLAC decoder/seek异常，严格完整观测检查使任务退出；没有删除这些请求或将其当作有效分数。

## 修改与验证

音频读取正常时保持原soundfile路径；只有FLAC读取异常才调用现有FFmpeg，使用`-xerror -err_detect explode`，不设置输出采样率或声道，不重采样。拒绝解码错误、非16000Hz或与流元数据不一致的完整帧数；仍按原规则平均声道、检查短音频/静音/有限值。提取记录新增`audio_decode`，记录实际reader、原声道/帧数及初次解码错误。模型、探针、裁剪、开发选择和所有已完成结果保持固定。

- unit/synthetic：新增5项检查，验证正常路径不调用FFmpeg、兼容路径PCM一致、错误采样率拒绝、截断帧及解码错误拒绝。完整回归192 passed、0 skipped、退出0，命令`python -m pytest -q`；见full-regression.command.json/.log。
- real-data解码：全部389条原失败录音及64条成功对照，共453条，严格FFmpeg/正常soundfile输出与独立`sox INPUT -t f32 -L -`逐样本一致；完整帧数均匹配，最大PCM绝对差异0。见real-decoder-validation.json；逐条记录及初次失败摘要保留本地，不上传音频。
- real-checkpoint、real-data、GPU：在cuda:0对16条原失败输入执行`python -m ontoresponse extract --config reports/flac-recovery-20261114/real-extraction/config.yaml --max-origins 16`，三专家×七视图共336观测全部有效，失败0；关系对应、缓存和EvidenceBatch读取通过。见real-extraction.command.json/.log及real-extraction-validation.json。初次后处理脚本误用不存在的mask属性，已改用valid并重新校验已保存真实观测，未重新运行专家或改变输出。

## 执行与限制

GPU0/1各2个worker，动态队列、专家batch32/32/64；临时缓存使用本地NVMe SSD，原始音频、最终观测和预测在项目磁盘，保留100GiB余量。已重新启动纯测试调度器和EER观察进程，命令/PID/UTC见relaunch.json及../round1-batch-20261113/launch.json；复用2019的279个观测包、36份完整报告和2021有效部分缓存。后台任务仍运行中，不将启动视为完成。GPU2/3其他任务未改动。

恢复后原先失败的首4块已全部完成打包：1024条、21504项专家视图观测均有效且有限，失败0，389条使用兼容解码；提取命令均退出0，cuda:0/cuda:1均参与。见resumed-parts-validation.json。后续分块继续运行，此记录不是完整2021效果结果。

2021剩余录音和ITW尚未全量验证；若出现实际损坏，严格解码仍将失败并保留请求/日志，不以不完整数据冒充完整EER。此修复以原PCM解码结果为依据，不使用测试EER选择算法或参数。公共专家2019开发曝光、RawNet2监督来源未知和开发来源组件数量较少的限制仍保留。
