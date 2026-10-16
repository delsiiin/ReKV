# 在线流视频 KV 缓存压缩：观察实验与方法设计

更新日期：2026-10-15。目标资源：**GPU 显存中的 KV 缓存**。本文结合当前 ReKV 工作区和公开论文提出实验协议及待验证方法；没有运行模型实验，文中的参数是起始配置，所有性能收益均待测。

## 1. 建议收敛的研究问题

将问题定义为：**未来问题未知、视频持续到达时，在固定 GPU 字节预算下，如何为历史片段各层的 K/V 选择精度，并决定何时减少帧、何时彻底删除，使未来问答损失尽可能小？**

三个设计问题应分别由观察实验回答：

| 你的问题 | 首要观察 | 方法决策 |
| --- | --- | --- |
| 如何划分连续片段，是否考虑量化敏感性？ | 语义相近是否意味着同一量化配置也合适；改变分段后，收益来自精度分配还是量化分组变化？ | 使用因果语义边界；仅在能带来额外收益时加入量化误差变化信号 |
| 如何分配精度、处理长期未激活的历史？ | 访问概率、量化误差和任务损失之间的关系；未被问过与被多次问过但无关的段落有何差别？ | 以“未来访问风险 × 降精度的条件损失 / 节省字节”决策，保留未知历史的探索余量 |
| 如何在固定预算下区分层？ | 层敏感性是否稳定，是否存在片段与层之间的交互？ | 先采用静态层分组，再检验动态片段×层分配是否值得增加开销 |

建议主方案保持一个中心机制：**用预计损失的边际增量统一比较降位宽和删帧两类操作**。分段、访问统计和层校准分别提供决策单元、未来使用风险、误差代价，避免三个彼此独立的启发式互相冲突。

优先完成 O0、O1、O3、O5，再决定是否投入复杂分段和渐进删除。若统一 INT4 已满足质量及预算要求，复杂自适应方案需要证明额外价值。

## 2. 任务边界与显存口径

### 2.1 主设置与对照设置

**主设置：GPU 常驻压缩历史。** 模型权重固定，视频按时间顺序到达；未来问题及答案不可用于压缩决策。保留的视频历史 KV、其检索索引和压缩元数据均受 GPU 预算约束。推理时不从 CPU/磁盘取回被删除的完整 KV，也不重新编码原始历史视频。近期窗口、系统前缀和当前问答的 KV 使用原始 BF16/FP16。

允许在离线观察阶段将参考轨迹保存到 CPU/磁盘，用于反事实测量；这些轨迹只供实验测量，不向在线策略提供未来信息。离线校准也不等于更新主模型权重。

**单独设置：允许卸载的系统对照。** 原生 ReKV 的 CPU 历史存储属于这一组。报告 GPU、CPU/磁盘占用与传输延迟；不能将它的可恢复历史直接用于主设置中的“精度提升”。

无限视频流、有限字节和无外部存储不能同时保证任意历史细节永久可恢复。应报告随历史年龄变化的保留率与问答质量，以及预算压力下的遗忘方式。

### 2.2 预算必须覆盖运行峰值

设层数为 \(L\)，第 \(\ell\) 层 KV 头数为 \(H^{KV}_\ell\)，头维度为 \(d_\ell\)，片段 \(s\) 保留 \(n_s\) 个 token。历史 payload 字节为：

\[
M_{payload}=\sum_{s,\ell}n_s H^{KV}_\ell d_\ell
\frac{b^K_{s\ell}+b^V_{s\ell}}{8}.
\]

预算约束应写成：

\[
M_{KV,peak}=\max_t\big[
M_{payload}+M_{recent}+M_{prefix}+M_{QA}
+M_{index}+M_{scale/offset}+M_{padding}
+M_{dequant}+M_{migration}+M_{other\ KV}
\big]\le B.
\]

其中反量化缓冲、重打包时新旧表示短暂共存、量化 scale/offset、页表和池中占用都要计入。权重、视觉编码激活及其他运行空间另报整卡峰值；\(B\) 固定后不能随已看视频长度增加。

同时记录三种值：组件实际 tensor/storage 字节、CUDA `max_memory_allocated`、`max_memory_reserved`，并辅以进程显存。固定的未使用预分配池仍占显存。按 storage 去重，避免把 view 重复算成多份，也避免切片后仍持有大底层 storage 却按逻辑 shape 少算。

**量化后重新存为 BF16 的模拟实验只能测质量。** INT2/INT4 要真实打包；问答时若完整展开历史，不能把静态 payload 的压缩倍数当成 GPU 峰值节省。

### 2.3 当前仓库给出的实现约束

本次检查基于 Git HEAD `6810a4d` 及当前工作区。现有源代码不在本次修改范围。

| 位置（相对仓库根目录） | 已观察到的行为 | 对方案的影响 |
| --- | --- | --- |
| `model/attention/kv_cache_manager.py:32`，`MemoryUnit` | 历史 KV 保存 CPU 副本，按需加载 GPU | 主设置需要 GPU packed store，不能只给 `cpu_data` 加量化 |
| 同文件 `ContextManager.init`，约 313–325 行 | `global_buffer` 与 `CudaCache` 都按原 dtype 分配 | 要替换全精度池和大展开缓冲，才能获得显存收益 |
| 同文件 `VectorTensor`、`block_k` | 帧代表向量在 GPU 上随帧数增长；容量倍增 | 索引也必须限额，并统计扩容峰值 |
| 同文件 `get_cosine_similarity`，约 172 行 | 实际执行 FP32 点积，函数内部没有归一化 | 分段用余弦时应显式归一化；检索沿用原点积时应明确区分 |
| 同文件 `_calc_block_topk`、`_append_global` | 以帧级 block 为基础，支持固定 `chunk_size` 聚合 | 变长语义段需要独立目录，不能直接用固定 chunk 算术索引 |
| `model/attention/rekv_attention.py` | 问答路径拼接历史 KV，并按 `n_local` 等参数处理窗口 | 需区分编码近期窗口与问答历史可见范围，防止新方法选中的远期 KV 被再次截断 |
| `model/attention/dot_production_attention/triton_impl.py` | 已有多阶段注意力累积；`get_score=True` 会保留 Q/K 引用供后处理 | 可作为分块读取切入点；采集注意力统计时避免长期保留反量化块 |
| `model/abstract_rekv.py:67` | 内存日志取第一层 CPU KV 字节再乘层数 | 无法衡量 GPU 峰值；异构层配置下也需逐层实算 |
| `video_qa/rekv_stream_vqa.py` | 按问题 `end_time` 编码视频前缀，进入问答 | 可添加 arrival/evidence 时间字段和策略日志；需要核对数据时间语义 |

本地 `llava-onevision-qwen2-7b-ov-hf/config.json` 显示 28 层、28 个 query heads、4 个 KV heads、hidden size 3584，即 head dim 128。当前适配器设定每帧 196 个 token。因此 BF16 双 K/V 的**理论 payload** 为：

\[
28\times4\times128\times196\times(2+2)
=11{,}239{,}424\ \text{bytes/frame}=10.71875\ \text{MiB/frame}.
\]

32 帧为 343 MiB；64 帧为 686 MiB。两者均未含元数据与临时空间。保持帧数不变时，K4V4 的 payload 是 BF16 的四分之一；这不是实测整卡收益。GQA 字节计算必须使用 4 个 KV heads，而非 28 个 query heads。

## 3. 与已有工作的关系

以下是截至本次检索可核验的相关证据。表中“本文应验证”属于研究推断，不表示既有工作从未涉及类似问题。

| 工作与来源状态 | 直接相关的已有内容 | 本文应验证的差异 |
| --- | --- | --- |
| [ReKV](https://arxiv.org/abs/2503.00540)，ICLR 2025，方法 | 滑窗编码、历史 KV 保存 RAM/磁盘、按问题检索 | 将压缩历史实际留在固定 GPU 预算内，并测峰值与在线遗忘 |
| [StreamKV](https://ojs.aaai.org/index.php/AAAI/article/view/37305)，AAAI 2026，方法；已读 [方法正文](https://arxiv.org/html/2511.07278v1) | 语义分段、未知未来问题时的 guidance prompt 压缩、跨层预算分配 | 不能把“未知问题＋分段＋分层预算”本身作为新颖点；需检验量化损失与生命周期决策的作用 |
| [WindowQuant](https://arxiv.org/abs/2605.02262)，2026；arXiv 注明 TACO 接收，方法；已读 §3 | 文本提示与视觉窗口相似度驱动位宽，阈值区分层，按位宽重排计算 | 当前问题可见与未来问题未知的差别；持续降位宽、访问机会和不可逆遗忘的影响 |
| [KIVI](https://arxiv.org/abs/2402.02750)，ICML 2024，方法 | K/V 非对称量化，提供基本量化配置参照 | 对视频 KV 重新验证量化轴和误差曲线 |
| [KVTuner](https://arxiv.org/abs/2502.04420)，ICML 2025，方法；已读 v5 §4–5 | 层敏感性、K/V 精度对、离线搜索与层聚类 | 分层静态配置是否已足够；片段×层动态决策能否带来额外收益 |
| [VidKV](https://arxiv.org/abs/2503.16257)，2025 arXiv v2，方法；已读 §3–4 | 视频专用低比特 KV；作者提出 V 按通道量化的选择 | 统一低比特视频基线足够强时，自适应分配是否仍有效；不直接移植论文中的质量结论 |
| [MiKV](https://arxiv.org/abs/2402.18096)，2024 arXiv，方法 | 重要 KV 高精度，其他 KV 以低精度保留 | 低精度保留与彻底删除的差别；持续输入下的有界存储 |
| [StreamingBench](https://arxiv.org/abs/2411.03628)，2024 arXiv，纯 benchmark | 视频时间线上提出问题，包含实时、上下文及多源理解 | 选择纯视觉适用任务，测持续更新而非每问重建缓存 |
| [OVO-Bench](https://arxiv.org/abs/2501.05510)，CVPR 2025，纯 benchmark | 含回溯、实时理解与未来触发场景 | 主测回溯和实时理解；未来触发需要不同交互协议，另作扩展 |

更有价值的候选贡献是：**揭示“未访问”信号的偏差，并用有校准依据的边际失真，在未知未来问题、不可逆精度下降和真实 GPU 峰值约束下统一分配资源。** 这一定位仍需实验与更完整的新颖性核验。

## 4. 共用实验协议

### 4.1 数据与时间因果性

第一模型使用仓库已有的 LLaVA-OneVision-Qwen2-7B；第二模型可用仓库支持的 LongVA-7B，需另行准备权重。检查模型目录不等于权重完整性验证。

| 数据/工作负载 | 用途 | 使用方式与边界 |
| --- | --- | --- |
| 本地 RVS ego/movie | 在线更新、重复提问、访问历史 | 沿用仓库前缀问答入口；先核验 `start_time/end_time` 含义。若只有证据时间而无真实 arrival time，称为“前缀回放协议” |
| 本地 CG-Bench | 证据覆盖、延迟历史提问 | 标注含 `temporal_windows`；将证据全部发生后的时刻设为提问时刻，是自定义因果回放，不等同原 benchmark 协议 |
| 本地 MLVU | 长视频、计数/顺序/总结任务与统一量化观察 | 本地文件为 `dev_debug_mc.json`，不能仅凭文件名宣称官方完整评测；末尾提问不能单独证明流式适应能力 |
| StreamingBench 视觉任务 | 对外报告在线任务能力 | 使用原始时间戳、官方划分与评价；音频依赖任务不计入纯视觉汇总。需增加数据入口 |
| OVO-Bench 回溯与实时任务 | 长期记忆和当前事件理解 | 分项报告；需增加数据入口 |
| 延迟首次提问 / 主题切换回放 | 直接检验遗忘机制 | 从有证据时间的 QA 构造，人工核验答案在新 arrival time 下仍成立，单独报告自定义结果 |

建议先选 30–50 个不同原始视频校准，另选 20–30 个视频开发；以原始视频/场景为单位拆分，不能把同一视频的片段或问题分到两侧。数据较小时采用按视频划分的交叉验证，保留最终未参与调参的测试集合。上述数量是启动建议，不是现有统计结果。

在线时间线必须满足：

1. 在 \(t\) 时刻只能编码时间戳不晚于 \(t\) 的帧，采样使用固定时间网格；不能先看完整视频决定保留哪些帧。
2. 问题到达前的压缩策略看不到该问题、答案和标注证据。到达后可用于当前检索，并影响之后的策略。
3. 同一视频连续提问共享缓存状态；问答临时 KV 默认在本问结束后释放。若保留聊天历史，所有方法使用相同设置并计入预算。
4. 只允许离线诊断中的 oracle 读取未来，用于机会差距分析；它不参与在线方法排名。
5. 同一时间点多问需要预定义次序；无顺序定义时，独立分支评估或明确标注人为顺序。未来标签只能写入评估日志，不能写回策略。

### 4.2 三种互补的测量轨道

**A. 固定状态的反事实干预。** 保存一个因果前缀的全精度参考 KV；只改一个片段、一个层或一组 K/V 位宽。固定检索集合、位置编号和回答输入，观察局部误差以及最终输出变化。它回答“该位置是否敏感”，不能代表长时间累计误差。

**B. 持续流式回放。** 从流起点运行各策略；历史压缩、查询、降级和删除持续累积。所有方法使用相同到帧时刻、到问时刻和输入内容。它检验真正部署效果。

**C. 真实 packed GPU 实现。** 在 B 上测实际字节、反量化、分配器及端到端耗时。模拟量化的 BF16 张量不会进入系统显存收益结论。

参考模型保留相同的滑窗编码、retrieval token 数、RoPE/位置规则。对短前缀可增设 dense reference，但其质量差异不能全部归因于量化。

### 4.3 指标与统计

质量：沿用各数据的准确率或开放式官方评分，报告相对同协议 BF16 参考的百分点变化；开放式评分固定 evaluator 版本、prompt、温度并抽样人工核验。不要用不同数据集评分直接平均。

机制：量化重建误差、K 导致的 attention 分布变化、attention 输出误差、回答 token 分布 KL、答案负对数似然变化、正确→错误的翻转率、证据覆盖和索引检索一致性。

\[
D^O=\frac{\|O_{quant}-O_{ref}\|_F^2}{\|O_{ref}\|_F^2+\epsilon},\quad
D^{KL}=\frac1{|Y|}\sum_j KL\big(p_{ref}(\cdot|Y_{<j})\|p_{quant}(\cdot|Y_{<j})\big).
\]

KL 使用相同 teacher-forced 前缀，用于诊断；另外必须保留自由生成评价。层内输出误差应同时测 attention 输出和经输出投影后的误差。删 KV 时需要重新计算 softmax 分母。

长期质量：按“距证据发生时间”和“距上次相关提问的机会数”分桶；分别统计首次被问到的历史事件、再次访问、计数/顺序、瞬时小事件、主题切换。未见相关标注不代表没有未来价值。

对有时间证据标注的问题，令 \(E_q\) 为已经到达且被共同输入采样覆盖的证据帧，\(R_t\) 为缓存仍保留的帧，\(S_q\) 为实际读取的帧。分别报告保留覆盖率 \(|E_q\cap R_t|/|E_q|\) 与读取覆盖率 \(|E_q\cap S_q|/|E_q|\)。跨多个必要事件的问题另报“所有证据区间均被覆盖”的比例；一帧重合不能证明完整证据保留。\(E_q\) 为空时标为不可测，并单独统计输入采样已漏掉证据的数量。

误删率使用未来确实需要该段的可评估事件为分母，统计提问前已删除对应证据的比例；另外记录因低位宽而答错、索引漏检和模型原本就答错，避免全部归因于删除。参考模型的注意力只用于代理分析，不能替代人工证据标签。

系统：KV 及整卡峰值、索引占比、最大存活历史 token 数、P50/P95 首 token 延迟（含检索、量化维护和等待）、回答生成时间、编码帧率、每段压缩耗时、重打包字节数、队列积压、OOM/超预算次数。若模拟真实到帧率，报告服务时延及 backlog；脱离实时的最快回放吞吐另报。

统计单位采用视频。相同视频和问题配对比较，按视频进行 bootstrap 95% 区间；随机策略至少 3 个随机种子，确定性策略无需为凑种子重复完全相同运行。预测指标比较应在未参与拟合的视频上评估，不能把数百万相关 token 当作独立样本。

### 4.4 起始参数

| 参数 | 建议起点 | 作用 |
| --- | --- | --- |
| sample FPS | 0.5 与 1 各一组，遵循原数据协议时优先原协议 | 分离输入采样损失与 KV 压缩损失 |
| 原始精度 | 模型原始 BF16；若后端需要 FP16则全方法一致 | 参考表示 |
| 位宽 | 16/8/4/2；初期 K/V 分开扫描 | 先测误差曲线，再裁剪配置空间 |
| 量化 group size | 32/64/128 的小规模校准，随后固定 | 防止分段收益混入分组参数收益 |
| 近期窗口 \(W\) | 32 帧，敏感性检查 16/64 帧 | 这里是采样帧数；按 FPS 换算秒数 |
| 语义段长度 | 4–16 帧起步 | 不强求边界与量化 group 对齐；尾块 padding 计费 |
| 编码 chunk | 16 帧起步，且不超过近期窗口和问题到达边界 | 控制峰值，并满足当前编码器窗口约束 |
| 历史检索量 | 固定历史 token 数，起步为 64 帧等价 token；近期去重 | 变长段不能以“相同 top-k 段数”声称等算力 |
| KV 峰值预算 \(B\) | 2/4/8 GiB 候选；先根据实际卡容量减去权重等固定项确定 | 所有时间点保持固定字节上限，不逐时按参考缓存比例放大 |

上述近期窗口设置需要先分离 ReKV 的编码窗口与问答窗口；直接改 `n_local` 可能截断检索历史。原实现配置应作为单独可复现参考保留。

## 5. 观察实验

### O0：GPU 内存究竟花在哪里？

**目的。** 确认可压缩部分占比，给后续位宽选择提供真实成本。

在短、中、长三个前缀以及每个问题到达点，分别记录编码前后、压缩迁移中、检索中、问答 prefill 和 decode 的内存组件。比较原生 ReKV、无卸载 BF16 历史原型、固定 INT4 packed 原型。BF16 无法容纳时使用较短可容纳前缀，不隐藏 OOM。

输出：显存分解表和峰值时间线，标出 `CudaCache`、`global_buffer`、`block_k`、recent、QA KV、scale、padding 与 scratch。核对实际 INT4 字节和理论公式。

**设计判断。** 若全精度缓冲或索引占主导，先修改存储与读取路径；若压缩后总显存几乎不变，不能继续把 payload 下降当成有效系统收益。

### O1：先辨认量化轴、K/V 不对称性及近期窗口影响

**假设。** 视频 KV 的量化误差受量化轴影响；K/V 位宽和保护近期帧的收益需要在目标模型上测量。

固定一批参考前缀、检索集合和 token 集合，对每个代表层执行：

- K-only：K 为 8/4/2 bit，V 为 16 bit；V-only 反之。
- K/V 联合：扫描 \(\{8,4,2\}^2\)，与 BF16 对照；重点比较同 payload 的 K8V4/K4V8、K4V2/K2V4。
- 量化轴：K 和 V 各比较 per-channel 与 per-token；固定 clipping 规则和实际 group size，记录有效 scale 数量。
- recent-only、history-only、两者都量化；同时改变近期窗口大小并保持总字节预算。

明确量化维度：将一个 KV head 写成 \(X\in\mathbb R^{N\times d}\)。per-channel 指固定 feature channel，在 token 轴的局部 group 上估计参数；per-token 指固定 token，在 feature 轴的局部 group 上估计参数。用显式 axis/group 描述，避免术语歧义。K 的测量要注明 pre-RoPE 或 post-RoPE，并与实际后端一致。

输出：`layer × K-bit × V-bit × axis` 误差热图、任务损失与真实字节的 Pareto 点；分析关键对象、小字/OCR、动作变化是否比静态背景敏感。

**设计判断。** 选择目标模型上稳定的量化轴；K/V 不对称须由数据支持。若近期窗口扩大只增加显存而无质量收益，缩小保护范围。若统一 INT2/INT4 已充分有效，自适应方案的研究重点转向更长时间范围或更严格预算。

### O2：什么边界能形成适合统一精度的片段？

**假设。** 语义变化与量化敏感性变化不完全重合；分段应使段内压缩决策接近一致，同时控制元数据和计算开销。

对同一连续视频计算四类因果信号：

\[
u_t=f_t/(\|f_t\|+\epsilon),\quad
c_t=1-u_t^\top u_{t-1},\quad
r_t=\left|\log\frac{\|f_t\|+\epsilon}{\|f_{t-1}\|+\epsilon}\right|,
\]
\[
d_t=1-\cos(u_t-u_{t-1},u_{t-1}-u_{t-2}),\quad
q_t=\|e_t-e_{t-1}\|.
\]

\(c_t\) 是内容方向变化，\(r_t\) 是幅值变化，\(d_t\) 是变化方向的转折。相邻变化幅度过小时关闭 \(d_t\)，防止静态视频中的数值噪声触发边界。\(e_t\) 是少数探测层上 K/V 的低比特重建误差、异常值比例和范围等组成的向量；它只是量化敏感性的候选代理。

比较六个分段器：固定长度；长度分布匹配的随机边界；余弦变化；幅值变化；余弦＋转折；余弦＋量化误差变化。均限定最短/最长段长，阈值在校准集确定。视觉特征先复用既有编码器输出，不新增昂贵模型。

**控制两类不同效果。**

1. 固定底层 token 量化 group、相同 token 保留集合和检索集合，只改变“共享位宽的段落边界”，测精度分配效果。
2. 再允许 group 在段边界重置，单独测量化 scale 同质性的收益与 padding 开销。
3. 匹配平均段长、段数和总实际字节；随机边界匹配段长分布。无法完全匹配时报告偏差，不仅比较段内误差均值。
4. 帧特征均值可能抹掉小物体变化，可增加 patch 变化上分位数作为一项消融；它不等价于有标注的事件边界。

用于在线比较的随机段长从校准集冻结的分布采样，不能读取当前测试视频未来的段长；利用完整测试分段匹配的控制只能列为离线诊断。

输出：段内真实单片段干预损失方差、段间差异、相同预算下 QA/KL、每千帧边界数、分段耗时；有人工事件标注时再报告边界 F1。可额外做已知完整轨迹的离线最优分段诊断，但明确其看到了未来。

**设计判断。** 只有当加入 \(q_t\) 在未见视频上改善相同字节下的质量或稳定性，且开销可接受，才进入方法。语义边界 F1 高但量化质量不变，不支持“量化感知分段”的必要性。转折信号无增益则保留简单余弦边界。

### O3：访问热度、重要性与量化敏感性如何区分？

**假设。** 被频繁访问的片段不一定需要高位宽；被少量访问但对低比特极敏感的片段可能更值得保护。

在参考轨道 A 中，按年龄、长度、检索分数和内容类型分层抽样片段。分别将单片段从 BF16 降到 8/4/2 bit，以及删除该片段，测量 \(D^O\)、\(D^{KL}\)、NLL 增量和自由生成翻转。每次只做一个干预，其他片段保持不变。

对比预测特征：年龄；过去检索次数/最近访问时间；归一化段 attention mass 与每 token mass；最大 head 的关注及 top-head 分位数；检索得分/排名；KV 重建误差及异常值；实际 attention 输出误差代理；“访问风险 × 条件敏感性”。同时测对整段总收益和单位节省字节收益的排序。

量化敏感性应测“降低一个精度档位的增量损失”，而非只有 BF16→INT2 的极端差异。对同样的 payload，比较“少量高精度帧”与“更多低精度帧”，以判断位宽与覆盖范围的替代关系。

**避免选择偏差。** 未被当前检索选中的段落，其本次干预损失可能为零；因此分别测固定 native retrieval、固定覆盖证据的诊断集合，以及全部可容纳历史。证据集合只用于离线分析。保持索引精度不变，再单独量化索引，区分“找不到”与“读不准”。

输出：在未参与拟合的视频上的 Spearman 排序相关、top-risk recall、同预算 allocator 的质量差距；按“高热度/低热度 × 高敏感/低敏感”列出四类实例，并标注检索机会是否充足。

**设计判断。** 若热度已解释全部收益，使用简单热度策略；若条件误差有独立增益，才保留二维风险建模。段 attention mass 受段长影响，须纳入长度或单位字节校正，不能直接比较不同长度段的总 mass。

### O4：长期未激活是否真的可以删除？

**假设。** 视频年龄与“经历了多少次可能相关的提问”不同；零访问还可能由检索偏差造成。

为每段记录：存在时长、见过的总问题数、其摘要被打分的次数、与其主题相关的提问机会数、真实被读取次数、读取后的 attention mass、最近一次相关机会、未来首次/再次被需要的评估标签。机会的在线估计只能使用当前已到达的问题和保留的摘要。标注证据只供评估。

构造四类工作负载：自然提问顺序；长时间无提问后首次追问；一段时间集中问主题 A 后转问 B；同一事件反复问后长间隔再次提问。延迟以采样帧或秒数、问题机会数两轴报告，不能只用一个“年龄”。

比较：按时间 TTL；LRU；访问次数衰减；按机会数校正的访问风险；机会风险＋内容多样性保护。随后在**相同字节预算**下比较直接删除与 `低比特全段 → 稀疏代表帧 → 删除`。各策略都统计最终被误删的证据，而非只统计缓存命中率。

输出：不同机会数下未来再次需要的概率曲线、首次访问延迟分布、概率校准曲线/Brier score、回溯 QA 准确率、误删率及其节省字节。视频结束还未再次访问属于右删失；可用生存分析，或仅用有完整后续观察窗口的样本，不能直接标为“永不再用”。完整参考轨道负责暴露被删除内容的后续需要。

**设计判断。** 无问题时不能因 hit count 为零认定不重要；若未来访问概率预测不可靠，采用保守的多样性/时间覆盖保留策略。若保护大量陌生历史使热点问答下降，则把这种权衡作为 Pareto 曲线，而非隐藏在总分中。

### O5：层敏感性、K/V 配对与片段×层交互

**假设。** 层有不同敏感性，但固定的层排序未必适合所有片段与问题。

第一轮逐层单独干预，扫描 K/V 位宽与量化轴，输出 \(D_{s\ell}(b_K,b_V)\)。第二轮做双层及关键层组联合干预，估计交互项：

\[
I_{\ell m}=\Delta\mathcal L_{\ell,m}-\Delta\mathcal L_\ell-\Delta\mathcal L_m.
\]

第三轮在完全相同字节预算下比较：所有层同位宽；每层固定配置；按层聚类配置；片段独立但所有层相同；片段×层联合配置。另设随机打乱层敏感度的控制组，位宽直方图和 token 保留集合保持不变。

将单干预得到的损失矩阵拟合为“片段风险 × 静态层系数 × 位宽误差”的可分解模型，再与有交互项的模型比较**留出视频上的预测误差和实际分配收益**。仅用拟合优度提升不足以证明复杂 allocator 值得使用。

输出：层×位宽热图、不同视频/问题类别的层排序一致性、双层交互分布、静态分层与动态分配在质量/控制耗时上的 Pareto 点。所有字节按实际 \(H^{KV}_\ell,d_\ell,n_s\) 加权，不能简单平均各层位宽。

**设计判断。** 层排序稳定且交互小：使用离线静态层分组。动态片段×层只有在固定预算下持续改善且控制成本有限时启用。单层都不敏感却联合崩溃时，要增加联合校准，不能宣称各层误差线性可加。

### O6：渐进降级与真实 GPU 路径是否可用？

**目的。** 将局部观察转换为持续系统证据，检查累计误差和实现开销。

使用同一个原始片段比较 BF16→INT2 一次量化与 BF16→INT8→INT4→INT2 多次重数量化；同时比较固定量化网格的嵌套编码。最终位宽、group、存活帧集合相同。测重建/问答误差、迁移时间、临时字节和每段转换次数。

再比较真实 packed 读取的三条路径：完整反量化后注意力（诊断用）；固定大小 tile 反量化并累积 softmax；融合 unpack/dequant-attention。各路径使用相同精度配置与检索集合，验证数值误差和峰值；融合版本未实现时不报告其加速。

在固定 \(B\) 下持续运行，至少覆盖预算首次饱和之后多个同等长度区间；加入无问题阶段和主题切换。记录显存是否有缓慢增长、缓存目录是否无界、临时 QA 缓存是否及时释放、arrival backlog 是否持续积累。

**设计判断。** 渐进重数量化误差明显时，限制转换次数或使用嵌套编码；片段过碎造成调度开销时，合并物理 pages 或改用较粗层组。若系统瓶颈在视觉编码，报告 KV 压缩收益及端到端收益的差别。

## 6. 候选方案：基于访问风险与失真代价的分段混合精度缓存

这是可实现的研究原型设计，最终保留哪些组件由 O1–O6 决定。主模型权重冻结；用小规模视频校准精度配置与风险估计器。

### 6.1 缓存结构与生命周期

缓存由近期 BF16 环形区、历史 packed pages、有限容量索引，以及固定大小的读写 scratch 组成。逻辑上使用连续片段作为策略单元，物理上使用固定大小 pages，允许一个段跨多个 pages。

| 状态 | 存储内容 | 进入条件 | 可执行的后续操作 |
| --- | --- | --- | --- |
| `RECENT` | 最近 \(W\) 帧完整 K/V；当前未闭合段 | 新帧到达 | 达到近期窗口边界后进入历史分配 |
| `HISTORY` | 所有保留帧的 packed K/V；各层 K/V 可不同位宽 | 退出近期窗口 | 保持、降位宽，或进入稀疏保留 |
| `ANCHOR` | 从该段选出的少量真实帧的 K/V、时间及帧索引 | 全段继续保留的边际价值较低 | 继续减少代表帧，或删除 |
| `REMOVED` | 不再存放该段可访问 KV | 预算压力下选择删除 | 不可恢复；从在线检索目录移除 |

`HISTORY` 的候选 K/V 精度对来自 O1/O5 的 Pareto 集，包括必要的 16 bit 保护。初始扫描不预先强制 K 位宽高于 V；例如 K8V4、K4V4、K4V2、K2V2 只是待验证候选。

`ANCHOR` 保留的是实际产生过的帧 KV。可先选段首、段尾和特征 medoid，再在固定保留率下与均匀抽帧、运动极值帧比较；长度不足时去重。减少帧时，各层采用相同 frame mask，便于维护时序对应；位宽仍可因层而异。不同层删不同帧留作后续扩展。

一个平均特征或检索向量只帮助找到片段，不能替代该段所有视觉细节。若要使用额外摘要 token，必须经过模型编码并作为单独机制验证，不能直接对 K/V 求平均便宣称等价。

主版本默认不可逆：历史位宽只降不升、frame mask 只减不增。旧段再次相关时，可以延长其当前表示的保留期限；INT2 转成 BF16 不会恢复已经丢失的信息。可恢复高精度需要额外残差、原始 KV 或重编码，均属于另一个计费设置。

### 6.2 因果分段器

复用视觉编码器的帧特征及少数探测层的 KV 统计。O2 若支持附加量化信号，则采用：

\[
z_t=\alpha\,\widetilde c_t+\beta\,\widetilde q_t
       +\gamma\,\widetilde d_t,
\]

其中各分量用校准集的稳健尺度归一化。\(d_t\) 仅在连续两次特征变化超过噪声门限时启用。\(r_t\) 保留为候选诊断，不默认加入最终式，避免亮度等幅值变化过度切段。

当段长达到 \(L_{min}\) 且 \(z_t>\tau\) 时，在当前已到达帧之前闭合旧段；达到 \(L_{max}\) 时强制闭合。可选双阈值滞回减少边界抖动。若 KV 统计直到当前帧编码后才能得到，此时只改变压缩段归属，不回溯改变已发生的 attention。

未闭合段始终可被当前问题读取。段长上限不超过可容纳的近期区；例如 \(L_{max}=16,W=32\)。问题发生在编码 chunk 内时，应先截断到到问时间再编码，不能等待未来帧来完成分段。

**复杂度目标。** 每帧统计仅依赖当前与前两帧，以及有限探测层；每帧描述开销近似 \(O(d_f)\)，KV 误差抽样开销另测。初始可用浅/中/深 3 个探测层，O5 后再决定是否足够。

### 6.3 分离索引、访问风险与精度损失

每个存活段维护以下有限信息：时间范围、存活帧列表、每层/层组的精度与地址、少量代表向量、过去提问机会统计、实际读取统计、量化误差摘要和当前表示状态。

**索引。** 初期保持 BF16 代表向量，以便把检索变化与 payload 量化变化分开。按 KV heads 存储而非复制到全部 query heads；检索时聚合对应 query heads。后续可独立研究索引 INT8 和共享层索引，并测 recall 及字节。物理 pool 的排列不改变逻辑帧编号。

**访问风险。** 定义 \(\hat p_s(H)\) 为未来 \(H\) 次提问内该段被需要的估计概率。校准标签来自参考回放中的未来真实证据或参考访问；使用后者时明确称为“模型访问代理”，不要混同任务真值。

一个最小可行估计器是对以下特征做分桶校准或正则逻辑回归：段年龄、相关提问机会数、过去访问频率、距上次相关问题的机会数、当前索引相似度趋势、内容新颖性及表示状态。可先取 \(H=5\)，用 1/5/10 次提问做敏感性检查。

- 每次问题到达，对所有存活摘要执行可计费的轻量打分，记录“有机会相关”与“实际选中”两个事件。
- 未被 top-k 选中的段没有真实 attention mass；其 mass 应为缺失值，不写成 0。
- 每次已读取段的 mass 用按问题归一化的更新，避免答案越长累计关注越大。
- 当前有问题但主题无关，不应等价于一次强负证据；机会阈值在校准集设定并验证。
- 无问题阶段不新增负观察。仅因视频更旧而降低概率属于待检验的先验，不能当作已观察到的不重要。

对观测稀少的段使用收缩到总体先验的估计，并构造保守风险 \(p_s^+=\operatorname{clip}(\hat p_s+\kappa u_s,0,1)\)。\(u_s\) 可用校准分桶置信区间宽度；在分布变化下它不是理论保证，应在 O4 测校准误差和首次访问召回。

**条件失真。** \(\hat D_{s\ell}(a)\) 估计段 \(s\) 在层 \(\ell\) 采用状态 \(a\) 时、被需要条件下的损失。起步用层组误差曲线乘当前段的误差尺度；O3/O5 若显示交互显著，再用含片段特征的回归表。用校准集的单干预、双层干预和端到端回放校准，不在每个在线时刻对所有配置重跑完整模型。

K 的快速数值代理可用少量**已观察到的 query**测：

\[
E^K_{s\ell}(b)=\mathbb E_{q\in\mathcal Q_{past}}
\frac{\|q(K_{s\ell}-\hat K_{s\ell}^{(b)})^\top\|_2^2}{d_\ell}.
\]

V 可用已读 attention 的加权重建误差。无历史 query 时采用校准查询协方差的对角近似和内容统计，并保持不确定性。视觉编码 query 与真实问题 query 要分别验证，不能默认分布一致。

若 \(\hat D\) 已从无条件未来 QA 损失拟合，它已包含访问频率，不应再乘 \(p_s\)。本文使用的是条件失真，两项分别校准，避免重复加权。

### 6.4 固定字节预算下的联合分配

令 \(a_s=(m_s,\{b^K_{s\ell},b^V_{s\ell}\}_{\ell=1}^L)\)，其中 \(m_s\) 为存活帧 mask。候选目标为：

\[
\min_{\{a_s\}}
\sum_s p_s^+(H)\sum_\ell\hat D_{s\ell}(a_s)
+\lambda C(\{m_s\})+\eta T_{maint}(\{a_s\}),
\quad M_{KV,peak}\le B.
\]

\(C\) 是时间/内容覆盖不足的惩罚，\(T_{maint}\) 是迁移与控制时间。这里的可加损失是调度近似，O5 用于检查其局限；不声称求得真实 QA 损失的全局最优解。

一个便于实现的**单调降级 allocator**如下：

1. 先预留 prefix、recent、最大当前问答 KV、固定 tile scratch、迁移 scratch、元数据和最坏 chunk 到达所需的字节。若最低固定开销已超过 \(B\)，此配置不可行，应在实验前换预算/窗口参数。
2. 新段在退出近期区时生成若干 Pareto 精度候选。已有段只能在当前信息可达的状态内变化，不能假设可从 INT2 恢复 INT8。
3. 枚举“一次降一个精度档位”“减少一组帧”“删除一个 anchor 段”等动作。对每个动作计算预计损失增量与**实际释放字节**：

\[
\rho(a\to a')=
\frac{\widehat{\Delta\mathcal L}(a\to a')
+\lambda\Delta C+\eta\Delta T_{maint}}
{M(a)-M(a')},\qquad M(a)>M(a').
\]

4. 选择 \(\rho\) 最低的可行动作，完成后刷新受影响的候选，直到满足容量及安全余量。由于量化的实际任务损失可能非单调，校准误差表可做单调拟合以稳定调度；原始观察仍保留。
5. 在段闭合、问题结束和达到预算水位时更新；正常情况下不给每个 decode token 调度全部历史。紧急输入到达时，使用同一预算规则提前腾空间。

可以给未知或低访问内容预留一小部分历史字节，例如以 10% 为起点，在 0/5/10/20% 上做消融；该比例是超参数。采用固定个数的时间桶与内容簇，各桶内做多样性保留，最老内容归并到溢出桶。桶数、索引和 reservoir 槽位都有上限，不能每个旧段永久保留一个增长中的“摘要例外”。

层配置先用 O5 得到的 4–8 个层组作候选起点，实际组数由校准决定。分配结果自然给出每层字节比例：

\[
\pi_\ell=\frac{M_\ell}{\sum_j M_j},
\]

无需人为设定“浅层少压、深层多压”。O5 不支持稳定分组时，可退回逐层静态配置或较粗的模型统一配置。

### 6.5 渐进降位宽的两种实现

**直接重数量化原型。** 退出近期区时从原始 KV 量化；后续只从当前量化表示解包，再生成更低位宽表示。每段转换次数限定为最多几个精度档，迁移逐 page 完成，及时释放旧 page。O6 测累计误差，不能把当前解包值当成原始真值。

**嵌套编码候选。** 固定每组初始裁剪区间 \([a,a+R]\)，先获得 8-bit 索引：

\[
q_8=\operatorname{clip}\left(\left\lfloor256\frac{x-a}{R}\right\rfloor,0,255\right),
\quad q_b=q_8\gg(8-b),\quad
\hat x_b=a+(q_b+\tfrac12)\frac{R}{2^b},\quad b\in\{8,4,2\}.
\]

常量组 \(R=0\) 单独处理。参数 dtype、clipping、group 和 padding 均固定并计费。8→4→2 只保留更粗的 bin 编号，重打包后实际释放低位数据；无需保存已经丢掉的高精度副本。这是固定网格的 mid-rise 量化器，和各位宽分别拟合 min-max 的方案不完全相同，必须比较其精度代价。

若低位数据仍保存在某个池中，就还占显存；仅修改 dtype 标签或 bit mask 不构成压缩。重新激活时可冻结后续降级，但不能找回已删除的 bit planes。

### 6.6 GPU 读取与正确的分块注意力

首个可实现版本使用 **packed 历史 + 固定 tile 反量化 + 全局 softmax 累积**。每次只展开一个或少数 tiles，避免将所有历史或整个检索集合展开成长期 BF16 tensor。之后再开发融合 kernel。

对于每个 query 和 head，读取第 \(j\) 个 tile，得到含正确位置编码与 mask 的 logits \(z_j\)，计算：

\[
m_j=\max z_j,\quad
\ell_j=\sum\exp(z_j-m_j),\quad
o_j=\sum\exp(z_j-m_j)V_j.
\]

以 FP32 累积 \((m,\ell,o)\)：

\[
m'=\max(m,m_j),\quad
\ell'=e^{m-m'}\ell+e^{m_j-m'}\ell_j,\quad
o'=e^{m-m'}o+e^{m_j-m'}o_j,
\quad O=o/\ell.
\]

使用稳定初始化并处理全 masked tile。不能分别对每段 softmax 后直接相加，否则段间归一化错误。recent、prefix、选中历史和当前文本 KV 的可见 token 集合须去重，再按同一注意力语义合并。

同精度 pages 可以批处理，减少 kernel launch；但物理重排必须携带逻辑 token/帧位置。ReKV 的逻辑位置及 RoPE 应与对应参考设置一致，不能因 page 排序重新编号。因果 mask 与滑窗 mask 也依赖逻辑位置，而非物理地址。原始视频时间戳另外保存，不直接假定它等于模型的位置编号。

当前多阶段 Triton 实现可供累积器复用，但还需实现 packed 解包接口、position/mask 映射和异步生命周期。若用 `get_score=True` 保留所有 K 引用，tile scratch 会失去有界性；在线统计应就地归约，或只对少数探测层/问题抽样。

索引检索默认保持固定历史 token 数，以隔离压缩贡献。增加可读取的历史 token 数可能改善答案，但同时增加计算；应再给出相同 P95 延迟下的比较，不能仅因低比特存了更多帧便宣称同算力更好。

### 6.7 在线伪代码

```text
offline_calibrate(calibration_videos):
    determine quantization axes, group size and precision candidates
    fit conditional distortion tables and layer groups
    calibrate causal segmentation and future-access risk estimator
    freeze these choices before held-out evaluation

on_frames(arrived_frames):
    split input at the next question timestamp and safe chunk size
    reserve space for incoming chunk and its temporary tensors
    if space is insufficient: apply feasible history downgrades first
    encode only arrived frames using the protected local window
    append new KV to the recent ring; update causal segment statistics
    close segments at observed boundaries or maximum length
    for data leaving the recent ring:
        transfer ownership to packed history without retaining FP duplicates
        initialize bounded index, risk prior and distortion summaries
    enforce budget including in-flight migration and scratch

on_question(q, arrival_time):
    encode all required frames up to arrival_time, never beyond it
    score bounded indexes using q; distinguish opportunities from accesses
    select historical tokens under the fixed read budget
    form a deduplicated logical view of prefix/recent/history/current text
    answer using tiled dequantization and global softmax accumulation
    update statistics from the current question and observed attention
    release question-specific temporary KV according to the fixed protocol
    protect currently relevant history from further immediate degradation
    run the monotone allocator if the watermark requires it

apply_action(action):
    lock source pages until current readers complete
    requantize or remove frames in bounded tiles using reserved scratch
    atomically replace addresses after the completion event
    release old pages and stale index entries
    recompute affected costs and assert peak budget accounting
```

为了保证既定近期窗口始终保持原始精度，相关空间须提前预留；若近期区暂时超出窗口，那部分 chunk 峰值也要计入。异步维护不能通过“尚未同步”隐藏临时分配或压缩延迟。

## 7. 对照与消融：如何证明增加的机制确有作用

### 7.1 同预算比较矩阵

所有主比较使用同一 GPU 常驻历史协议、骨干、采样帧、近期窗口、精度保护、位置策略、回答上限及评价器。若方法无法支持该协议，应保留其原生实验并清楚标注差异。

| 对照 | 回答的问题 | 公平性要求 / 实现状态 |
| --- | --- | --- |
| 同编码/检索的 BF16 参考 | 压缩造成多大损失？ | 短前缀可实跑；长前缀参考可使用外部离线测量，不能视为同预算可部署方法 |
| 固定 INT8/INT4/INT2 | 自适应是否胜过简单量化？ | 同 quantizer、group、recent 与真实字节；预算不足时加统一 eviction 并披露 |
| 仅近期 BF16；BF16 均匀历史抽样 | 位宽压缩是否优于直接缩短记忆？ | 使用相同总字节，报告能保留的时间跨度 |
| KIVI / VidKV | 强量化器是否已解决主要问题？ | 核验原实现、版本、后端与模型兼容性；本次未运行。移植版须标“适配” |
| MiKV | 重要内容高精度、其他低精度是否足够？ | 无界低精度历史不能直接用于无限流；原生与有界适配分别标明 |
| KVTuner 静态层配置 | 动态段落分配的增益是否超出静态层差异？ | 在相同视频校准集重新搜索；保留其完整搜索设置后才称完整基线 |
| StreamKV | 语义分段、提示压缩、层预算的近邻对照 | 按正式方法完整复现；核验 GPU/CPU 资源设置。改变后端的版本单列 |
| WindowQuant | 窗口相似度分配位宽的近邻对照 | 原生已知 prompt 设置单列；因果改造只用已到达问题或固定通用 prompt，称“适配”，不能看未来问题 |
| 年龄规则：recent→8→4→2→删除 | 复杂风险估计能否胜过时间启发式？ | 同转换档位、删除策略空间及实际字节 |
| 仅访问热度；仅失真；两者联合 | 两类信号是否互补？ | 其他组件完全一致；比较单位释放字节的价值 |
| 原生 ReKV 卸载 | 与当前可用系统有什么实际差别？ | 独立资源设置；同时报告 CPU/磁盘、传输量与延迟 |

论文方法尚未在当前仓库核验可运行，表格是待实现/待复现矩阵。仅使用某方法的一条启发式时，将其列为机制对照，不冒称完整方法。

### 7.2 必要消融与结果解释

| 消融 | 固定什么 | 支持/否定什么 |
| --- | --- | --- |
| 语义段换固定段、随机边界 | 段长分布、量化 group、字节、检索集合 | 分段位置本身是否有价值 |
| 去掉量化变化信号 | 语义阈值、段长、控制预算 | 量化感知边界是否必要 |
| 去掉访问风险或条件失真 | 动作空间与字节 | 分配收益来自哪一种信息 |
| 把未选中的段记为零 vs 保持缺失 | 其余更新逻辑 | 选择偏差是否导致错误遗忘 |
| 不确定性/多样性保护为 0 | 总预算不变，释放字节给其他段 | 首次迟到提问的保护及热点质量代价 |
| 所有层统一 vs 静态层组 vs 动态层组 | 总字节、段落、token mask | 分层自适应的必要程度 |
| 直接删除 vs 先降位宽再删帧 | 最终预算与工作负载 | 渐进保留是否改善历史质量 |
| 一次量化 vs 多次降级 vs 嵌套编码 | 最终精度和帧集合 | 不可逆转换的额外误差 |
| 模拟量化 vs packed tiled 读取 | 完全相同量化数值与检索集合 | 算法质量与系统实现是否一致 |

避免所有配置做庞大全因子实验。先以 O1/O3/O5 确定量化器与主要信号，再对入选方案执行上述机制消融。

## 8. 实施顺序与仓库落点

| 阶段 | 最小交付 | 拟修改/新增位置 | 继续投入的依据 |
| --- | --- | --- | --- |
| P0：测量 | GPU 内存分解、统一时间线日志、参考前缀 | `Abstract_ReKV.calc_memory_usage`；stream runner；KV manager | 预算口径可复核，输入无未来泄漏 |
| P1：辨认误差 | O1、O3、O5 的模拟干预与留出验证 | 新增独立 observation runner；quantizer 接口 | 局部异质性和分配收益确实存在 |
| P2：建立真实预算基线 | 固定 INT4 packed store，tile 反量化读取 | `kv_cache_manager.py`；attention 路径；量化模块 | packed 字节真实下降，问答峰值符合预算 |
| P3：最小自适应方案 | 固定长度段＋静态层组＋风险/失真 allocator | 新增 segment directory 与 budget controller | 同字节下优于统一量化和年龄规则 |
| P4：观察驱动完善 | O2 支持的边界；O4 支持的保护与渐进删除 | segmenter；统计器；降级/删帧动作 | 对延迟首次问答或主题切换有稳定收益 |
| P5：系统完善 | bounded metadata、异步安全、可选融合 kernel | packed allocator；Triton 后端 | O6 证明峰值、时延及累计质量符合目标 |

本次只交付研究设计，不修改上述运行代码。现有 `run.sh`、评价脚本及测试中的工作区改动保持原样。

建议先做一个 1–2 分钟级前缀上的可容纳 BF16 参考，再扩大到预算饱和后的长回放。这里的“1–2 分钟”是测量规模建议，不是最小长期记忆测试；O4/O6 的时间跨度必须覆盖所声称的历史保持范围。

实现时只需要几项关键正确性检查：打包/解包与尾块；分块 softmax 与同量化张量的稠密计算一致；位置/mask/GQA 一致；不可逆状态迁移及释放无泄漏；预算饱和仍不超峰值。它们属于实现验证，不作为论文质量结果。当前阶段没有实验代码变更，无需启动模型或编造 smoke 结果。

## 9. 日志字段与可直接填数的结果表

### 9.1 推荐数据结构

保持三张关联表，避免只保存聚合均值而无法分析错误来源。

```text
run:
  run_id, git_commit, model_revision, quantizer_revision, attention_backend
  dataset_revision, video_split_hash, seed, sample_fps, arrival_protocol
  budget_bytes, recent_frames, retrieval_token_budget, max_answer_tokens
  quant_axes_KV, group_size, scale_dtype, clipping_rule, rope_policy
  offload_allowed, calibration_video_ids_hash, policy_configuration

segment_action:
  run_id, video_id, segment_id, action_time, start_time, end_time
  original_frames, retained_frames, layer_or_group, bK, bV, representation
  opportunities, accesses, last_access_opportunity, attention_mass_or_null
  predicted_access_risk, uncertainty, conditional_distortion_estimate
  action_from, action_to, actual_bytes_freed, migration_peak_bytes, action_ms

query_result:
  run_id, video_id, query_id, arrival_time, evidence_time_ranges
  retrieved_frame_ids, retrieved_logical_positions, first_or_repeat_query
  qa_metric, reference_qa_metric, KL, NLL_delta, evidence_coverage
  kv_payload_bytes, kv_metadata_bytes, kv_scratch_peak_bytes
  gpu_allocated_peak_bytes, gpu_reserved_peak_bytes, process_peak_bytes
  host_kv_bytes, transferred_bytes, queue_delay_ms, ttft_ms, generation_ms
  overflow_count, oom, encoded_through_time
```

`evidence_time_ranges`、未来访问标签等字段仅写到评估侧，策略接口不可读取。attention 缺失保持 null。全部字节字段采用 bytes，展示时明确 MiB/GiB；所有延迟字段说明包含哪些阶段。

### 9.2 观察结果模板

| 观察 | 留出视频上的主结果 | 开销/预算 | 据此选择 |
| --- | --- | --- | --- |
| O1：量化轴与 K/V 配对 | TBD | scale/padding 字节 TBD | TBD |
| O2：语义/误差边界 | 同预算 QA / KL 差异 TBD | ms/千帧 TBD | TBD |
| O3：热度×失真 | 风险排序、allocator 收益 TBD | 更新耗时 TBD | TBD |
| O4：机会校正 | 延迟首次问答质量、误删率 TBD | 保护字节占比 TBD | TBD |
| O5：层交互 | 静态/动态差异与区间 TBD | 校准和控制耗时 TBD | TBD |
| O6：持续 packed 运行 | 累计质量和超预算次数 TBD | 峰值与 P95 TTFT TBD | TBD |

### 9.3 主结果模板（每数据集、模型和固定预算各一张）

| 方法 | GPU KV 峰值 GiB | 整卡峰值 GiB | 存活历史 token | 原生 QA 指标 | 首次迟到 QA | 重访 QA | P95 TTFT ms | 压缩维护 ms/段 | 外部历史存储 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 同协议 BF16 参考 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 按参考模式注明 |
| 固定低比特＋统一有界策略 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 无 |
| 静态层配置＋固定分段 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 无 |
| 年龄驱动渐进量化 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 无 |
| 风险×失真分配 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 无 |
| 完整入选方案 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 无 |

StreamKV、WindowQuant、VidKV 等完整基线在完成兼容性核验后加入对应协议的表格。不能把不同 backbone、不同问题可见性、不同预算下的论文数字直接填入。

建议最终制作五类图：质量–GPU 峰值 Pareto；片段×层量化损失热图；未来访问风险与机会数曲线；预算饱和后历史年龄–质量曲线；端到端时延分解。它们均待真实数据后绘制。

## 10. 最先回答的四个问题

1. **统一 INT4/INT2 在当前视频模型上是否已经足够？** O1 决定自适应分配是否有可观空间。
2. **单段的访问热度和条件量化损失是否提供互补信息？** O3 决定统一边际损失调度是否成立。
3. **层敏感性是否稳定到可离线固定？** O5 决定采用简单静态层组还是动态片段×层配置。
4. **无访问到底意味着无价值，还是没有机会被问到？** O4 决定渐进丢弃的依据及必须保留的历史覆盖。

若这四项得到支持，再由 O2 选择最简单有效的分段方式，并以 O6 验证真正的 GPU 压缩收益。这样每一个方法组件都有对应观察，而不是先确定复杂框架再寻找支持它的结果。

