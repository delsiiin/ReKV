# 在线视频 KV 压缩

采用 ReKV 单进程顺序执行方式的单模型、GPU 常驻压缩后端，当前支持本地 LLaVA-OneVision Qwen2 checkpoint、单路视频、一个在途问题。原有评测命令不变。新增入口支持逐帧视频回放和按时间到来的问题；Python API 支持外部实时帧源。

## 运行

在 `rekv` 环境、项目根目录运行：

```bash
python -m video_qa.rekv_stream_vqa --compressed \
  --video data/rvs/ego/videos/9198b9a4-8d8f-4ba6-9924-4c86982d890a.mp4 \
  --device cuda:0 \
  --budget-tokens 4096 --max-segment-frames 32 \
  --question-time 2 --question 'What is the person doing?' \
  --max-frames 8 --output results/streaming-kv/result.json
```

也可以直接运行 `python -m video_qa.compressed_stream_vqa`。`--questions path.json` 接受按时间升序排列的 `[{"time": 2.0, "question": "..."}, ...]`；不指定时间时在视频结束后提问。问题快照不会包含晚于该问题的帧。每个问题都回答完毕后再编码后续帧，不在答案生成期间插入视频编码。

模型在调用方进程中加载一次，视频编码、段总结和问答使用同一个对象顺序调用。没有推理子进程、IPC 队列或异步解码调度。设备编号使用 **PyTorch/CUDA 的逻辑编号**，默认 `--device cuda:0`。旧的 `--encode-device/--qa-device` 仅接受相同设备作为兼容别名；不同设备会明确报错。输入为 `.npy` 时沿用 RVS 每秒一帧的约定；MP4 按实际视频 FPS 采样。

```python
from model.streaming.runtime import StreamingConfig, StreamingRuntime

if __name__ == '__main__':
    config = StreamingConfig(device='cuda:0', budget_tokens=4096)
    with StreamingRuntime(config) as stream:
        stream.append_frame(rgb_frame_0, timestamp=0.0)
        answer = stream.ask('What is happening?', cutoff=0.0)  # 同步完成回答
        stream.append_frame(rgb_frame_1, timestamp=1.0)  # 回答后再继续编码
        stream.close_segment()  # 视频结束时提交末段
        stats = stream.memory_stats()
```

帧和问题必须按事件时间提交；不支持在已经编码未来帧后查询历史快照。`reset()` 仅清空视频状态，不重新加载模型。模型执行异常会释放问题快照并关闭当前 runtime；`close()` 释放该 runtime 持有的模型引用。接口面向单线程顺序调用。

## 算法

- 分段复用 observation 的投影/池化前视觉特征距离、历史 EMA 阈值、首对帧 warmup；默认 `alpha=.2`、`min_segment_frames=1`。达到 `max_segment_frames=32` 或 summary 上下文容量上限时闭段，不重置 EMA。
- summary 使用 observation 代码中的提示词，在独立原生上下文生成；重新分词并提取纯文本 prefill 的 RoPE 前 Q。每层按原生 GQA 映射求 summary Q 与暂存的 **RoPE 前 K** 的点积均值，不做 softmax，不跨层平均。大于或等于该层帧内均值的 token 归为高分区，其余归为低分区。分区掩码作用于 **RoPE 后 K 和原始 V**；V 不应用 RoPE。summary 为空/计算失败时全帧按高分保留并记录原因。
- 持久缓存、量化载荷和问题快照中的 K 均为编码时局部 attention 使用的 RoPE 后 K。RoPE 前 K 仅作为分区评分工作内存，段提交成功或评分失败后均释放；不进入快照。未启用压缩的基线也存储 RoPE 后 K，但不保留评分副本。
- sink 为初始化提示词；最新两帧是 recent，始终使用原始模型精度。未闭合段全精度暂存并可检索。各层的标签在段提交时固定。
- 每个段提交后若超预算：直接丢弃所有已闭合非保护帧的低分区；若仍超预算，则将这些帧各层的高分区各降一级（16→8→4→2），每次事件最多降一级；仍不足才淘汰 retrieve 频率最低的整帧，频率相同则淘汰更早入库的帧，即按 `(hits, frame_id)` 升序淘汰（LFU，同频时 FIFO）。`hits` 为累计问题检索命中次数，`frame_id` 在入库时单调递增；每淘汰一帧后重新检查预算。
- 一个问题在任一层用到某帧，该帧只加一次命中，recent 同样计数；summary 不增加命中。采用单调问题 ID 去重，不积累无限增长的历史 ID 集合。
- 分组方向参考 KIVI：**K per-channel**，固定 head 和 channel，沿 token 维分组；**V per-token**，固定 head 和 token，沿 channel 维分组。默认 group size=128。首次压缩先生成 16bit 整数编码，再降到 8bit；后续逐次降为 4/2bit。初始化、整数降位和打包仍采用 PM-KVQ 的公式、int16 容器和高位优先顺序。不使用 outlier calibration；sink、recent 始终保留原始模型精度，高分区在预算不足时渐进降位。

### 量化计算

仅量化已经分类为高分的 K/V，输入形状为 `[H_kv, N_high, D]`，每层、每帧独立处理，`G=group_size` 必须为 8 的正整数倍。

| 缓存 | 分组范围 | scale/offset 形状 | 内部打包方向 |
| --- | --- | --- | --- |
| K（`axis='channel'`） | 每个 channel 内的 G 个存活 token | `[H_kv, D, ceil(N_high/G)]` | 转为 `[H_kv, D, N_high]` 后沿 token 维打包 |
| V（`axis='token'`） | 每个 token 内的 G 个 channel | `[H_kv, N_high, ceil(D/G)]` | 沿 channel 维打包 |

K 的分组按高分 token 的原始顺序进行，不跨帧或层拼组；组内 token 共享该 channel 的量化参数。反量化后 K/V 都恢复为 `[H_kv, N_high, D]`，保持原始 token 顺序。较短帧的 K 尾组会增加每个有效 token 分摊的参数与填充开销，预算账本按实际 storage 字节计量。

这里的 K 是 RoPE 后 K。每帧记录编码时的旋转起点 `rope_start`，存活 token 的源位置为 `rope_start + 原始帧内索引`。检索拼接、删 token 或滑窗移动后，读取的 K 只施加位置差旋转 `R(目标位置−源位置)`，不对已旋转 K 再施加完整 RoPE。该处理保留 ReKV 的局部重排位置语义；sink 补充 attention 阶段单独变换到原有零位置语义。量化发生在源位置旋转之后，位置差旋转不会重新量化或改写持久缓存。

底层 `StreamingKVCache.append_frame(timestamp, layers, score_keys=..., rope_start=...)` 的 `layers` 接收 RoPE 后 K/V，`score_keys` 接收逐层 RoPE 前 K；缺少评分 K 时分区会明确记录错误并按全高分保留，不会误用 RoPE 后 K 评分。decoder 返回的每层缓存为 `RotaryCache(key, value, positions)`，编码和后续解码必须传递位置元数据。

第一次超预算压缩的路径是 **原始模型精度 → 16bit 整数编码 → 8bit 整数编码**。INT16 是本次压缩的中间阶段，不作为新增预算事件；未触发降位的高分区仍为原始模型精度，可压缩帧的低分区在超预算事件中直接删除。每个组取 `m=min(x)`、`M=max(x)`：

```text
scale16 = clamp((M-m)/65535, 1e-5, 1e5)
offset  = -m
q16     = clamp(round((x+offset)/scale16), 0, 65535)
```

随后使用 PM-KVQ 原始整数移位公式降位。16→8、8→4、4→2 的 scale 分别乘以 **257、17、5**；offset 不变。反量化为 `x_hat=q*scale-offset`，不保留原精度备份，不重新计算范围。整数编码虽为无符号数值，存储容器为 `torch.int16`，解包时通过掩码恢复无符号值。

16bit 初始化、scale 截断、整数降位及打包布局与本地 PM-KVQ 对照验证；K 的分组方向改为 KIVI 的 per-channel，并在转置布局上验证逐级编码与反量化。适配 ReKV 时保留 FP32 参数计算/存储，读取时恢复模型 dtype；PM-KVQ 数值对照以其 **FP32 输入量化器**为参考，使用相同输出 dtype 比较反量化结果。没有接入 PM-KVQ 的 SmoothAttention 重参数化或预算分配，也没有改为 KIVI 的直接目标位宽量化或 residual 缓存策略。

本后端支持两个方向的尾组：只用有效元素计算 min/max，K 的 token 维和 V 的 channel 维分别补齐到 8 的倍数供 2bit 打包，填充值不参与量化范围。模型 head_dim=128、G=128 时 V 无需填充，K 是否填充由该层该帧的高分 token 数决定。


## 缓存与内存口径

`budget_bytes = budget_tokens × 4 × Σ(layer.kv_heads × layer.head_dim)`，包括 K/V 两份、每个 FP16 元素 2 字节。实际计量覆盖管理器中的持久 K/V、量化参数、填充、位置索引、代表向量与元数据；以唯一 tensor storage 的 `nbytes` 加 Python 对象 `sys.getsizeof` 计量，不把逻辑 token 数当成真实压缩存储。

该指标是**受管理缓存字节**，包含 GPU tensor storage 和少量 CPU Python 元数据，不是进程 RSS 或 CUDA allocator 的全部占用。模型权重、summary/问答临时张量、未闭合段非 recent 部分、被问题固定的旧版本属于工作内存。RoPE 前评分 K 由缓存账本单独计入 `cache_working_bytes`，不计入持久预算，段提交后释放。预算保证发生在段提交处理完毕时；中间帧到来后可暂时超过预算。启动时会保守检查 sink/recent 的最低容量。

预算和报告使用 `accounting.py` 的增量引用计数账本：帧新增、分类、降位、删除以及快照获取/释放时更新变化的对象与存储。Python 对象身份和底层 tensor storage 分别去重；张量 view、跨层共享对象和旧快照均不会重复计费。正常预算查询与 `memory_stats()` 不扫描全部帧或访问全部 tensor storage，开销不随保留帧数增长；新记录的登记成本在帧更新时支付。仅修改命中数或关闭标记时，复用原有层与张量的登记记录。

段提交复用缓存返回的统计结果；模型参数字节数只在引擎初始化时计算。开放帧、位宽分布和逐帧 token 数也在更新时维护。整帧淘汰后的预算判断只读取账本，不再对剩余缓存重新递归核算。账本仅保存对象 ID、字节数、引用关系和弱引用，不额外持有已淘汰张量；回调采用弱引用，丢弃缓存不需要等待循环垃圾回收来释放 GPU 张量。

`cache.audit_accounting()` 可显式运行原始递归统计，核验持久字节、tensor/CUDA 字节、快照工作内存和持久帧集合；它用于测试或低频诊断，不在正常预算路径调用。通过 `cache.frames[id] = replace(...)`、`del`、`pop` 等操作更新不可变记录会自动更新账本；整体替换 `cache.frames` 属于兼容路径，会重建账本。入库后张量的形状和 storage 应保持不变。持久映射按实际保留的容器容量计费，因此元数据字节可能与旧版每次新建临时 dict 的结果略有差异。账本与索引自身的 CPU 管理结构由进程 RSS 反映，不包含在受管理 KV 预算中。

正确性测试：`python -m unittest discover -s tests -p 'test_streaming_accounting.py'`。可用下面的合成基准比较管理路径；`--baseline` 接受修改前保存的 `model/streaming/cache.py`，省略时仅测当前实现。默认每帧 8 个 token、28 层，主要复现对象数量；`--replay` 额外计时逐帧入库、token 评分、每两帧提交及报告。测量不包含模型前向、summary 生成或量化，不能作为整模型加速比例。

```bash
python tests/benchmark_streaming_accounting.py \
  --baseline /tmp/cache-before.py --frames 128 --repeats 5 --replay
```

本机 CPU 合成基准（128 帧、28 层，单项预热 2 次后取 5 次中位数；连续重放为单次测量）：

| 管理路径 | 修改前 | 增量统计 |
| --- | ---: | ---: |
| 单次预算查询 | 92.70 ms | 0.0047 ms |
| 单次内存统计 | 187.18 ms | 0.0064 ms |
| 一帧提交及统计，不触发压缩 | 577.57 ms | 0.105 ms |
| 128 帧连续入库、评分、64 次提交及报告 | 31.17 s | 1.48 s |

增量登记有写入成本：同次基准中单帧入库从 1.85 ms 增至 2.77 ms；连续重放仍减少约 95% 的总耗时。上述结果只度量合成管理路径，不含真实视频模型推理。

主缓存和问题快照都留在调用方进程中；K/V、低位载荷、scale/offset、位置索引、代表向量、summary Query 和待总结视觉特征均保留在同一 GPU，压缩、检索、反量化不经过 CPU offload。CPU 只处理输入图片、帧 ID、时间、计数和文本等控制信息。降位和删除通过替换不可变记录完成；问答完成逐层检索后立即记录激活并释放快照，答案继续使用其选出的临时 GPU KV。不创建跨进程模型或缓存副本。检索代表向量使用量化前、RoPE 后的整帧 Key 均值，查询 Q 在各帧存储坐标中紧随该帧应用 RoPE 后评分；不使用分区暂存的 RoPE 前 K。删除部分 token 不改写代表向量。recent 占用 `topk` 名额，最终按时间拼接每层存活 token，并通过位置差旋转对齐 ReKV 滑动窗口注意力。

输出字段：

| 字段 | 含义 |
| --- | --- |
| `commits[].before_bytes/after_bytes` | 段提交时压缩前后的持久缓存字节 |
| `memory.persistent_bytes` | 当前持久缓存占用 |
| `memory.cache_device / offload_enabled` | 缓存设备 / 固定为 false |
| `runtime.inference_pid / model_instances / execution_mode` | 当前调用方 PID / 单模型实例 / sequential |
| `memory.model_parameter_bytes` | 当前模型参数的字节数 |
| `memory.cache_working_bytes` | 开放段、旧快照等额外缓存占用（当前值） |
| `memory.feature_working_bytes` | 待总结段的视觉语言特征占用 |
| `gpu_peak_bytes` | 对应进程 PyTorch 分配显存峰值，包含模型权重 |
| `process_peak_rss_bytes` | 对应进程 RSS 高水位，包含模型加载 |
| `memory.bits` | 高分区各位宽的“层×帧”记录数；兼容键 `0` 保留且为 0，低分区删除见 `memory.counts` |
| `memory.counts` | 超预算事件、低分区删除帧次、整帧淘汰次数 |
| `summary_seconds/score_compress_seconds/encode_seconds` | 当前进程中编码相关累计耗时；编码时间不含闭段 summary |
| `answers[].qa_seconds` | 同步问题检索与完整答案生成耗时 |

分段/summary 的存在会增加编码延迟。此实现验证功能和预算约束，不预设答案质量提升或实时吞吐达标。OneVision 当前要求每帧 196 个语言模型 token；模型配置不匹配时明确报错。PM-KVQ 仅作为算法参考，不需要安装其旧版 Transformers Cache 类。

## 验证

```bash
python -m unittest discover -s tests -p 'test_streaming_*.py' -v
python -m unittest discover -s tests -v
```

测试包含 observation 数值对照、真实位打包、逐级预算策略、保护区逐值一致性、层间变长缓存、小型 Qwen 前向、同进程顺序事件、单次模型构造计数、reset 不重载模型、CUDA 缓存禁止 offload、编码/summary 后切回问答的一致性、激活去重和异常清理。真实模型短视频结果保存在运行时指定的 `--output` 中。

## 与原 ReKV 的关系

原 `video_qa/base.py` 在每个评测实例中调用一次模型加载，`ReKVStreamVQA` 依次调用同一个对象的 `encode_video()` 和 `question_answering()`；原缓存管理器使用 CPU offload。README 描述了跨进程/跨 GPU 的分离能力，但当前流式评测入口没有单份模型的跨进程并发前向实现。新增后端不修改原缓存管理器，也不改变未启用 `--compressed` 的入口。

新增后端同样在调用方进程中加载一次模型，按 `append_frame → ask（完成回答）→ append_frame` 顺序执行。与原 ReKV 的主要区别是新增后端将压缩缓存保留在 GPU，按预算执行量化和淘汰。旧版新增后端的 `ask_async()`、延迟取结果的 `answer()`、队列和超时配置已移除；改用同步 `ask()`。

模型直接加载时默认使用 FP16；observation 网页注入的模型可使用 FP16、BF16 或 FP32。sink、recent、评分暂存与未压缩帧保留模型 dtype，反量化恢复同一 dtype。FP32 使用 PyTorch attention，CUDA 上的 FP16/BF16 使用现有 Triton attention。预算仍以 FP16-token 等价字节定义，FP32 张量按实际 4 字节计入账本。`bits`/`bit_order` 中的兼容键 `16` 表示未量化状态，真实浮点精度由 `cache_dtype` 或 `full_precision_dtype` 指明。
