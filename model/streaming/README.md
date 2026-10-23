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
- summary 使用 observation 代码中的提示词，在独立原生上下文生成；重新分词并提取纯文本 prefill 的 RoPE 前 Q。每层按原生 GQA 映射求 summary Q 与在线视频 K 的点积均值，不做 softmax，不跨层平均。大于或等于该层帧内均值的 token 保持 FP16。summary 为空/计算失败时全帧按高分保留并记录原因。
- sink 为初始化提示词；最新两帧是 recent，始终使用原始 FP16。未闭合段全精度暂存并可检索。各层的标签在段提交时固定。
- 每个段提交后若超预算：所有已闭合非保护帧的低分区各降一级（16→8→4→2）；仍超预算则从旧到新逐帧删除低分区，每帧后重新检查；仍不足才按 `(问题命中次数, 时间, frame_id)` 淘汰整帧。
- 一个问题在任一层用到某帧，该帧只加一次命中，recent 同样计数；summary 不增加命中。采用单调问题 ID 去重，不积累无限增长的历史 ID 集合。
- K/V 都使用 PM-KVQ 的 `per_group` 方向：固定 head 和 token，沿通道维分组，默认 group size=128。首次压缩先生成 16bit 整数编码，再降到 8bit；后续逐次降为 4/2bit。打包采用 PM-KVQ 的 int16 容器和高位优先顺序。不使用 outlier calibration；高分 token、sink、recent 始终保留原始 FP16。

### 量化计算

仅量化已经分类为低分的 K/V，输入形状为 `[H_kv, N_low, D]`，每层、每帧独立处理。K/V 的参数形状均为 `[H_kv, N_low, ceil(D/G)]`，其中 `G=group_size`。G 必须为 8 的正整数倍，不同 token 不共享量化参数。

第一次超预算压缩的路径是 **原始 FP16 → 16bit 整数编码 → 8bit 整数编码**。INT16 是本次压缩的中间阶段，不作为新增预算事件；未触发压缩的低分区仍为原始 FP16。每个组取 `m=min(x)`、`M=max(x)`：

```text
scale16 = clamp((M-m)/65535, 1e-5, 1e5)
offset  = -m
q16     = clamp(round((x+offset)/scale16), 0, 65535)
```

随后使用 PM-KVQ 原始整数移位公式降位。16→8、8→4、4→2 的 scale 分别乘以 **257、17、5**；offset 不变。反量化为 `x_hat=q*scale-offset`，不保留 FP16 备份，不重新计算范围。整数编码虽为无符号数值，存储容器为 `torch.int16`，解包时通过掩码恢复无符号值。

分组维度、16bit 初始化、scale 截断、整数降位及打包布局均与本地 PM-KVQ 对照验证。适配 ReKV 时保留 FP32 参数计算/存储，读取时恢复 FP16；数值测试以 **FP32 输入的 PM-KVQ 量化器**为参考，使用相同输出 dtype 比较反量化结果。没有接入 PM-KVQ 的 SmoothAttention 重参数化或预算分配。

PM-KVQ 原版要求通道维可被 group size 整除。本后端额外支持尾组：只用有效元素计算 min/max，整数码补齐到 8 的倍数供 2bit 打包，填充值不参与量化范围。模型 head_dim=128、G=128 的常用配置无需填充。


## 缓存与内存口径

`budget_bytes = budget_tokens × 4 × Σ(layer.kv_heads × layer.head_dim)`，包括 K/V 两份、每个 FP16 元素 2 字节。实际计量覆盖管理器中的持久 K/V、量化参数、填充、位置索引、代表向量与元数据；以唯一 tensor storage 的 `nbytes` 加 Python 对象 `sys.getsizeof` 计量，不把逻辑 token 数当成真实压缩存储。

该指标是**受管理缓存字节**，包含 GPU tensor storage 和少量 CPU Python 元数据，不是进程 RSS 或 CUDA allocator 的全部占用。模型权重、summary/问答临时张量、未闭合段非 recent 部分、被问题固定的旧版本属于工作内存。预算保证发生在段提交处理完毕时；中间帧到来后可暂时超过预算。启动时会保守检查 sink/recent 的最低容量。

主缓存和问题快照都留在调用方进程中；K/V、低位载荷、scale/offset、位置索引、代表向量、summary Query 和待总结视觉特征均保留在同一 GPU，压缩、检索、反量化不经过 CPU offload。CPU 只处理输入图片、帧 ID、时间、计数和文本等控制信息。降位和删除通过替换不可变记录完成；问答完成逐层检索后立即记录激活并释放快照，答案继续使用其选出的临时 GPU KV。不创建跨进程模型或缓存副本。检索代表向量使用原始帧 Key 均值；删除部分 token 不改写代表向量。recent 占用 `topk` 名额，最终按时间拼接每层存活 token，再采用 ReKV 的 RoPE 与滑动窗口注意力。

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
| `memory.bits` | 各位宽/已删低分区的“层×帧”记录数 |
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
