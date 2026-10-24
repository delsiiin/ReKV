# suprKV：摘要引导的渐进式 KV 缓存压缩

Python 包为 `model.suprkv`，数据集评测入口为 `video_qa.run_eval --method suprkv`。`video_qa.suprkv_stream_vqa` 集中定义参数、数据集 worker 和评测调度，作为分片 worker 的命令行入口；测试与性能基准文件统一使用 `suprkv` 命名。

采用 ReKV 单进程顺序执行方式的单模型、GPU 常驻压缩后端，支持 LLaVA-OneVision Qwen2 checkpoint。每个 worker 顺序处理视频和问题；数据集评测可以按 GPU 分片。原有 ReKV 评测命令不变。

## 运行

### 数据集评测

在 `rekv` 环境、项目根目录执行 `bash run_suprkv.sh`，默认遍历 `rvs_ego` 标注文件中的全部视频及问题。也可以直接运行：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python -m video_qa.run_eval \
  --method suprkv \
  --model llava_ov_7b \
  --dataset rvs_ego \
  --num_chunks 1 \
  --max_samples 0 \
  --sample_fps 0.5 \
  --retrieve_size 64 \
  --budget_tokens 4096 \
  --max_segment_frames 32
```

`--model`、`--dataset`、`--num_chunks`、`--max_samples`、`--sample_fps`、`--retrieve_size`、`--debug` 和 `--only_eval` 复用 ReKV 的含义。`--max_samples 0` 表示标注文件中的全部视频；非零值在分片前全局截取，保留每个视频的全部问题。`--debug true` 则仅处理每个分片的首个视频。

支持 `qaego4d`、`egoschema`、`cgbench`、`mlvu`、`activitynet_qa`、`rvs_ego`、`rvs_movie`，使用 ReKV 相同的默认标注路径和评分脚本。RVS 按标注中的 `end_time` 顺序回答，累计编码采样数组 `video[:int(end_time * sample_fps)]`，每次只编码新增部分；其他数据集编码完整视频后回答全部问题。选择题输出包含 `pred_choice`、`correct_choice`、`qa_acc` 和任务类型；无公开答案时保留空值。

每个 worker 仅加载一次模型，在视频之间调用 `reset()`。`--num_chunks` 是 worker 数量，每个 worker 使用一张 GPU，按 `CUDA_VISIBLE_DEVICES` 中的顺序分配；例如 `CUDA_VISIBLE_DEVICES=2,3` 配合 `--num_chunks 2`。SuPr-KV 目前不支持单模型跨 GPU 切分，72B 模型也必须能放入单张设备。`--model_path` 可覆盖 `--model` 对应的 checkpoint；仅支持 OneVision Qwen2 系列。

`--n_local` 是 ReKV 的局部注意力窗口，不能作为 SuPr-KV 预算使用；SuPr-KV 模式会明确拒绝该参数。使用 `--budget_tokens` 设置压缩缓存预算、`--recent_frames` 设置保护的最近帧数。SuPr-KV 扩展参数同时接受下划线和连字符写法。

默认结果目录为 `results/suprkv/<model>/<dataset>/<retrieve_size>-<sample_fps>/budget_<budget_tokens>/`，限量运行追加 `samples_<max_samples>/`。每个分片保存 `<num_chunks>_<chunk_idx>.csv` 和逐视频缓存统计 `.stats.jsonl`，成功完成全部分片后合并为 `results.csv`，再调用原评分脚本。worker 失败时停止合并和评分。不同实验可用 `--save_dir` 指定独立目录，`--anno_path` 可覆盖默认标注文件。

开放式问答评分沿用原项目的 GPT 评分配置；只生成答案可加 `--skip_eval`（或 `bash run_suprkv.sh --skip_eval`），之后用相同参数加 `--only_eval` 单独评分。新预测合并后会清理该结果目录下旧的评分缓存 `tmp/`，避免复用旧答案的分数。

默认从 Hugging Face 加载 `llava-hf/llava-onevision-qwen2-7b-ov-hf`，首次使用自动下载并缓存。
可用 `--model_path` 指定其他 OneVision 模型 ID 或本地目录，用 `HF_HOME` 指定模型缓存目录。ReKV 与 SuPr-KV 共用 `video_qa/sampling.py`，采用原 ReKV 的采样规则：NPY 视为每秒一帧，使用 `np.linspace(0, 总帧数 - 1, int(总帧数 × sample_fps), dtype=int)`；MP4 使用 `int(round(原视频平均FPS) / sample_fps)` 的固定整数步长。这与 observation 在开始时间过滤、帧数限制之前的正常采样结果一致。MP4 时间戳仍由帧索引除以未取整的平均 FPS 估算，并非逐帧容器时间戳。两种方法的 RVS 问答均恢复原 ReKV 的 `int(end_time * sample_fps)` 数组切片边界，右边界不包含在内，超出视频长度时自然截断。ReKV 按问题批量编码新增帧，SuPr-KV 逐帧编码；重复边界不重复编码。SuPr-KV 回答时使用已编码帧的缓存快照，不再按标注时间做第二次过滤，因为数组切片选中的源帧时间可能等于或超过 `end_time`。

### Worker 运行时接口

模型在 worker 进程中加载一次，视频编码、段总结和问答使用同一个对象顺序调用。每个问题都回答完毕后再编码后续帧。worker 内没有额外的推理子进程、IPC 队列或异步解码调度；默认使用分配给该 worker 的 GPU，其逻辑编号为 `cuda:0`。

```python
from model.suprkv.runtime import StreamingConfig, StreamingRuntime

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

正确性测试：`python -m unittest discover -s tests -p 'test_suprkv_accounting.py'`。可用下面的合成基准比较管理路径；`--baseline` 接受修改前保存的 `model/suprkv/cache.py`，省略时仅测当前实现。默认每帧 8 个 token、28 层，主要复现对象数量；`--replay` 额外计时逐帧入库、token 评分、每两帧提交及报告。测量不包含模型前向、summary 生成或量化，不能作为整模型加速比例。

```bash
python tests/benchmark_suprkv_accounting.py \
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

### 压缩 KV 的 CUDA 融合路径

`packed_kernels.py` 参考 KIVI 的按需解包、反量化后直接乘加思路，用 Triton 编译 CUDA kernel；无需编译或安装 `kivi_gemv`。它保持本项目的 INT16 容器、MSB-first 位序、FP32 scale/offset、`code * scale - offset` 和逐级 16→8→4→2 位降精度。K 按 channel 分组，V 按 token 分组，尾组保留原有参数。反量化先舍入到模型 dtype，再参与乘法。

CUDA 上的量化、降位、反量化默认使用 Triton；CPU 或不支持的输入保留 PyTorch 路径。`backend='torch'` 可强制数值参考实现，`backend='triton'` 可显式要求 GPU kernel。新增 `packed.matmul(left, transpose=True)` 计算 QK，`transpose=False` 计算 PV，支持 GQA 和 FP16/BF16/FP32；短 query 使用 split-K GEMV，多行 query 使用分块 GEMM，不创建完整浮点 KV。单独调用这些接口不会改变 attention 后端。

量化、降位和独立反量化的 token 数、打包长度、参数组数及输入步长使用运行时参数，并关闭这些参数的值/对齐特化。逐帧、逐层高分区长度变化会复用同一个 kernel，不再为每种长度编译一份。dtype、位宽、分组大小等格式仍可以产生不同 kernel。首次压缩直接调用 `quantize(..., bits=8)`：kernel 内先计算 INT16 code，再执行完全相同的整数降位算式，直接写出 INT8 载荷；不分配中间 INT16 缓存。后续降位同时写出新的 code 和 scale。CPU/PyTorch 参考仍逐级执行，量化数值、尾组规则及缓存预算策略保持一致。

变长形状的性能回归需计入首次编译，不能只测预热后的固定矩阵。可从 observation 结果中提取每层实际保留长度，回放首次量化、后续降位及反量化；`--cache-dir` 指向新的目录时包含冷编译，脚本也报告每阶段的 kernel 变体数。此回放使用随机 BF16 KV，重放最终量化帧层的形状和位宽，不重放视频模型或全部预算事件。

```bash
python tests/benchmark_suprkv_quantization.py \
  --observation-result results/rvs_observation/RUN_ID/result.json \
  --cache-dir /tmp/rekv-quantization-fresh-cache
python -m unittest discover -s tests -p 'test_suprkv_quantization_dispatch.py' -v
```

一次已保存 observation 的 3,304 个量化帧层包含 42 种长度（70–114 token）。RTX 4090 上用独立冷缓存回放：修复前首次 INT8 量化 23.43 秒、继续降位 4.58 秒、反量化 17.56 秒；修复后分别为 2.76、0.73、1.11 秒。量化/降位/反量化变体由 84/83/156 降为 1/1/4；重复回放不新增变体。该数据用于定位并验证编译回归，不代表完整模型加速倍数。

也通过 observation 原服务复跑了保存结果 `ba9c334156f75e33ae3f3f8b` 的完整参数：`ego:0`、150 帧、0.5 FPS、BF16、预算 8192、top-k 32、9 个问题。复跑在 RTX PRO 5000 72GB 上使用新的 Triton 缓存目录，包含首次编译时间。与原保存结果相比，量化阶段从 31.65 秒降到 2.95 秒，KV 读取/反量化/拼接从 21.40 秒降到 2.74 秒。所有回答文本、检索帧、整个时间线的帧层状态及持久字节数（449,224,164）完全相同。历史运行与复跑的其他阶段也有时延变化，不能将完整运行时差全部归因于本次 kernel 修改。

完整问答融合路径需要显式启用：

```bash
python -m video_qa.run_eval --method suprkv --dataset rvs_ego --packed_attention
```

Python 接口使用 `StreamingConfig(packed_attention=True)`；直接注入模型的 `OneVisionEngine` 配置字典也接受 `packed_attention=True`。默认值为 `False`，因为已驻留浮点 KV 的连续 decode 仍可能比压缩路径更快。

启用后，检索按位宽合并不可变缓存的副本，只复制压缩码和参数，不重新量化。K 的帧尾补齐到 group 边界，补齐位置不参与 softmax。问答缓存持有这些压缩副本、位置/旋转参数及一个浮点生成尾部；首次查询文本仍不写回视频缓存。融合 kernel 完成反量化、Key 的 RoPE 位置差旋转、QK、在线 softmax 和 PV，并通过 split-K 合并多个分块结果，不分配完整浮点 KV 或完整 attention logits。混合位宽、未压缩 low/sink、GQA 和帧内删 token 均保留原位置语义。

这条路径用于单 batch、head_dim ≤ 256、CUDA/Triton 的推理。问答总长度在 local window 内时保持压缩；超出 window 时恢复已有 sliding-window/sink 路径。CPU、训练和缺少 Triton 的情况使用原路径。默认问答 window 已包含检索 token、生成 token 和额外余量。分块归约与原 attention 的浮点运算顺序不同，验证使用 dtype 对应误差容限，不要求 attention 输出逐位相同。

合成基准（不加载模型，不包含 JIT；同步后的主机延迟中位数，峰值为该次调用相对已驻留输入的新增 PyTorch 分配）：

```bash
python tests/benchmark_suprkv_packed.py --device cuda:1 --tokens 4096 --frames 16 --bits 4 --repeats 20
python -m unittest discover -s tests -p 'test_suprkv_packed_attention.py' -v
python -m unittest discover -s tests -p 'test_suprkv_quantization.py' -v
```

本机 RTX 4090，PyTorch 2.10.0+cu130，FP16 输入、4-bit KV、16 个 Q head / 4 个 KV head、head_dim=128、4096 个缓存 token / 16 个块、单 token query，预热 3 次后取 20 次中位数：

| 操作 | 参考路径 | 融合路径 |
| --- | ---: | ---: |
| QK（参考包含 PyTorch 解包和 GQA 展开） | 0.377 ms | 0.149 ms |
| PV（参考包含 PyTorch 解包和 GQA 展开） | 0.341 ms | 0.127 ms |
| 准备缓存并执行一次 attention | 2.399 ms | 0.882 ms |
| 上一行的新增分配峰值 | 44.000 MiB | 4.425 MiB |
| 缓存已准备后的单次 attention | 0.269 ms | 0.646 ms |

attention 参考使用已经加速的 Triton 反量化，拼接后一次性旋转 K，再运行现有 flash attention；融合路径包含压缩块合并。两者共用预先构造的输入和位置表。这里完整压缩 attention 的收益主要是首次准备和工作显存，持续 decode 仍有代价，因此保持可选。16-token query 的 PV GEMM 在同次测试中为 0.349 ms（参考）和 0.355 ms（融合），并非所有形状都会加速。

尾组的准备成本也不可忽略：改成 16 块 × 196 token（`--tokens 3136 --frames 16 --bits 4 --queries 1`）后，准备加 attention 为 2.346 ms（dense）与 3.785 ms（packed），新增峰值分别为 33.781 MiB 与 8.438 MiB；缓存已准备后的单次 attention 为 0.240 ms 与 0.625 ms。当前补齐与复制由 PyTorch 发起，196-token 帧的这组测试没有延迟收益，主要节省工作显存，不能直接套用 256-token 整组块的加速比例。

基准区分「反量化后 attention」「准备压缩副本后 attention」和两种已经准备好的 decode 路径，避免把一次性反量化成本计入每个 decode token。它不包含帧检索评分、输入/位置表构造、Python 模型前向、真实视频或完整回答，不能作为端到端速度结论。压缩块合并会暂时增加压缩副本和尾组 padding，实际收益与帧数、保留 token 数、分组大小及生成长度有关。

闭段评分按层批量处理同段各帧：每层 summary Q 的 FP32 转换、文本 token 均值和 GQA 分组均值只计算一次，Q/K 有限值检查合并为每层一次主机同步。单帧和批量评分都在 FP32 中聚合、点积和计算帧内均值，评分内部禁用 autocast，保留 RoPE 前 K 和原有 GQA 映射；持久 KV 精度不受影响。各帧仍使用自己的分数均值划分高低分区。分类先计算高低分位置，再复用索引提取 K/V；索引提取产生独立 storage，不再额外 clone。任一层评分失败仍回退整个段，并释放评分 Key；批量临时张量只保留一层，预算执行前释放。相较此前 FP64，极接近阈值的 token 可能因舍入改变分区。

`test_suprkv_scoring.py` 对照逐帧评分，覆盖 CPU/CUDA、FP16/BF16/FP32/FP64 输入统一输出 FP32、1/2/8/32 帧、均值相等的 token、非有限输入、整段失败回退、快照与账本；另用显式 GQA 展开的 FP64 参考验证数值误差，并检查 autocast 下仍输出 FP32。数值对照采用 `rtol=1e-5, atol=1e-6`。闭段微基准可以使用修改前保存的 `cache.py` 对比：

```bash
python tests/benchmark_suprkv_scoring.py --baseline /tmp/rekv-scoring-cache-before.py
```

以下为切换 FP32 前的历史 FP64 评分基准，不代表当前 FP32 版本的耗时。RTX 4090、28 层、每帧 196 token、4 个 KV head、head_dim=128、BF16 缓存、32 个 summary token；每组预热 2 次，交替执行新旧版本，取 5 次中位数，计时前后同步 CUDA：

| 段长 | 原逐帧闭段 | 按层批量闭段 |
| --- | ---: | ---: |
| 1 帧 | 36.26 ms | 28.90 ms |
| 2 帧 | 62.60 ms | 35.71 ms |
| 4 帧 | 110.91 ms | 33.52 ms |
| 8 帧 | 141.24 ms | 55.47 ms |
| 32 帧 | 577.82 ms | 225.10 ms |

该基准包含评分、分区和缓存登记，不包含模型前向、summary 生成或量化，不能作为端到端加速比例；不同段长的独立测量也会受系统调度影响。批量处理会增加临时显存：本次 32 帧闭段的峰值增量相比原实现多约 42 MiB，持久缓存内容不变。省略 `--baseline` 仅测当前版本。

```bash
python -m unittest discover -s tests -p 'test_suprkv_*.py' -v
python -m unittest discover -s tests -v
```

测试包含 observation 数值对照、真实位打包、逐级预算策略、保护区逐值一致性、层间变长缓存、小型 Qwen 前向、同进程顺序事件、单次模型构造计数、reset 不重载模型、CUDA 缓存禁止 offload、编码/summary 后切回问答的一致性、激活去重和异常清理。真实模型短视频结果保存在运行时指定的 `--output` 中。

## 与原 ReKV 的关系

原 `video_qa/base.py` 在每个评测实例中调用一次模型加载，`ReKVStreamVQA` 依次调用同一个对象的 `encode_video()` 和 `question_answering()`；原缓存管理器使用 CPU offload。README 描述了跨进程/跨 GPU 的分离能力，但当前流式评测入口没有单份模型的跨进程并发前向实现。新增后端不修改原缓存管理器，通过 `run_eval --method suprkv` 选择；ReKV 使用默认方法或显式指定 `--method rekv`。

新增后端同样在调用方进程中加载一次模型，按 `append_frame → ask（完成回答）→ append_frame` 顺序执行。与原 ReKV 的主要区别是新增后端将压缩缓存保留在 GPU，按预算执行量化和淘汰。旧版新增后端的 `ask_async()`、延迟取结果的 `answer()`、队列和超时配置已移除；改用同步 `ask()`。

模型直接加载时默认使用 FP16；observation 网页注入的模型可使用 FP16、BF16 或 FP32。sink、recent、评分暂存与未压缩帧保留模型 dtype，反量化恢复同一 dtype。FP32 使用 PyTorch attention，CUDA 上的 FP16/BF16 使用现有 Triton attention。预算仍以 FP16-token 等价字节定义，FP32 张量按实际 4 字节计入账本。`bits`/`bit_order` 中的兼容键 `16` 表示未量化状态，真实浮点精度由 `cache_dtype` 或 `full_precision_dtype` 指明。
