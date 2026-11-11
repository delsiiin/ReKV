# StreamKV 源码执行逻辑移植

`--method streamkv` 默认按提供的 `StreamKV/` 源代码执行。之前适配版的精确预算分配、
重新计算全部相邻相似度、增量视频缓存和同步 CPU KV 管理器已移除。
生产推理与源代码一样需要 CUDA；当前模型 API 适配针对 Transformers 5.5.4 的 LLaVA-OneVision。
新增可选 `--budget_tokens` 历史 KV 总预算模式，具体语义见下文；未设置时仍使用原比例压缩。

## 源码对应关系

| 当前文件 | 对照源文件 | 改动范围 |
| --- | --- | --- |
| `abstract_streamkv.py` | `StreamKV/model/abstract_streamkv.py` | 文件内容原样保留 |
| `cache.py` | `StreamKV/model/attention/kv_cache_manager.py` | 仅更换注意力工具的导入路径 |
| `attention.py` | `StreamKV/model/attention/streamkv_attention.py` | 更换导入路径，允许注入预算模式缓存类；默认类不变 |
| `generation.py` | `StreamKV/model/llava_onevision_streamkv.py` | 原样提取 `question_answering` 方法 |
| `runtime.py` | 原 OneVision 模型/patch 接口 | 模型 API 适配及可选预算模式入口 |
| `budget.py` | 新增实验变体 | 对全部历史帧块引导评分、按预算保留并重建检索索引 |
| `video_qa/streamkv_sampling.py` | 原 `StreamKVStreamVQA.load_video` | 原样提取视频/NPY/帧目录采样方法 |
| `video_qa/streamkv_stream_vqa.py` | 原流式评测 | 相同采样前缀复用视频归档；前缀变化时重算；记录写入当前 BaseVQA |
| `video_qa/streamkv_offline_vqa.py` | 原离线评测 | 完整视频一次编码，然后依次回答 |

`selection.py`、`segmentation.py` 只导出上述源码函数，不再包含替代算法。
注意力的 Torch/Triton 计算工具复用 `model.rekv` 中与提供的源代码相同的实现。

## 执行逻辑

1. 编码初始提示，创建每层 `ContextManager`。
2. `encode_video` 每次独立分段，并在返回前编码尾段。均匀分段时整段批量调用视觉编码器；
   语义分段时逐帧编码、使用原始 dtype 的帧均值计算余弦相似度。
3. 语义段超过代表帧上限时，按源代码维护的 `cur_chunk_cossim` 选择合并对象。
   保留 `past_frame_feature` 和局部相似度更新规则，不重新计算所有相邻边。
4. 可选摘要逐视觉 token 位置对段内帧取平均，追加一个完整帧块。
5. 压缩开启时将引导提示接在视频后：引导提示 Q 与帧均值 K 点积评分，
   使用原 `obtain_cdf_num` 和 `rescale_sum_match` 分配各层预算。
   保留二分搜索误差阈值、原取整/clip/修正顺序及等值数组的提前返回行为。
6. 每层选高分帧，摘要额外保留；只压缩历史归档，编码局部窗口按源码维护。
   引导提示 KV 不写入历史。CPU pinned memory、CUDA 全局流、LRU 和 128 块池均恢复。
7. 问题先按基础预算检索并收集各层分数，再按原函数分配预算、重新检索。
   检索上界公式保留 `min(retrieve_size*(1+retrieve_temp), 128)`，不额外修正原函数输出。
   Q 先在原 dtype 求均值，再转 FP32 点积。
8. 完整问题提示使用检索结果进行贪心生成，保留原 `topk(2)` 取第一个 token 的行为、EOS 和缓存分支。

**流式评测复用当前采样前缀的视频归档。** 首题或实际截取帧数变化时执行
`clear_cache → encode_init_prompt → encode_video(video[:cutoff])`；相同前缀的后续问题
跳过视觉编码、分段、视频 prefill 和压缩，但仍逐题执行检索预算分配、检索和生成。
时间乘法保留原 `torch.tensor([start_time, end_time]) * sample_fps` 的运算和整数截断，
按切片后的实际帧数判断是否相同，因此不同时间戳也可能命中同一前缀。
前缀增长、缩短或切换视频都会重算，不把旧尾段当成可直接追加的增量状态。
仅保留当前视频归档；问答沿用独立的检索/生成 KV，不深拷贝整份历史归档，不新增一份
视频 KV 快照。检索的 LRU 缓存可以在问题之间保持热状态，仍受原块池上限约束。
视频处理结束或编码/问答异常时释放缓存；异常会向上传播，重新执行时从新缓存开始。
当前 ReKV 数据集若省略 `start_time`，使用 0；源码的普通流式切片本来也始终从视频开头开始。

## 参数

Python CLI 默认值与参考 `video_qa/run_eval.py` / `video_qa/base.py` 一致；
`run_streamkv.sh` 则显式设置参考 `run_eval.sh` 的算法参数，数据集选用当前仓库的 `rvs_movie`：

| 参数 | Python CLI 默认 | 启动脚本 |
| --- | --- | --- |
| `segment_mode` | `uniform` | `semantic` |
| `min_segment_size` | 8 | 4 |
| `max_segment_size` | 64 | 64 |
| `segment_theta` | 0.95 | 0.99 |
| `compression_ratio` | 1 | 0.6 |
| `budget_tokens` | None（关闭） | 未设置 |
| `compress_temp` | 0 | 1 |
| `retrieve_temp` | 0 | 0.5 |
| `n_local` | 15000 | 15000 |
| `retrieve_size` | 64 | 64 |
| `retrieve_local_size` | 8 | 8 |
| `use_segment_summary` / `retrieve_local` | True / True | True / True |
| `retrieve_chunk_size` | 1 | 1 |
| `sample_fps` | 1 | 0.5 |
| `num_chunks` | 1 | 8 |
| 数据集 | 显式指定 | `rvs_movie` |

模型默认 `llava_ov_7b`，支持三个源代码 OneVision 预设。模型保持 FP16，通过共享加载器固定到进程内 `cuda:0`；eval 用 `--gpus` 或 `CUDA_VISIBLE_DEVICES` 为每个进程分配一张 GPU，72B 也使用同一规则。
`n_init`、`n_encode` 来自原提示的 token 数；`block_size=exc_block_size=196`、
`max_cached_block=128`、`fattn=True`、`pin_memory=True`、`async_global_stream=True`。
不再提前要求 `n_local` 容纳最大段及摘要；保留源码对实际段和执行块的断言。

### 按 budget 压缩

```bash
bash run_streamkv.sh --budget_tokens 8192
python -m video_qa.run_eval --method streamkv --dataset mlvu --budget-tokens 8192
```

Python 接口为 `StreamKVConfig(budget_tokens=8192)`，传给 `StreamKVRuntime.load(..., config)`。
设置预算后覆盖 `compression_ratio` 和 `compress_temp`，包括脚本中显式给出的值。

预算限制**每层整个视频的历史 KV 归档 token 数**，不是每段配额或检索配额。
完整帧块为 196 tokens，预算向下取整：8192 最多保留 41 块，即每层 8036 tokens；
预算至少为一块。段摘要同样占一块并参与评分，不保证永久保留。
每段完成后，用当前引导提示 Q 对全部历史块和新块统一评分，逐层保留最高分的预算内块，
再恢复时间顺序并更新 CPU KV、代表向量、GPU 缓存索引与段映射。
各层都遵守同一个上限，预算模式不使用原比例模式的跨层自适应配额。

这个上限在每段编码结束后成立。初始提示、`n_local` 编码窗口、GPU 工作缓冲区、
问题生成缓存及编码当前段的临时峰值不计入预算，因此它不是进程总内存或显存上限。
流式评测同样复用相同采样前缀，变化时重算。预算值写入结果元数据和目录的 `_budget<N>` 后缀。

## 运行

```bash
# 原启动脚本的算法参数，评测 rvs_movie，默认 8 个 worker
bash run_streamkv.sh

# 单 GPU，使用当前仓库的 RVS 标注路径
bash run_streamkv.sh --num_chunks 1 --max_samples 5

# 切换到离线问答，保留脚本的压缩和分段配置
bash run_streamkv.sh --dataset mlvu --num_chunks 1 --max_samples 5

# 直接调用 CLI 使用原 CLI 默认值：uniform、无压缩
python -m video_qa.run_eval --method streamkv --dataset mlvu --max_samples 5
```

数据集入口与其他方法共用 `video_qa/run_eval.py` 的注册表：

| 评测模式 | `--dataset` 可选值 |
| --- | --- |
| 离线问答 | `mlvu`、`qaego4d`、`egoschema`、`activitynet_qa`、`cgbench` |
| 流式问答 | `rvs_ego`、`rvs_movie` |

标注路径与已有方法一致，视频路径按标注本身解析。没有方法专属的数据集注册或 `--data_root` 参数。

结果目录为 `results/streamkv/source/<model>/<dataset>/<settings>/fps_<fps>/`，
避免与先前不同算法行为的结果混用。模型路径表、任务划分、CSV 保存和 worker 调度沿用当前仓库；
它们不参与分段、预算分配或注意力计算。`--model_path` 可显式指定同一份权重以进行对照。

## 验证与边界

```bash
python -m unittest discover -s tests -p 'test_streamkv.py' -v
python -m unittest discover -s tests -p 'test_streamkv_budget.py' -v
PYTHONPATH=tests python -m unittest test_streamkv_reuse -v
```

测试直接读取 `StreamKV/`：检查核心源码一致性、原函数差分、FP16/FP32 分段、
视觉编码批次边界、FP16 Q 均值、逐层预算、KV 块、问题生成、前缀重算、
两套默认参数以及调度/采样。
`test_streamkv_budget.py` 不依赖上游源码目录，覆盖预算参数传递、连续段压缩、摘要计费、
帧块取整、全历史评分、缓存索引重映射、压缩后问答以及默认比例路径兼容性。
`test_streamkv_reuse.py` 也不依赖上游目录，覆盖重复/舍入后相同/越界截断/缩短前缀、
视频边界和异常释放。真实小模型对照原逐题重算入口，在均匀/语义分段、无压缩/比例压缩/
固定预算、摘要开关的组合中检查检索索引、逐次 logits、答案及视频归档张量不变。
小型随机 OneVision 模型测试使用仅存在于测试中的 CPU CUDA-plumbing 替身：
仅替换设备断言、分配和 stream/event 操作，源算法及缓存逻辑保持不变。
另有无此替身的 CUDA 对照测试，在无 GPU 时跳过。没有真实权重跨旧版/新版 Transformers
的端到端精度结果，不能据此宣称重现原论文分数或原硬件吞吐。

源码中的退化行为也保留：例如常量/单元素分数的标准差、空语义段异常、预算不精确等。
没有为它们安装替代分配或空视频问答逻辑；后续如需修正，应另设明确的实验变体。
