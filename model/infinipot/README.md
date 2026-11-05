# InfiniPot-V → `infinipot`

本适配以仓库内 `InfiniPot-V/qwen_inference_ovu.py` 和
`InfiniPot-V/kvcache_utils.py` 为执行规范。源 README 说明该发布代码是研究复现，
未包含论文的全部组件；这里复现发布代码的行为。

## 执行流程

```mermaid
flowchart TD
    A[现有 JSON 标注 / 视频或 NPY] --> B[取完整视频或问题可见前缀]
    B --> C[Qwen 均匀采样、缩放、processor、重建 video_pad]
    C --> D{时间网格数大于 block_size 且启用压缩?}
    D -- 否 --> E[原生 model.generate，300 tokens]
    D -- 是 --> F[计算完整问题序列的源版本 3D RoPE]
    F --> G[首块 32 个时间网格；后续每次 8 个]
    G --> H[当前块视觉编码 → LLM prefill → 更新 KV]
    H --> I{末块?}
    I -- 否 --> J[按剩余长度确定保留量 → 每层每头 TaR + VaN]
    J --> G
    I -- 是 --> K[输入问题后缀 → greedy 自回归，最多 300 tokens]
    E --> L[现有 CSV 与评分入口]
    K --> L
```

1. **输入与单位。** 视频使用 Qwen 的均匀帧采样，默认 2 FPS，最多 768 张原始帧。
   空间缩放设 `max_pixels=128*28*28=100352`。Qwen 的时间 patch 合并相邻两帧；
   源程序的 `frame_number`、`block_size`、`compress_frame_num` 实际均以
   `video_grid_thw[0,0]` 的时间网格为单位。因此 32 个网格通常对应 64 张原始帧。
   token/网格由真实 grid 决定，不能假设固定 128 或 196。
2. **分块。** 首块包含系统前缀（含 vision_start）与 32 个网格；后续仅输入
   `32-24=8` 个新网格。每块独立执行视觉塔，再执行语言模型前向。
   问题后缀不参与视觉 prefill，但计算完整序列的位置时包含问题后缀。
3. **动态尾部预算。** 非末块保留量为 `max(24, 32-剩余网格数)`。
   最后不足 8 个网格时，前一块会多保留一些 KV；末块完全不压缩。
   `compress_frame_num` 是**保留量**，不是删除量。
4. **TaR。** 每层每 KV head 独立选择。将缓存中视觉 K 按当前逻辑网格重排，
   最近 `int(current_frame_num*query_ratio)` 个网格充当 query；对归一化 K 的
   同一空间位置计算相似度，沿 query 网格求均值并取负，选取与近期不同的历史 token。
   TaR 的预算为 `keep_tokens-round((1-tar_ratio)*keep_tokens)`，其中先扣除
   必须保留的全部 query tokens，再对历史做 top-k。这里使用的是已施加 RoPE 的缓存 K。
5. **VaN。** 使用 V 的 L2 norm。可选逐层 3D/2D/1D/不池化（核大小 7/5/3/1），
   默认关闭。将 TaR 选中的位置赋值为全局 `max(score)+1`，再做保留预算大小的
   top-k，按原索引排序后同时 gather K/V。系统 token 始终保留。
   压缩后 token 不再构成完整物理帧，但下一轮仍按 `token_per_frame` 分组；这是源实现的行为。
6. **位置与生成。** 每块使用完整序列对应的原始 3D position IDs，压缩后不重新旋转 K。
   问题生成起点保留源代码的 `position_ids_full[0,0,-1]+1`：它位于完整 prompt
   后面，而不是最后一个视觉 token 后面。分块分支 greedy argmax，EOS 停止，最多 300 tokens；
   非分块分支调用原生 `generate`，保留 checkpoint 的生成配置。
7. **问题隔离。** 每个问题重新建立 DynamicCache，不继承前一问题的 KV。
   RVS 对每个可见前缀重放上述流程，因为尾块保留量依赖该前缀的总长度；
   不把它改造成持续提交、无法回退的增量压缩。

## 参数来源

默认采用发布的 `scripts/run_sample.sh` 与 README 的 InfiniPot-V 实验配置。
源 Python parser 本身默认关闭分块，是不同的 baseline 配置；两者在此明确区分。

| 参数 | `infinipot` / 发布实验脚本 | 源 Python parser |
|---|---:|---:|
| `block_size` | 32 | -1 |
| `compress_frame_num` | 24 | 0 |
| `compression_method` | infinipot-v | uniform |
| `max_frames_num` | 768 | 32 |
| `tar_ratio` | 0.5 | 0.5 |
| `query_ratio` | 0.25 | 0.25 |
| `adaptive_pooling` | false | false |
| `per_frame` | false | false |
| 生成上限 | 300 | 类常量 300 |
| 解码采样 FPS | 2 | qwen-vl-utils 默认 2 |
| 视频 `max_pixels` | 100352 | 类常量 100352 |

`per_frame` 在源压缩函数中没有使用，这里同样不改变 token 选择。
源 `use_block_processing` 仅用于打印，实际分支条件是 `block_size>0`、
`frame_count>block_size`、`compress_frame_num>0`，本适配保留实际条件。
没有沿用源程序硬编码的 MLVU `.pt` dump 路径；直接读取现有视频，避免依赖另一份数据布局。

显式传 `--block_size -1 --compress_frame_num 0 --compression_method uniform --max_frames_num 32`
即可使用源 parser 的 baseline。`--max_new_tokens` 是可选的显式实验覆盖；
默认情况下，BaseVQA 的 16/256/1024-token 限制不会覆盖源算法的 300。
无效块大小、TaR 查询预算会报错，不通过截断 top-k 等方式偷偷修改算法。

## 数据集与模型

| 路由 | 现有数据集 | 输入 |
|---|---|---|
| offline | mlvu、qaego4d、egoschema、activitynet_qa、cgbench | 完整视频，依次回答 conversations |
| stream | rvs_ego、rvs_movie | 每个问题 `end_time` 前的视频前缀 |

沿用 `run_eval.DATASETS` 的标注路径、样本限制、分块调度、CSV 字段与评分器。
选择题保留源 MLVU/EgoSchema 提示词，并扩展到 CGBench 的 A–H 选项；开放题直接使用原问题。
EgoSchema 私有答案仍遵循现有仓库的占位约定，最终评测以其 submission 为准。

普通视频遵循 qwen-vl-utils 的 `smart_nframes` + rounded linspace，而不是 ReKV 的固定步长。
源依赖未锁定 qwen-vl-utils 版本，本适配固定采用 **0.0.14** 的帧数及 bicubic resize 行为，
不额外安装该包。`--sample_fps` 可覆盖默认 2；其他方法仍默认 1。
NPY 按仓库约定视为 1-FPS RGB 数组，先截取前缀，再按相同规则采样，不人为补足到 2 FPS。
RVS 保留 source timestamp `< end_time` 的帧，不读取未来；`start_time` 不用于裁剪历史，
与仓库的前缀问答任务一致。此边界采用真实时间，不是 ReKV 的采样数组切片。
零帧前缀执行纯文本问答；单帧前缀重复该帧构成时间 patch。

支持当前模型表中的 `qwen2_5_vl_3b`、`qwen2_5_vl_7b`、`qwen2_5_vl_72b`。
源项目还支持 Qwen2-VL，但当前仓库没有其 preset；LLaVA 不具备源算法依赖的 Qwen
视频接口与 MRoPE，故不宣称支持。加载保留 `dtype='auto'`、`device_map='auto'`、
`attn_implementation='flash_attention_2'`。

```bash
# 发布实验配置；可覆盖 dataset、模型、样本上限、参数
bash run_infinipot.sh --max_samples 5
bash run_infinipot.sh --dataset rvs_movie --max_samples 1

# 统一调度入口
python -m video_qa.run_eval --method infinipot --model qwen2_5_vl_7b \
    --dataset egoschema --max_samples 5

# 仅预测，不触发开放题评分服务
python -m video_qa.infinipot_stream_vqa --model qwen2_5_vl_7b \
    --anno_path data/rvs/ego/ego4d_oe.json --save_dir results/infinipot_rvs \
    --max_samples 1
```

结果目录记录压缩、帧数、生成参数及 FPS。CSV 附加同样的配置字段。

## 兼容性与验证

源版本要求 Transformers 4.57.3；当前仓库固定 5.5.4。
`cache.py` 原样复制源压缩函数；`core.py` 保留源分块和生成函数，只对视觉特征的
`pooler_output` 封装及 RoPE 调用做兼容处理。
`rope.py` 保留 Transformers 4.57.3 的 `get_rope_index`，避免新版在视频后的文本位置
计算差异。runtime 仅绑定自身模型实例的位置计算函数，关闭时还原；无全局 attention patch。
源非分块分支的未定义全局 `MAX_GEN_TOKENS` 修正为类常量引用，否则短视频会直接报错。

```bash
PYTHONPATH=tests python -m unittest test_infinipot -v
```

测试覆盖源函数逐层逐头 K/V 差分、三种压缩方式与池化组合、真实随机小 Qwen 的
33/40/41/49 网格块边界和生成位置、短视频原生分支、问题隔离、NPY 前缀、全部 7 个数据集
路由及 CSV 合约。测试使用本地 `InfiniPot-V/` 作为 oracle，源目录缺失时仅跳过差分部分。
另已用源 sample MP4 对比 qwen-vl-utils 0.0.14，16 帧采样及缩放张量逐元素一致。

2026-11-05 复查补充了独立的跨版本运行对照：

```bash
# reference-path 指向独立的 Transformers 4.57.3 + 兼容依赖目录，
# 不修改当前环境；本脚本不下载依赖或预训练权重。
python tests/check_infinipot_versions.py --reference-path /path/to/transformers457
```

相同随机小 Qwen 权重，在 4.57.3 中运行源类，在 5.5.4 中运行适配类；
覆盖 2/32/33/41/49 个时间网格。CPU FP32/eager 条件下，processor 的像素与 token 张量、
完整位置、每次 forward 的 K/V 和 logits，以及生成结果全部一致，最大绝对差为 **0**。
该对照直接执行源预处理与生成函数；仅给源非分块分支补上缺失的生成上限全局常量，
并将双方生成长度缩短到 3 tokens。源文件未修改。

复查还修正了两处适配边界：

- 答案提取按真实选项数保持源 A–D/A–E 正则；此前统一 A–H 会把四选题中的
  `Final answer: D` 误取成 F。五选题的空白/括号处理也保留源行为。
- TaR 预检查纳入实际视频长度和动态尾部预算。例如 `block_size=32`、
  `compress_frame_num=1`、33 个网格时，源程序实际保留 31 个网格，不能按保留 1 个误拒绝。
  TaR 关闭时允许源程序未使用的 `query_ratio=0`。

当前执行环境无可用 CUDA，未完成预训练 Qwen GPU 推理或数据集准确率测量；
上述结果验证了代码流程及小模型的跨版本一致性，不等同于预训练模型在 GPU 上的
逐 bit 输出一致或论文指标复现。RVS/NPY、零/单帧前缀及更多选项是数据集适配扩展；
源项目不含这些任务入口。默认参数对应发布实验脚本，而非默认关闭压缩的源 Python parser。
