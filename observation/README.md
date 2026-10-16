# RVS 本地注意力观察实验

在项目环境中启动：

```bash
conda activate rekv
bash run_observation.sh
# 浏览器打开 http://127.0.0.1:8765
```

服务默认读取 `data/rvs` 的 ego / movie 全部标注，使用本地
`model_zoo/llava-onevision-qwen2-7b-ov-hf`，在 `cuda:0` 上以 FP16 运行。
HTML、CSS、JavaScript 和计算结果均在本地，无 CDN 或前端构建步骤。
需要现有环境中的 torch、transformers 5.5.4、accelerate、numpy、Pillow、decord。
直接双击 HTML 无法执行模型计算，需要上述 Python 服务。

```bash
# 更换端口 / 显卡
CUDA_VISIBLE_DEVICES=1 bash run_observation.sh --port 8766
# 指定 Python，无需激活环境
PYTHON_BIN=/home/yangx/miniconda3/envs/rekv/bin/python bash run_observation.sh
# 远程机器运行时，可通过 SSH 将服务转发到本机
ssh -L 8765:127.0.0.1:8765 user@server
```

页面支持：切换 RVS 分组和视频、查看全部问题与原始标注、设置最大帧数 / 采样 FPS /
起始时间、生成或取消计算、切换观察层、查看逐帧图片 / mass / 熵和 token 热力图、
在原图上定位 token、比较 attention mass 分布、生成所选问题的模型回答、
查看逐帧 K/V 在 token × channel 上的 magnitude 三维曲面与热力图、
下载完整 JSON 和当前层帧统计 CSV。首次打开需点击「生成观察数据」；之后切换样本
默认自动计算，可关闭自动计算选项。参数变更后需点击生成按钮。

## 输入与采样

- 默认：与 ReKV 相同的初始化提示词 + 视频特征，末尾 32 个**视频 token**作为 query 窗口。
  所选问题不进入默认注意力观察输入；完成观察后追加问题生成回答。参考答案只用于对照展示。
- 可选：初始化提示词 + 视频特征 + ReKV 问题模板，取整个输入最后 32 个 token。
  短问题可能导致窗口同时包含视频和文本；可从完整 token 图和 JSON 查看确切范围。
- ego 使用 `range(0, source_frames, int(round(source_fps) / sample_fps))`；
  movie 使用 `np.linspace(0, source_frames - 1, int(source_frames * sample_fps), dtype=int)`，
  与 `ReKVStreamVQA.load_video` 一致。movie 源数组视为 1 FPS，按 mmap 读取。
- 先生成完整视频的采样索引，再保留时间戳 `>= start_time` 的前 `max_frames` 个。
  默认 8 帧，界面上限 128（可用 `--max-frames` 调整）；超过模型上下文会明确报错。
  `start_time` 之前的帧不会编码，起始时间不根据问题区间自动调整。
- 不对处理器二次采样：`do_sample_frames=False`。帧列表与特征提取共用同一份 RGB 数组，
  页面展示源尺寸的 JPEG（quality=95），并记录源帧序号、真实时间戳和 token 区间。
  JPEG 为查看用有损副本；模型接收解码后的 RGB，不从 JPEG 重新读取。
- 视频特征提取遵循仓库 `_get_video_features`，每帧 token 数从实际输出推导；
  不额外插入标准多模态聊天接口的 video newline token。

## 图片 token 映射与 mass 分布

当前处理器直接将每帧缩放为 384×384，不做视频中心裁剪或 padding。
SigLIP 的 patch_size=14，形成 27×27 个 patch，再通过
`interpolate(mode='bilinear', align_corners=False)` 池化为 14×14 个视频 token。
输出 token 按行展开。界面通过实际池化采样坐标将空间网格映射回原图；
每个 token 的归一化坐标、参与池化的 patch 索引和权重保存在 JSON 的 `spatial.cells` 中。

图片中的橙色实线表示 token 空间网格，虚线表示参与池化的 patch 范围。
两者与帧内网格、全局 token 图联动：悬停图片或帧内网格、点击全局 token 图即可定位；
帧内网格支持方向键。坐标读数给出全局 token 序号、行列、原图像素边界、原始 mass 和帧内概率。
patch 网格仅覆盖预处理图像左上 378×378 像素；右侧和底部剩余 6 像素不分配空间锚点。
这是空间映射，不是独立感受野；视觉 Transformer 已在 patch 间混合信息。
对于不支持的裁剪/缩放配置会明确报错，避免给出错误坐标。

- 总 mass 分布条：历史文本、历史视频、观察窗口，三者之和约为 1。
- 逐帧柱状图：可显示原始 mass `M_f`，或历史视频内部占比 `M_f / sum_f M_f`。
  点击柱形定位图片，切换层时同步更新。
- 图片空间分布：默认按原始 token mass 着色，所有帧共用所选层的最大视频 token mass
  作为线性色阶上限；可切换为帧内条件概率并在当前帧内缩放，或关闭叠加 / 调整透明度。
  观察窗口 token 显示灰色，不混入历史评分或帧内熵。

## 逐帧 K/V magnitude

完成观察输入的 prefill 后、追加回答 token 之前，保存原生 Transformers KV cache：
Key 为 **post-RoPE**，Value 为原始缓存值；保留所有原生 KV heads，不按 GQA 重复展开，
不平均层或 head。每帧保存完整视频 token，包括处于末尾观察窗口中的 token，
不包含初始化提示词、问题或生成的回答 token。

本地 7B 模型的每个切片为 196 token × 128 channel，提供 28 层、4 个 KV head。
“逐帧 Key / Value magnitude”面板与所选视频帧联动；其层/head 独立选择，均从 0 编号。
两张三维曲面的横轴为帧内 token、纵轴为 channel、高度为 `abs(K[t,c])` / `abs(V[t,c])`。
默认各自使用标尺，可勾选共用标尺；俯仰角和方位角可切换。
曲面由本地 Matplotlib 生成独立 PNG，使用全部 token/channel 网格，**不做降采样或截断**。
首次切换某个视角需几秒绘图，之后从本地缓存读取；不重新运行模型。

对应的二维热力图保留全部元素：横向 token 从左到右，纵向 channel 从上到下。
悬停查看同一位置的 `|K|` / `|V|`，点击或使用方向键可以联动原图中的 token 空间位置。
面板还显示最大值、平均绝对值和 P99。颜色仅用于展示，不改变数值；magnitude 不是 attention mass。
PNG 可直接下载；“K/V 数据”下载当前帧/层/head 的 NPZ，包含**有符号**的
`key`、`value`（形状 `[token, channel]`）、全局 `token_positions` 和层/帧/head 编号。

快照按层保存为 `kv/layer-XXX.npy`，布局为 `[key_value, frame, kv_head, token, channel]`，
第一维 0=Key、1=Value。FP16 原样保存；BF16 转为可精确表示其数值的 FP32。
JSON 的 `kv_snapshot` 记录维度、dtype、采集阶段、磁盘数据大小和切片 API。
后端通过 mmap 读取当前帧/head，浏览器按需取数据。
7B FP16 模型约每帧占用 10.7 MiB（8 帧约 85.8 MiB）；快照保存在已有结果缓存目录。
旧版本结果不含 K/V 快照，更新服务后重新生成即可。

## 模型回答

每次计算都会对所选问题生成真实模型回答，使用与观察实验**完全相同的已选视频帧**。
默认视频模式先采集窗口注意力，再在该完整 prefill 的 KV cache 上追加 ReKV 问题模板；
问题模式的观察输入已包含问题，直接复用其 prefill。观察钩子在生成前移除，
回答的 query 不会改变之前的窗口统计，也不执行 ReKV 检索或 KV 淘汰。

回答使用贪心解码，默认最多 128 个新 token，可在界面设为 1–512。
遇到 EOS、回答长度上限或上下文上限停止，页面显示相应状态。
JSON `answer` 保存问题、回答文本、生成 token IDs、输入/输出 token 数、停止原因和耗时。
参考答案不输入模型。改变问题会清除旧回答；启用自动计算时会重新生成。

## 注意力与熵口径

这是独立的**完整因果注意力 prefill**观察路径，不调用 ReKV 的 KV 淘汰或检索，
不修改现有评测推理。这样所选输入范围内的历史 token 都能被窗口观察。
使用模型原生 RoPE 和 SDPA 计算隐藏状态，并在语言模型每层截取 post-RoPE Q/K。
只额外构建最后 32 个 query 的注意力行，按组处理 heads；不保存完整 N×N 权重。
完整 prefill 的计算量仍随输入长度近似二次增长，限制帧数可降低计算成本。

令序列长度为 T，窗口 W=[T−32,T)，历史 P=[0,T−32)：

```
A[l,h,q,i] = causal_softmax(Q[l,h,q] K[l,h,i]^T / sqrt(d))
s[l,i] = mean_{h, q in W} A[l,h,q,i], i in P
s[all,i] = mean_l s[l,i]
M[f] = sum_{i in frame f intersect P} s[i]
p[i|f] = s[i] / M[f]
H[f] = -sum_i p[i|f] ln(p[i|f])
H_normalized[f] = H[f] / ln(n_f)
effective_tokens[f] = exp(H[f])
```

Softmax 分母包含因果可见的全部 key（包括窗口内部已可见的 key），随后仅截取 P。
因此历史 token 的 mass **不会被重新归一化到 1**。
JSON 中 `video_mass + text_mass + window_mass ≈ 1`，可用于审计分数。
GQA 根据 query head 对应的 KV head 展开后统计；所有 attention heads 等权。

帧内条件分布消除了帧整体 attention mass 的乘法缩放影响。`H` 单位为 nats；
`H_normalized` 使用参与评分的 token 数，通常范围 [0,1]。
这不能保证不同层 / 不同输入得到相同分布，也不等于模型因果解释。
跨层视图先平均重要性再计算熵，是聚合分布的熵，不是逐 head / 逐层熵的平均。
最后一帧可能仅部分进入历史区；灰色窗口 token 不计分，JSON 记录
`scored_tokens` 和 `window_tokens`。0 mass 或没有历史 token 时熵为 `null`；
仅 1 个历史 token 且 mass>0 时归一化熵记为 0。

参考 [SnapKV 官方实现](https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py)
与 [论文](https://arxiv.org/abs/2404.14469) 的末尾窗口投票。这里对 query 取均值，
与其求和相差常数 32，排名和帧内条件熵相同；不执行 pooling、top-k 选择或压缩。

## 结果与复现

结果保存到 `results/rvs_observation/<fingerprint>/`：每帧 JPEG 与 `result.json`。
JSON 包含完整样本、输入参数、采样策略、token 布局、每层及全部层平均的重要性、
帧统计、空间映射、模型回答、K/V 快照元数据、模型路径 / dtype / 层数 / head 数、软件版本及耗时。
缓存指纹包含参数、数据 / checkpoint 文件路径大小和修改时间、观察代码及关键依赖版本。
相同配置复用结果。服务只运行一个 GPU 计算任务，另允许少量排队任务；切换样本时取消旧任务。
取消在阶段或层边界生效，首次模型加载期间需等待加载结束。模型在服务进程内复用，
退出服务会释放显存。用 `Ctrl+C` 退出。

模型路径、数据根目录和缓存目录可通过 `--model-path` / `--data-root` / `--cache-dir` 设置。
当前适配 LLaVA-OneVision Qwen2（Transformers 5）；不自动下载模型，不支持其他 Video-LLM。

验证核心数学口径和与原生 attention 的一致性：

```bash
python -m unittest discover -s tests -p 'test_observation.py' -v
```
