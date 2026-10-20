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
起始时间 / 视频末尾 token 数 N、生成或取消计算、切换观察层、查看逐帧图片 / mass / 熵和 token 热力图、
在原图上定位 token、比较 attention mass 分布、生成所选问题的模型回答、
查看选中帧全部 token 在问题前 / 问题后及其差值的热图与原图叠加、
查看逐帧 K/V 在 token × channel 上的 magnitude 三维曲面与热力图、
查看 RoPE 前/后逐帧 K/V 最近邻冗余比例的层/head 分布、
查看每帧 K/V 最近邻距离空间热图与原图叠加、
下载完整 JSON 和当前层帧统计 CSV。首次打开需点击「生成观察数据」；之后切换样本
默认自动计算，可关闭自动计算选项。参数变更后需点击生成按钮。

## 输入与采样

- 默认：与 ReKV 相同的初始化提示词 + 视频特征，末尾 N 个**视频 token**作为 query 窗口，N 默认 32。
  可在页面“视频末尾 token 数 N”修改；窗口为整个视频序列的连续后缀，允许跨帧。
  N 必须是正整数且不超过实际视频 token 总数，超过时报错，不自动截断。
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

## 逐帧 Key / Value 独立最近邻比例 R_NN

对同一帧、同一层、同一原生 KV head 的有符号 K/V，使用全部帧内视频 token
（含观察窗口），排除提示词、问题及回答，不展开 GQA。每个切片独立计算：

```text
s_K = sqrt(mean_i(||k_i||_2^2)) + epsilon
s_V = sqrt(mean_i(||v_i||_2^2)) + epsilon
d^NN,K_i = min_{j != i} ||k_i - k_j||_2 / s_K
d^NN,V_i = min_{j != i} ||v_i - v_j||_2 / s_V
tau_K = mean_i(d^NN,K_i)
tau_V = mean_i(d^NN,V_i)
R_NN,K = mean_i(d^NN,K_i < tau_K)
R_NN,V = mean_i(d^NN,V_i < tau_V)
```

Key 和 Value 各自计算最近邻距离，并以各自当前帧、当前层/head 内的距离均值作为阈值。
默认 `epsilon=1e-8`，加在平方根外；离线命令支持 `--epsilon`。阈值自动计算，不提供固定 τ 参数。
K、V 分别使用当前帧全部 token 的 RMS 范数作尺度，不逐 token 归一化，因此保留幅值差异。
K、V 独立寻找最近邻，分别保存最近邻索引、阈值和分数，不合并两种距离。
明确排除自身，距离相同时取最先达到最小值的帧内索引。阈值采用严格小于，等于 τ 不计入冗余。

最终分数是 `[0,1]` 内最近邻距离**严格低于本切片均值**的 token 比例，描述距离分布的相对关系，
不能单凭该比例判断绝对冗余程度。分数不使用 Shannon 熵，也不计算 `1-R_NN`。
零向量有效，由 epsilon 防止除零；所有距离相等时分数为 0。
特别地，所选状态的全部向量完全重复时距离为 0、τ=0，因为采用严格小于，分数也是 0。
只有一个 token 时没有其他 token 可比较，分数未定义，原因为 `no_other_tokens`。
输入非有限值或 K/V 形状不一致时明确报错，不删除 token。

评分和热图支持 **pre-RoPE / post-RoPE Key**，通过上方“Key 阶段”统一切换，默认 RoPE 后。
页面通过 Key / Value 选项同步切换分布图、空间热图和当前切片下载。
Key 按所选 RoPE 阶段独立计算；Value 使用原生缓存 V（V 不经过 RoPE），前后阶段的 V 结果相同。
RoPE 前 Key 在当前 Qwen2 每层的 `k_proj` 上以只读 forward hook 采集，保留 bias 并按原生 KV head 拆分；
不从旋转后的缓存逆推。钩子只在观察输入的完整 prefill 中启用，在追加问题和生成回答前移除，
异常或取消也会清理。这里的 pre-RoPE 指当前层的位置旋转之前。

RoPE 前 Key 保存到 `key_pre_rope/layer-XXX.npy`，布局为 `[frame, kv_head, token, channel]`；
`pre_rope_key_snapshot` 记录来源、维度和 dtype。FP16 原样保存，BF16 转为 FP32。
RoPE 后 K 和两阶段 V 读取 `kv/layer-XXX.npy`，布局为 `[key_value, frame, kv_head, token, channel]`。
缺少 pre-RoPE Key 的旧结果需重新生成才能使用前阶段。

分布图展示每帧全部层/head 的独立 R_NN，箱体为 Q1–Q3、横线为中位数、须为最小–最大、圆点为均值。
可独立筛选层与 head；先逐切片计算，再做统计，不先平均 K/V 或距离。
切换 Key 阶段时分布图与空间热图同步切换，并给出相同筛选下 RoPE 前后的 R_NN 均值。

### K/V 最近邻距离空间热图与导出

热图的每格是所选 Key 或 Value 的一个 token 的 `d^NN_i`，按原图 token 网格排列，数值越大表示越独特。
默认色阶为 `0 → 当前帧最大距离`；固定 `0 → 2` 便于跨帧比较，但距离没有固定的 2 上界，
超出时仅颜色饱和，悬停读数和导出保留真实值。全零或无有效值时自动色阶回退到 `0 → 2`。
悬停可读取最近邻距离、最近邻索引、是否低于 τ 和切片 R_NN；
青色实线标出当前 token，紫色虚线标出最近邻。点击或方向键选择可联动其他原图视图。
切换帧、层/head、RoPE 阶段时从快照按需读取，无需重新推理。

JSON `uniqueness` 与 `uniqueness_pre_rope` 分别保存后/前阶段结果。
指标标识为 `within_frame_separate_kv_nearest_neighbor_redundancy_mean_threshold`，`version=5`、`score_units=ratio`，
包含 `threshold_rule=mean_nearest_neighbor_distance`、`epsilon`、`evaluation=independent_key_and_value` 和公式。`thresholds`（每个切片的 τ）、`scores`、`uniqueness_sums`、`mean_uniqueness`、
`max_uniqueness`、`scales`、`zero_norm_tokens`、`undefined_reasons` 均按状态存储：
`field.key[layer][frame][kv_head]` 和 `field.value[layer][frame][kv_head]`。
为保留现有接口名称，`uniqueness` 字段现表示最近邻距离，`uniqueness_sums` 是距离之和。

CSV 分数列为 `nearest_neighbor_redundancy`，每个切片分别导出 `state=key`、`state=value` 两行，包含各自 τ、epsilon 和 Key 阶段。
切片与分布 PNG API 支持 `state=key|value`（默认 key），缓存和下载文件名区分状态。
Token NPZ/JSON 按所选状态导出，包含 `state`、`uniqueness`、`nearest_token`、`scale`、
`redundant`（距离是否严格小于 τ）、`score`、尺度、距离统计和参数；不再导出余弦或归一化概率。
最近邻索引从帧内 0 开始；未定义索引在 NPZ 中为 −1、JSON 中为 null，
未定义浮点值在 NPZ 中为 NaN、JSON 中为 null。NPZ 还保存全局 `token_positions`。

已有 KV 快照可以离线计算，保留原始 result.json 和 KV 文件：

```bash
python -m observation.vendi results/rvs_observation/<fingerprint>/result.json
python -m observation.vendi results/rvs_observation/<fingerprint>/result.json --position pre
# 可选：--epsilon 1e-8 --output-dir results/my-uniqueness
```

默认分别导出到 `uniqueness/post/`、`uniqueness/pre/`，包含 `scores.json`、`scores.csv`、`distribution-key.png` 和 `distribution-value.png`。
新观察保存到 `schema_version=14` 的主结果。旧余弦/熵、固定阈值或 K/V 联合结果不会复用为当前分数，
图表缓存区分指标、Key/Value、阈值规则、epsilon 和 RoPE 阶段。离线计算不修改页面主结果；
更新服务并重新生成观察数据后可在页面查看新分布。

## 问题前后逐帧 token 对照

每次计算都额外保存两组 query 对同一批视频 token 的注意力：

- **问题前**：视频序列末尾 N 个 token，与原视频窗口共用 N。
- **问题后**：问题正文的全部 token。对完整 ReKV 问题后缀一次分词，通过字符偏移选中与正文
  相交的 token，排除纯换行、角色标记等模板 token；不会为了凑 32 个 query 混入视频。
  若正文和相邻换行由同一 token 表示，该 token 整体参与统计，实际边界和文本均可查看。
- 每组 query 与 attention head 等权平均，softmax 包含所有因果可见 key。
  “全部层平均”先平均各层分数，再计算帧内熵。两组 query 的数量可能不同，页面显示实际数量与位置。

对照统计覆盖每帧的**全部视频 token**，包括视频窗口内部的位置；原历史重要性面板仍将窗口标灰并排除。
问题前窗口内部各 key 的因果可见次数不同，未对这种可见次数重新归一化。
追加问题不会改变旧视频 token 已计算的注意力；这里比较的是两组 query 对视频内容的关注。
对照在回答生成前采集，不包含生成回答的 query。

对照面板只展示当前选中帧，并排显示“问题前 / 问题后 / 差值”，差值 = 后 − 前。
可通过面板内的视频帧下拉框、原帧列表、注意力图表或 K/V 帧选择切换，所有视图同步。
切换帧仅更新展示，不重新计算模型；JSON 与对照 CSV 仍包含全部帧。
默认原始 token mass，所选层
全部帧的前后图共用同一线性色阶；差值共用以零为中心的对称色阶，蓝色减少、橙色增加。
可切换为帧内条件概率（各自除以该阶段帧 mass），再计算概率差；零 mass 的条件概率显示灰色“未定义”。
支持切换原图叠加、调整透明度、悬停读取数值、点击或方向键选择 token，并联动原帧细节与 K/V。
原帧细节仍显示原窗口统计，完整前后数值在对照面板中读取。

视频模式复用原视频 prefill 和缓存追加问题；原问题模式在完整 prefill 中选取两组 query。
统计按最多 32 个 query、4 个 head 分块，较大的 N 不会额外构造完整 N×序列长度矩阵。
大窗口仍会增加统计时间；取消可在统计分块边界生效。

任务参数新增 `video_window_size`（默认 32），参与缓存指纹。原“视频 + 问题末尾 32 token”
模式仍固定 32，N 在该模式下只控制新增对照的“问题前”部分。
JSON `schema_version=14`，`attention_comparison` 包含：

- `queries.before / after`：query 位置、数量及 token 明细；视频 token 以帧号、帧内序号标识。
- `token_positions`：两侧共享的完整视频 key 全局位置。
- `views`：`id=all` 或从 0 编号的层；`before / after` 的 `scores` 按 `token_positions` 排列，
  `frames` 保存完整帧 mass、条件熵等统计，`video_mass + text_mass ≈ 1`。

“对照 CSV”导出所选层的全部视频 token，包括 N、两组 query 数量、帧号、全局/帧内位置、
行列（均从 0 编号）、前后 mass 与条件概率及其差值。未定义的概率留空，零值保留为 0。
原 JSON / 帧统计 CSV 和 K/V 下载继续可用。旧 JSON 没有对照字段时，页面提示重新生成。

## 模型回答

每次计算都会对所选问题生成真实模型回答，使用与观察实验**完全相同的已选视频帧**。
默认视频模式先采集窗口注意力，再在该完整 prefill 的 KV cache 上追加 ReKV 问题模板；
问题模式的观察输入已包含问题，直接复用其 prefill。视频模式追加问题时额外采集正文注意力；
观察钩子在生成前移除，
回答的 query 不会改变之前的窗口统计，也不执行 ReKV 检索或 KV 淘汰。

回答使用贪心解码，默认最多 128 个新 token，可在界面设为 1–512。
遇到 EOS、回答长度上限或上下文上限停止，页面显示相应状态。
JSON `answer` 保存问题、回答文本、生成 token IDs、输入/输出 token 数、停止原因和耗时。
参考答案不输入模型。改变问题会清除旧回答；启用自动计算时会重新生成。

## 注意力与熵口径

这是独立的**完整因果注意力 prefill**观察路径，不调用 ReKV 的 KV 淘汰或检索，
不修改现有评测推理。这样所选输入范围内的历史 token 都能被窗口观察。
使用模型原生 RoPE 和 SDPA 计算隐藏状态，并在语言模型每层截取 post-RoPE Q/K。
原窗口统计使用最后 W 个 query（视频模式 W=N，问题模式 W=32），并额外采集前后对照。
query 与 heads 均分块处理；不保存完整序列长度 × 序列长度权重。
完整 prefill 的计算量仍随输入长度近似二次增长，限制帧数可降低计算成本。

令序列长度为 T，窗口大小为 w（视频模式取 N，问题模式取 32），窗口 W=[T−w,T)，历史 P=[0,T−w)：

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
与其求和相差窗口大小这一常数，排名和帧内条件熵相同；不执行 pooling、top-k 选择或压缩。

## 结果与复现

结果保存到 `results/rvs_observation/<fingerprint>/`：每帧 JPEG 与 `result.json`。
JSON 包含完整样本、输入参数、采样策略、token 布局、每层及全部层平均的重要性、
帧统计、空间映射、模型回答、K/V 快照元数据、逐帧 K/V 最近邻冗余比例、模型路径 / dtype / 层数 / head 数、软件版本及耗时。
缓存指纹包含参数、数据 / checkpoint 文件路径大小和修改时间、观察代码及关键依赖版本。
相同配置复用结果。服务只运行一个 GPU 计算任务，另允许少量排队任务；切换样本时取消旧任务。
取消在阶段或层边界生效，首次模型加载期间需等待加载结束。模型在服务进程内复用，
退出服务会释放显存。用 `Ctrl+C` 退出。

模型路径、数据根目录和缓存目录可通过 `--model-path` / `--data-root` / `--cache-dir` 设置。
当前适配 LLaVA-OneVision Qwen2（Transformers 5）；不自动下载模型，不支持其他 Video-LLM。

验证核心数学口径和与原生 attention 的一致性：

```bash
python -m unittest discover -s tests -p 'test_observation*.py' -v
```
