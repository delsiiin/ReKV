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
起始时间 / 视频末尾 token 数 N、生成或取消计算、切换观察层、查看逐帧图片 / mass 和 token 热力图、
在原图上定位 token、比较 attention mass 分布、生成所选问题的模型回答、
基于 vision encoder 输出距离自动分段、逐段主动总结，以及每帧图片 token 对全部 summary 内容 token 的平均余弦相似度热力图。
可下载完整 JSON、当前层帧统计 CSV 和图片 token 平均相似度 CSV。首次打开需点击「生成观察数据」；之后切换样本
默认自动计算，可关闭自动计算选项。参数变更后需点击生成按钮。

当前采集所选输入末尾窗口的注意力与帧 mass，并生成模型回答；默认启用新增的 segment 观察实验。
问题前后对照、K/V magnitude、K/V 冗余与有效数量、注意力熵的计算及接口已移除。
新结果使用 `schema_version=19`，`segment_experiment` 记录逐 token 相似度入口，
另存各语言模型层未池化的视频 / summary Key 与 hidden states，不保存 Value。旧缓存文件保留在磁盘上；
代码指纹变化会使新任务重新计算。更新后请重启服务并刷新页面。

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
  观察窗口 token 显示灰色，不混入历史评分。

## 模型回答

每次计算都会对所选问题生成真实模型回答，使用与观察实验**完全相同的已选视频帧**。
默认视频模式先采集窗口注意力，再在该完整 prefill 的 KV cache 上追加 ReKV 问题模板；
问题模式的观察输入已包含问题，直接复用其 prefill。观察钩子在观察 prefill 后移除；追加问题和生成回答均使用原生 SDPA，
回答的 query 不会改变之前的窗口统计，也不执行 ReKV 检索或 KV 淘汰。

回答使用贪心解码，默认最多 128 个新 token，可在界面设为 1–512。
遇到 EOS、回答长度上限或上下文上限停止，页面显示相应状态。
JSON `answer` 保存问题、回答文本、生成 token IDs、输入/输出 token 数、停止原因和耗时。
参考答案不输入模型。改变问题会清除旧回答；启用自动计算时会重新生成。

## 视觉特征距离、EMA 分段、主动 summary 与热力图

页面“03 / SEGMENT 观察”默认启用三步实验，也可关闭以仅运行原注意力和问答。
所有已选视频帧（包含末尾观察窗口）均参与分段。
分段使用 **vision encoder 输出的 patch 向量**：按 checkpoint 的 `vision_feature_layer`
从 `vision_tower` 的 `hidden_states` 选择视觉层，再应用 checkpoint 的 feature selection strategy。
向量取自多模态 projector 与空间池化之前，不使用语言模型 Key 或 hidden states。
当前 checkpoint 每帧有 729 个视觉 patch；相似度可视化仍对应语言模型输入的 196 个图片 token。
若 checkpoint 指定多个视觉层，则按其配置在特征维拼接。

对于相邻帧 t−1 / t 和相同空间 patch 位置 i，令 v 是上述视觉输出向量：

```
s[t] = sqrt(mean_i ||v[t,i]||_2^2 + epsilon)
d[t,i] = ||v[t,i] - v[t-1,i]||_2 / s[t]
D[t] = mean_i d[t,i]
similarity[t] = 1 / (1 + D[t])
tau[t] = max(epsilon, multiplier * EMA[t-1])
boundary[t] = D[t] > tau[t] and current_segment_frames >= min_frames
EMA[t] = (1-alpha) * EMA[t-1] + alpha * D[t]
```

`epsilon=1e-6`。尺度按当前帧全部 patch 的向量范数计算；距离使用 FP32。
对应空间位置逐一计算后取平均，不先对整帧向量池化；跨 vision encoder 批次的相邻帧也正常比较。
默认 `alpha=0.2`、`multiplier=1.5`、`min_frames=1`，均可在页面设置。
第一对帧初始化 EMA、不产生边界；只有 1–2 帧时形成一段。若第一对帧已经发生切换，此初始化无法检出它。
先使用历史 EMA 判定，再更新（边界距离也纳入更新）。切分发生在当前帧之前，新段从当前帧开始。
最后一段可以少于 `min_frames`，不会丢弃或回并图片。
分段与相似度面板的表示类型、语言模型层选择独立，切换相似度表示不会重新分段。
页面给出距离 / 阈值曲线、逐对数值、判定原因，以及按段分组的全部图片。

每段使用同一 LLaVA-OneVision 模型与已提取的视频特征，在独立的新上下文中生成真实 summary。
系统提示词要求主动跟踪对象、人物、动作、文字/数字、时序变化和异常；段提示词要求按时间顺序
概括可见属性、数量、空间布局、动作与状态变化、可读文字、结果和区分相似事件的线索。
提示词仅允许有图片证据的因果关系，不推断采样帧之间的精确发生时间。
输入附上该段各帧真实时间戳；只输出约 60–100 词的事实性段落。
完整提示词保存在 `observation/segments.py` 和结果 JSON，页面可展开查看。
每段从新 cache 开始，不输入问题、参考答案、其他段图片或其他 summary；不改写原注意力和问答上下文。
默认贪心生成至多 192 token，界面允许 1–512；记录 EOS / 长度上限 / 上下文上限，截断时在段卡片提示。

热力图仅展示**每个图片 token 在所选 summary 全部内容 token 上的平均余弦相似度**：

1. 选择一段文字 summary、一个图片 segment 和其中的一帧。默认选择 summary 的来源段，也可跨段对照。
2. “相似度表示”可选 **Key states**（默认）或 **Hidden states**，两种表示在同一次计算中采集：
   - Key：各语言模型层 `k_proj` 输出，RoPE 前，拼接原生 KV heads。
   - Hidden：各 decoder block 在 attention 与 MLP 残差更新后的输出，未经过语言模型最终的 RMSNorm。
     第 1 层对应第一个 decoder block 的输出，不是输入 embedding；最后一层也取最终 RMSNorm 之前。
3. 图片表示来自原完整视频 prefill，各图片 token 分开保存。文字使用生成后可显示的 summary 重新分词，
   以原 `INIT_PROMPT + summary` 独立做纯文本 prefill，按相同位置采集 Key 和 Hidden。
   只保留每个内容 token 的向量，排除提示词、EOS 和聊天标记；不输入图片或生成指令。
   JSON `summary.tokens` 保存此次重新编码的 token 序号、ID、解码文本，与生成时的 `token_ids` 分开记录。
4. 令 X 为所选表示：`C[l,f,i,j] = cosine(X_summary[l,i], X_video[l,f,j])`。
   Key 与 Hidden 分别在各自的向量空间内计算，不混合两种表示。
   图片 token j 的显示分数为 `score[l,f,j] = mean_i C[l,f,i,j]`。
   先计算每对 token 的余弦，再对全部 summary 内容 token 取算术平均；不先平均 Key 向量。
   “全部层”再对层取平均。若任何参与平均的值未定义，或 summary 没有内容 token，对应分数显示灰色。

页面将这些平均分数按图片空间网格叠加回原图（当前模型每帧 196 个位置），
色阶为当前 summary / 帧 / 层下有效分数的 `[min, max]`，图例显示实际端点。
最小值为浅色、最大值为深青色；切帧或切层时重新缩放，跨图比较应读取原始分数。
所有分数相同时统一着色；无有效分数时显示灰色。CSV 和悬停读数保留原始余弦均值。
仅显示图片 token 的均值热力图，不展示文字 × 图片矩阵或单独文字 token 的选择器。
旁边额外展示剪枝图片：以该帧有效图片 token 分数的算术平均作为阈值，
严格低于阈值的 token 空间区域覆盖为不透明白色，其余区域显示原图（等于均值保留）。
阈值使用原始余弦均值，不使用色阶归一化值；切换 summary / 帧 / 层后同步更新。
页面显示阈值、剪枝数和比例。无效分数不参与阈值且保留原图区域；全部无效时不展示剪枝图。
遮罩沿用 `spatial.cells[].box`，不覆盖右侧和底部无 token 锚点的窄边缘。
该图是空间遮罩预览，不改变模型输入或 KV cache。
支持悬停读数、点击固定位置、方向键移动、切帧和切层。
面板内“相似度层”可选择全部层平均或单层，与页面顶部的注意力“观察层”独立；
切换 summary 或帧时保留表示类型与相似度层选择；切换表示时保留当前帧和层。
均值热力图、min–max 色阶、帧内均值剪枝和 CSV 同步使用所选表示；CSV 文件名与列记录 `key` / `hidden`。
CSV 每行是一个图片 token，包含 summary / 帧 / 层、参与平均的文字 token 数、
图片帧内与全局 token 序号、网格行列、平均余弦分数、帧内阈值和剪枝标记（1=剪去，0=保留，无效项为空）；索引从 0 开始。
这是**所选模型表示空间的观察量，未经语义匹配校准**；高余弦本身不等同于语义正确或摘要质量。
原图片表示的因果上下文与独立编码文字的上下文不同，解读结果时应保留这一条件。

JSON `segment_experiment` 保存分段参数、视觉特征来源/层/patch 数/维度、相邻距离、EMA 更新、阈值、边界及其原因、
段的帧索引 / 时间戳、实际 summary / token IDs / 提示词 / 停止原因，以及逐 token 相似度接口元数据。
`token_keys/video-{layer}.npy` 保存 `[帧数, 每帧 token 数, KV heads × head_dim]`，
`token_keys/summary-{summary_index}-{layer}.npy` 保存 `[内容 token 数, KV heads × head_dim]`。
`token_hidden/` 使用相同文件名，最后一维为语言模型 `hidden_size`，帧和 token 索引与 Key 文件相同。
保存模型原始精度（bfloat16 无 NumPy 原生支持时无损转换为 FP32）；切片读取，不常驻全部层表示。
服务通过 `GET /api/token-similarity/{job_id}?summary=0&frame=0&layer=all&representation=hidden` 按需返回
`[文字 token 数, 图片 token 数]` 矩阵、文字 token 元数据和帧的全局 token 起点。
`representation` 可为 `key` 或 `hidden`，省略时为 `key`；响应记录实际表示。
前端将矩阵沿文字 token 维度取平均后展示与导出。旧 Key-only 结果不能切换 Hidden，需重新生成。
索引从 0 开始，layer 可为 `all` 或层序号。矩阵从已完成任务的表示文件计算，不重新运行模型。
JSON 不内嵌全部 token 对，以控制结果大小；复现需保留同目录下的 `token_keys/` 与 `token_hidden/`。
磁盘占用随层数、帧数、token 数与表示维度线性增加；Hidden 向量更宽，会增加缓存大小。
新代码指纹使旧分段结果自动失效；更新后重启服务并重新生成一次以获得视觉分段和两种表示。
段的 `start_frame` 含起点、`end_frame` 不含终点；时间范围仅指首尾采样帧，不表示精确事件边界。
所有新参数均进入缓存指纹。关闭实验时不生成该字段。

## 注意力计算口径

这是独立的**完整因果注意力 prefill**观察路径，不调用 ReKV 的 KV 淘汰或检索，
不修改现有评测推理。这样所选输入范围内的历史 token 都能被窗口观察。
使用模型原生 RoPE 和 SDPA 计算隐藏状态，并在语言模型每层截取 post-RoPE Q/K。
原窗口统计使用最后 W 个 query（视频模式 W=N，问题模式 W=32）。
query 与 heads 均分块处理；不保存完整序列长度 × 序列长度权重。
完整 prefill 的计算量仍随输入长度近似二次增长，限制帧数可降低计算成本。

令序列长度为 T，窗口大小为 w（视频模式取 N，问题模式取 32），窗口 W=[T−w,T)，历史 P=[0,T−w)：

```
A[l,h,q,i] = causal_softmax(Q[l,h,q] K[l,h,i]^T / sqrt(d))
s[l,i] = mean_{h, q in W} A[l,h,q,i], i in P
s[all,i] = mean_l s[l,i]
M[f] = sum_{i in frame f intersect P} s[i]
p[i|f] = s[i] / M[f]
```

Softmax 分母包含因果可见的全部 key（包括窗口内部已可见的 key），随后仅截取 P。
因此历史 token 的 mass **不会被重新归一化到 1**。
JSON 中 `video_mass + text_mass + window_mass ≈ 1`，可用于审计分数。
GQA 根据 query head 对应的 KV head 展开后统计；所有 attention heads 等权。

帧内条件概率由 token mass 除以帧总 mass 得到，0 mass 时未定义。
跨层视图先平均重要性，再汇总帧 mass。最后一帧可能仅部分进入历史区；
灰色窗口 token 不计分，JSON 记录 `scored_tokens` 和 `window_tokens`。

参考 [SnapKV 官方实现](https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py)
与 [论文](https://arxiv.org/abs/2404.14469) 的末尾窗口投票。这里对 query 取均值，
与其求和相差窗口大小这一常数，排名和帧内条件概率相同；不执行 pooling、top-k 选择或压缩。

## 结果与复现

结果保存到 `results/rvs_observation/<fingerprint>/`：每帧 JPEG 与 `result.json`。
JSON 包含完整样本、输入参数、采样策略、token 布局、每层及全部层平均的重要性、
帧统计、空间映射、模型回答、模型路径 / dtype / 层数 / head 数、软件版本及耗时。
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
