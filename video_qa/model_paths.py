"""Model presets shared by all video QA inference methods."""

MODEL_PATHS = {
    'llava_ov_0.5b': 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
    'llava_ov_7b': 'llava-hf/llava-onevision-qwen2-7b-ov-hf',
    'llava_ov_72b': 'llava-hf/llava-onevision-qwen2-72b-ov-hf',
    'video_llava_7b': 'LanguageBind/Video-LLaVA-7B-hf',
    'longva_7b': 'lmms-lab/LongVA-7B',
}

SUPRKV_MODELS = tuple(name for name in MODEL_PATHS if name.startswith('llava_ov_'))
