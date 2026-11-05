"""Prompts and visual token extraction shared by full-context and retrieval inference."""


def model_family(model_name):
    """Resolve checkpoint architecture independently of the inference method."""
    for prefix, family in (('llava_ov_', 'onevision'), ('video_llava_', 'video_llava'),
                           ('qwen2_5_vl_', 'qwen2_5_vl')):
        if model_name.startswith(prefix):
            return family
    raise ValueError(f'Unknown model architecture: {model_name}')


INITIAL_PROMPTS = {
    'onevision': '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user ',
    'video_llava': 'USER: ',
    'qwen2_5_vl': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>',
}


def question_prompt(family, query, mc=False):
    endings = {'onevision': '<|im_end|><|im_start|>assistant\n',
               'video_llava': ' ASSISTANT:',
               'qwen2_5_vl': '<|im_end|>\n<|im_start|>assistant\n'}
    prompt = f'\n{query}{endings[family]}'
    if family == 'qwen2_5_vl':
        prompt = '<|vision_end|>' + prompt
    if mc:
        prompt += ' Best option: (' if family == 'video_llava' else 'Best option: ('
    return prompt


def video_features(model, family, pixels):
    if family == 'onevision':
        return model.model.get_video_features(
            pixel_values=pixels, vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            return_dict=True).pooler_output
    if family == 'video_llava':
        batch, frames = pixels.shape[:2]
        features = model.model.get_video_features(
            pixel_values_videos=pixels, vision_feature_layer=model.config.vision_feature_layer,
            return_dict=True).pooler_output
        return features.reshape(batch, frames * features.shape[1], -1)
    raise ValueError(f'Unknown model family: {family}')


def preprocess_video(processor, family, frames):
    return processor.video_processor(videos=frames, do_sample_frames=False,
                                     return_tensors='pt').pixel_values_videos
