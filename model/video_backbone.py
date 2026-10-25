"""Prompts and visual token extraction shared by full-context and retrieval inference."""


INITIAL_PROMPTS = {
    'onevision': '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user ',
    'video_llava': 'USER: ',
    'longva': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
}


def question_prompt(family, query, mc=False):
    endings = {'onevision': '<|im_end|><|im_start|>assistant\n',
               'video_llava': ' ASSISTANT:',
               'longva': '<|im_end|>\n<|im_start|>assistant\n'}
    prompt = f'\n{query}{endings[family]}'
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
    if family == 'longva':
        features = model.get_model().get_vision_tower()(pixels)
        features = model.get_model().mm_projector(features)
        return model.get_2dPool(features).flatten(0, 1).unsqueeze(0)
    raise ValueError(f'Unknown model family: {family}')


def preprocess_video(processor, family, frames):
    if family == 'longva':
        return processor.preprocess(frames, return_tensors='pt').pixel_values
    return processor.video_processor(videos=frames, do_sample_frames=False,
                                     return_tensors='pt').pixel_values_videos
