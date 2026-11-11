"""Single-video prefill snapshots for the Transformers 5.5 generation API."""
import copy

import torch
from transformers import DynamicCache


def clear_prefill_cache(runtime):
    runtime._prefill_cache = None
    runtime._prefill_prefix_ids = None
    runtime._prefill_position = None
    runtime._prefill_config = None
    runtime._prefill_mode = None
    runtime._prefill_positions = None
    runtime._prefill_attention_mask = None


def cached_generate_inputs(runtime, inputs, video_token_id):
    """Prefill through the last video token; leave decoding to native generate.

    Full prompt IDs are retained for logits processors (e.g. repetition penalty),
    beam search, stopping rules and output trimming. generate slices off the
    cached prefix using the full attention mask. No pixels are passed twice.
    """
    model = runtime.model
    generation = model.generation_config
    if ('pixel_values_videos' not in inputs or generation.use_cache is False
            or generation.cache_implementation is not None
            or generation.prefill_chunk_size is not None):
        # Respect explicit non-default cache strategies instead of overriding
        # them with our DynamicCache, and keep pure-text generation unchanged.
        runtime._clear_prefill_cache()
        return dict(inputs)
    ids = inputs['input_ids']
    markers = (ids[0] == video_token_id).nonzero().flatten()
    if ids.shape[0] != 1 or not markers.numel():
        raise ValueError('Video prefill reuse expects one video prompt')
    end = int(markers[-1]) + 1
    if end == ids.shape[1]:
        runtime._clear_prefill_cache()
        return dict(inputs)

    # Use the same positions as an uncached generate, including Qwen's fourth
    # (text) position axis and its per-question rope_deltas side effect.
    positions = model._prepare_position_ids_for_generation(ids, dict(inputs))
    prefix_ids = ids[:, :end]
    prefix_positions = positions[..., :end]
    mask = inputs['attention_mask'][:, :end]
    if (runtime._prefill_mode != 'native' or runtime._prefill_config != runtime.config
            or not torch.equal(runtime._prefill_prefix_ids, prefix_ids)
            or not torch.equal(runtime._prefill_positions, prefix_positions)
            or not torch.equal(runtime._prefill_attention_mask, mask)):
        runtime._clear_prefill_cache()
        prefix_inputs = dict(inputs)
        for name in ('input_ids', 'attention_mask', 'mm_token_type_ids', 'token_type_ids'):
            if name in prefix_inputs:
                prefix_inputs[name] = prefix_inputs[name][:, :end]
        prefix_inputs['position_ids'] = prefix_positions
        # Match native generate's default cache, including configured sliding layers.
        prefix_inputs['past_key_values'] = DynamicCache(config=model.config.get_text_config(decoder=True))
        output = model.model(**prefix_inputs, use_cache=True, return_dict=True)
        runtime._prefill_cache = output.past_key_values
        runtime._prefill_prefix_ids = prefix_ids.clone()
        runtime._prefill_positions = prefix_positions.clone()
        runtime._prefill_attention_mask = mask.clone()
        runtime._prefill_config = runtime.config
        runtime._prefill_mode = 'native'

    cache = copy.deepcopy(runtime._prefill_cache)
    # HF expands prompt tensors, but does not expand a caller-provided cache.
    expansion = max(generation.num_beams or 1, generation.num_return_sequences or 1)
    if expansion > 1:
        cache.batch_repeat_interleave(expansion)
    return dict(input_ids=ids, attention_mask=inputs['attention_mask'],
                position_ids=positions, past_key_values=cache)
