import torch
from transformers import LlamaModel, MistralModel, Qwen2Model
from transformers.modeling_outputs import BaseModelOutputWithPast

from model.rekv.attention import RotaryEmbeddingESM, rekv_attention_forward


def huggingface_forward(forward):
    """Bridge the Transformers 5 attention API to ReKV's per-layer caches."""
    def hf_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Native decoder layers return only hidden states in Transformers 5.
        # Keep the updated ReKV cache in a shared list for this forward call.
        output, cache = forward(
            self, hidden_states, hidden_states,
            position_ids, use_cache, past_key_values[self.layer_idx],
            self.q_proj, self.k_proj, self.v_proj, self.o_proj,
            self.head_dim, self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )
        past_key_values[self.layer_idx] = cache
        return output, None

    return hf_forward


def patch_hf(model, attn_kwargs=None, base=None, distance_scale=None, attention_factory=None, **kwargs):
    attn_kwargs = {**(attn_kwargs or {}), **kwargs}
    decoder = model if isinstance(model, (LlamaModel, MistralModel, Qwen2Model)) else getattr(model, "model", None)
    if not isinstance(decoder, (LlamaModel, MistralModel, Qwen2Model)):
        raise ValueError(f"Only supports llama, mistral and qwen2 models, not {model.__class__.__name__}.")

    def model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rekv_position_bias=None,
        **kwargs,
    ):
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.return_dict if return_dict is None else return_dict
        if not use_cache:
            raise ValueError("ReKV requires use_cache=True.")
        if output_attentions:
            raise ValueError("ReKV does not return attention weights.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            caches = [None] * len(self.layers)
        elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) == len(self.layers):
            caches = list(past_key_values)
        else:
            raise ValueError("ReKV expects one ContextManager or (key, value) pair per decoder layer.")

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=self.position_bias if rekv_position_bias is None else rekv_position_bias,
                past_key_values=caches,
                use_cache=True,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(caches),
            hidden_states=all_hidden_states,
        )
        return output if return_dict else output.to_tuple()

    # RoPE is now owned by the decoder, and its settings live in rope_parameters.
    config = decoder.config
    rope_parameters = config.rope_parameters
    head_dim = decoder.layers[0].self_attn.head_dim
    dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
    rope = RotaryEmbeddingESM(
        dim,
        rope_parameters["rope_theta"] if base is None else base,
        1.0 if distance_scale is None else distance_scale,
        device=decoder.rotary_emb.inv_freq.device,
    )
    if base is None:
        # Preserve the checkpoint's frequencies (including static RoPE scaling).
        rope.inv_freq.copy_(decoder.rotary_emb.inv_freq)
    decoder.position_bias = rope

    factory = rekv_attention_forward if attention_factory is None else attention_factory
    forward = huggingface_forward(factory(**attn_kwargs))
    for layer in decoder.layers:
        attention = layer.self_attn
        if not hasattr(attention, "_old_forward"):
            attention._old_forward = attention.forward
        attention.forward = forward.__get__(attention, type(attention))
    if not hasattr(decoder, "_old_forward"):
        decoder._old_forward = decoder.forward
    decoder.forward = model_forward.__get__(decoder, type(decoder))
    return model
