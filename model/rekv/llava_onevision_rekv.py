import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.video_backbone import INITIAL_PROMPTS, question_prompt, video_features
from model.rekv.patch import patch_hf
from model.rekv.abstract_rekv import Abstract_ReKV


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size)

    def get_prompt(self, query, mc=False):
        return question_prompt('onevision', query, mc)

    @property
    def language_model(self):
        return self.model.language_model

    def _get_video_features(self, pixel_values_videos):
        return video_features(self, 'onevision', pixel_values_videos)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None, *, kv_snapshot=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        past_key_values, forward_options = self._retrieve_question(
            input_ids, retrieved_indices, kv_snapshot)

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values, **forward_options)
                past_key_values = out.past_key_values
                logits = self.lm_head(out.last_hidden_state)
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                    **forward_options,
                )
                logits = self.lm_head(out.last_hidden_state)
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='llava-hf/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1):
    device = 'cuda'
    n_frame_tokens = 196
    processor = LlavaOnevisionProcessor.from_pretrained(model_path)
    
    init_prompt = INITIAL_PROMPTS['onevision']
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
    )
    patch_hf(model.language_model, **inf_llm_config)
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
