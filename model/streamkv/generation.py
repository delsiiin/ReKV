"""Question-answering body copied from the supplied StreamKV OneVision adapter."""
import torch
from .abstract_streamkv import obtain_cdf_num, rescale_sum_match


class StreamKVGeneration:
    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)

        for layer_kv in self.kv_cache:
            layer_kv.set_retrieval()

        if retrieved_indices is None:
            for layer_kv in self.kv_cache:
                layer_kv.set_retrieval_prefill()

            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            retrieval_score = []
            for layer_kv in self.kv_cache:
                layer_kv.reset_retrieval_prefill()
                retrieval_score.append(layer_kv.layer_retrieval_score)

            target_num = len(self.kv_cache) * self.retrieve_size
            retrieval_nums = obtain_cdf_num(retrieval_score, target_num)
            retrieval_nums = rescale_sum_match(retrieval_nums,max(self.retrieve_size*(1-self.retrieve_temp),1),min(self.retrieve_size*(1+self.retrieve_temp),128))
            for i, layer_kv in enumerate(self.kv_cache):
                layer_kv.retrieval_num = retrieval_nums[i]

            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values
        else:
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values

        for layer_kv in self.kv_cache:
            layer_kv.reset_retrieval()

        output_ids = []
        stopped = False
        
        for i in range(max_new_tokens):
            if i == 0:  
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
            else:
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
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

