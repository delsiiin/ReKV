"""Single-process OneVision model. Native isolated summaries, ReKV streaming attention."""
from contextlib import contextmanager
import time
import resource
import os

import torch
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from model.patch import patch_hf
from observation.engine import INIT_PROMPT, question_inputs, greedy_token_ids
from observation.segments import SUMMARY_SYSTEM, summary_prompt, selected_vision_features, capture_queries
from .attention import EncodeLayer, RetrieveLayer, streaming_attention_forward
from .cache import StreamingKVCache
from .segments import OnlineSegmenter


@contextmanager
def native_decoder(decoder):
    modules = [decoder] + [layer.self_attn for layer in decoder.layers]
    previous = [module.forward for module in modules]
    try:
        for module in modules:
            module.forward = module._old_forward
        yield
    finally:
        for module, forward in zip(modules, previous):
            module.forward = forward


class OneVisionEngine:
    def __init__(self, config):
        self.config = config
        device = config['device']
        if str(device).startswith('cuda'):
            torch.cuda.set_device(device)
        self.processor = LlavaOnevisionProcessor.from_pretrained(config['model_path'], local_files_only=True)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config['model_path'], local_files_only=True, dtype=torch.float16,
            device_map=device, attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        self.decoder = self.model.model.language_model
        self.device = self.decoder.get_input_embeddings().weight.device
        self.tokenizer = self.processor.tokenizer
        self.prefix = self.tokenizer(INIT_PROMPT, return_tensors='pt').input_ids.to(self.device)
        self.frame_tokens = 196
        self.sink_tokens = self.prefix.shape[1]
        local = config['recent_frames']*self.frame_tokens
        patch_hf(self.decoder, attention_factory=streaming_attention_forward,
                 n_local=local, n_init=self.sink_tokens, fattn=str(device).startswith('cuda'))
        self.features, self.frame_info = [], []
        self.cache = None
        self.timings = dict(summary_seconds=0., score_compress_seconds=0., encode_seconds=0.)
        self.reset()

    def _set_window(self, mode):
        window = (self.config['recent_frames']*self.frame_tokens if mode == 'encode' else
                  self.config['topk']*self.frame_tokens+self.config['max_new_tokens']+1024)
        for layer in self.decoder.layers:
            layer.self_attn._streaming_n_local = window

    @torch.inference_mode()
    def reset(self):
        self._set_window('encode')
        # Capture the actual initialization K/V through the same ReKV attention path.
        output = self.decoder(input_ids=self.prefix, use_cache=True, return_dict=True)
        sink = tuple((k[0], v[0]) for k, v in output.past_key_values)
        self.cache = StreamingKVCache(sink, self.config['budget_tokens'], self.frame_tokens,
                                      self.config['recent_frames'], self.config['group_size'])
        self.segmenter = OnlineSegmenter(self.config['ema_alpha'], self.config['min_segment_frames'],
                                         self.config['max_segment_frames'])
        self.features, self.frame_info = [], []
        self.timings = dict(summary_seconds=0., score_compress_seconds=0., encode_seconds=0.)

    def _summary_ids(self, frames):
        system = f'<|im_start|>system\n{SUMMARY_SYSTEM}<|im_end|><|im_start|>user\n'
        system_ids = self.tokenizer(system, return_tensors='pt').input_ids.to(self.device)
        suffix = question_inputs(self.tokenizer, summary_prompt(frames)).to(self.device)
        return system_ids, suffix

    def _summary_length(self, features, frames):
        system_ids, suffix = self._summary_ids(frames)
        return system_ids.shape[1] + suffix.shape[1] + sum(f.shape[1] for f in features)

    def _summary_inputs(self, features, frames):
        system_ids, suffix = self._summary_ids(frames)
        embedding = self.decoder.get_input_embeddings()
        visual = torch.cat([f.to(self.device) for f in features], 1)
        return torch.cat([embedding(system_ids), visual, embedding(suffix)], 1)

    @torch.inference_mode()
    def append_frame(self, image, timestamp):
        began = time.monotonic()
        closed_seconds = 0.
        pixels = self.processor.video_processor(videos=[image], do_sample_frames=False,
                    return_tensors='pt').pixel_values_videos.to(self.device, torch.float16)
        output = self.model.model.get_video_features(pixel_values=pixels,
                    vision_feature_layer=self.model.config.vision_feature_layer,
                    vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                    return_dict=True)
        raw = selected_vision_features(output, self.model.config.vision_feature_layer,
                                      self.model.config.vision_feature_select_strategy)
        feature = output.pooler_output.reshape(1, -1, self.decoder.config.hidden_size)
        if feature.shape[1] != self.frame_tokens:
            raise ValueError(f'Expected {self.frame_tokens} frame tokens; got {feature.shape[1]}')
        boundary = self.segmenter.push(raw[0])
        info = dict(index=self.cache.next_id, timestamp=float(timestamp))
        # Check the exact summary prompt size, including timestamps and generation reserve.
        context_full = self._summary_length(self.features+[feature], self.frame_info+[info]) + self.config['summary_max_new_tokens'] > self.decoder.config.max_position_embeddings
        if self.features and (boundary or context_full):
            close_started = time.monotonic()
            self.close_segment(reset_segmenter=False)
            closed_seconds += time.monotonic()-close_started
            self.segmenter.count = 1
        if self._summary_length([feature], [info]) + self.config['summary_max_new_tokens'] > self.decoder.config.max_position_embeddings:
            raise ValueError('Even one frame plus summary prompt exceeds the model context')
        contexts = []
        recent = [self.cache.frames[i] for i in self.cache.recent_ids]
        for layer, (sink_k, sink_v) in enumerate(self.cache.sink):
            parts = [frame.layers[layer].materialize() for frame in recent]
            key = torch.cat([sink_k]+[x[0] for x in parts], 1)[None]
            value = torch.cat([sink_v]+[x[1] for x in parts], 1)[None]
            contexts.append(EncodeLayer((key, value)))
        self._set_window('encode')
        self.decoder(inputs_embeds=feature, past_key_values=contexts, use_cache=True, return_dict=True)
        self.cache.append_frame(timestamp, [context.captured for context in contexts])
        self.features.append(feature.detach().clone())
        self.frame_info.append(info)
        self.timings['encode_seconds'] += time.monotonic()-began-closed_seconds
        if self.segmenter.count >= self.segmenter.max_frames:
            self.close_segment()

    @torch.inference_mode()
    def close_segment(self, reset_segmenter=True):
        if not self.features:
            return self.stats()
        began = time.monotonic()
        queries, error, text = None, '', ''
        try:
            with native_decoder(self.decoder):
                inputs = self._summary_inputs(self.features, self.frame_info)
                output = self.decoder(inputs_embeds=inputs, use_cache=True, return_dict=True)
                ids, _ = greedy_token_ids(self.decoder, self.model.lm_head, output,
                    self.config['summary_max_new_tokens'], {self.tokenizer.eos_token_id})
                text = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                del output, inputs
                if text:
                    content = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
                    combined = torch.cat([self.prefix, content], 1)
                    queries = [None]*len(self.decoder.layers)
                    def capture(layer, states):
                        attention = self.decoder.layers[layer].self_attn
                        queries[layer] = states[0, self.prefix.shape[1]:].reshape(-1,
                            attention.config.num_attention_heads, attention.head_dim).detach().clone()
                    with capture_queries(self.decoder, capture):
                        self.decoder(input_ids=combined, use_cache=False, return_dict=True)
                else:
                    error = 'summary_empty'
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            error = f'{type(exc).__name__}: {exc}'[:256]
        self.timings['summary_seconds'] += time.monotonic()-began
        began = time.monotonic()
        self.cache.close_segment(queries, error)
        self.timings['score_compress_seconds'] += time.monotonic()-began
        self.features, self.frame_info = [], []
        if reset_segmenter:
            self.segmenter.close()
        stats = self.stats()
        stats['summary'] = text
        stats['summary_error'] = error
        return stats

    @torch.inference_mode()
    def answer(self, snapshot, question, on_retrieved):
        """Complete retrieval and generation before allowing further video encoding."""
        began = time.monotonic()
        hits = set()
        contexts = [RetrieveLayer(snapshot, layer, self.config['topk'], hits)
                    for layer in range(len(self.decoder.layers))]
        ids = self.tokenizer(question, return_tensors='pt').input_ids.to(self.device)
        self._set_window('qa')
        result = self.decoder(input_ids=ids, past_key_values=contexts, use_cache=True, return_dict=True)
        past = result.past_key_values
        del result, contexts, snapshot
        on_retrieved(hits)
        output = self.decoder(input_ids=question_inputs(self.tokenizer, question).to(self.device),
                              past_key_values=past, use_cache=True, return_dict=True)
        del past
        tokens, reason = greedy_token_ids(self.decoder, self.model.lm_head, output,
                         self.config['max_new_tokens'], {self.tokenizer.eos_token_id})
        return dict(answer=self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    hit_frames=sorted(hits), finish_reason=reason, qa_seconds=time.monotonic()-began,
                    inference_pid=os.getpid(),
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0)

    def stats(self):
        result = self.cache.memory_stats()
        result.update(self.timings)
        result['inference_pid'] = os.getpid()
        result['model_instances'] = 1
        result['model_parameter_bytes'] = sum(p.numel()*p.element_size() for p in self.model.parameters())
        result['feature_working_bytes'] = sum(x.untyped_storage().nbytes() for x in self.features)
        result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result['gpu_peak_bytes'] = torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0
        return result
