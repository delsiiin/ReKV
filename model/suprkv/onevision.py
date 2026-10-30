"""Single-process OneVision model. Native isolated summaries, ReKV streaming attention."""
from contextlib import contextmanager
from functools import partial
import time
import resource
import os

import torch
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from model.rekv.patch import patch_hf
from observation.engine import INIT_PROMPT, question_inputs, greedy_token_ids
from observation.segments import selected_vision_features, capture_queries
from .attention import EncodeLayer, RetrieveLayer, RotaryCache, streaming_attention_forward
from .cache import StreamingKVCache
from .segments import OnlineSegmenter
from .profiling import stage, measure
from .summarizer import OneVisionSummarizer, Summarizer, validate_summarizer_path


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
    def __init__(self, config, *, model=None, processor=None, profiler=None,
                 check_cancel=lambda: None, summarizer: Summarizer | None = None):
        self.config = config
        self.profiler = profiler
        self.check_cancel = check_cancel
        self.compression_enabled = config.get('compression_enabled', True)
        self.scoring_strategy = config.get('scoring_strategy', 'summary')
        if self.scoring_strategy not in ('summary', 'frame_mean_query'):
            raise ValueError('Invalid scoring_strategy')
        device = config['device']
        if str(device).startswith('cuda'):
            torch.cuda.set_device(device)
        if (model is None) != (processor is None):
            raise ValueError('Provide both model and processor, or neither')
        if model is None:
            self.processor = LlavaOnevisionProcessor.from_pretrained(config['model_path'])
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                config['model_path'], dtype=torch.float16,
                device_map=device, attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        else:
            self.model, self.processor = model, processor
        if self.model.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Streaming model requires float16, bfloat16 or float32')
        self.dtype = self.model.dtype
        self.decoder = self.model.model.language_model
        self._model_parameter_bytes = sum(p.numel()*p.element_size() for p in self.model.parameters())
        self.device = self.decoder.get_input_embeddings().weight.device
        self.tokenizer = self.processor.tokenizer
        self.prefix = self.tokenizer(INIT_PROMPT, return_tensors='pt').input_ids.to(self.device)
        self.frame_tokens = 196
        self.sink_tokens = self.prefix.shape[1]
        local = config['recent_frames']*self.frame_tokens
        patch_hf(self.decoder, attention_factory=streaming_attention_forward,
                 n_local=local, n_init=self.sink_tokens,
                 packed_attention=config.get('packed_attention', False),
                 fattn=str(device).startswith('cuda') and self.dtype != torch.float32)
        self.main_summarizer = OneVisionSummarizer(self.model, self.processor,
            config.get('model_path', 'main'), partial(native_decoder, self.decoder))
        path = validate_summarizer_path(config.get('summarizer_model_path')) if self.scoring_strategy == 'summary' else None
        self.summarizer = None
        if self.compression_enabled and self.scoring_strategy == 'summary':
            self.check_cancel()
            self.summarizer = summarizer if summarizer is not None else (
                OneVisionSummarizer.load(path, self.device, self.dtype) if path else None)
            self.check_cancel()
        self.features, self.frame_info = [], []
        self.cache = None
        self.timings = dict(summary_seconds=0., score_compress_seconds=0., encode_seconds=0.)
        self.reset()

    def _set_window(self, mode, max_new_tokens=None):
        limit = self.config.get('max_new_tokens', 256) if max_new_tokens is None else max_new_tokens
        window = (self.config['recent_frames']*self.frame_tokens if mode == 'encode' else
                  self.config['topk']*self.frame_tokens+limit+1024)
        for layer in self.decoder.layers:
            layer.self_attn._streaming_n_local = window

    @torch.inference_mode()
    @stage('cache_init')
    def reset(self):
        self._set_window('encode')
        # Capture the actual initialization K/V through the same ReKV attention path.
        output = self.decoder(input_ids=self.prefix, use_cache=True, return_dict=True)
        sink = tuple((k[0], v[0]) for k, v in output.past_key_values)
        self.cache = StreamingKVCache(sink, self.config['budget_tokens'], self.frame_tokens,
                                      self.config['recent_frames'], self.config['group_size'], profiler=self.profiler,
                                      dtype=self.dtype)
        self.segmenter = OnlineSegmenter(self.config['ema_alpha'], self.config['min_segment_frames'],
                                         self.config['max_segment_frames'])
        self.clear_segment()
        self.summary_counts = dict(summarizer_failures=0, summary_retries=0, summary_failures=0)
        self.last_summary = None
        self.timings = dict(summary_seconds=0., score_compress_seconds=0., encode_seconds=0.,
                            summarizer_vision_seconds=0.)

    def clear_segment(self):
        self.features, self.frame_info, self.summarizer_features = [], [], []
        self.summarizer_error = ''
        if self.cache is not None:
            self.cache.clear_frame_queries()

    def _summary_length(self, features, frames):
        return self.main_summarizer.context_length(features, frames)

    @staticmethod
    def _recoverable_error(exc):
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            raise exc
        return f'{type(exc).__name__}: {exc}'[:256]

    def _external_feature(self, image):
        began = time.monotonic()
        try:
            self.check_cancel()
            with measure(self.profiler, 'summarizer_vision_encoding'):
                return self.summarizer.encode_frame(image), ''
        except (ValueError, RuntimeError) as exc:
            return None, self._recoverable_error(exc)
        finally:
            self.timings['summarizer_vision_seconds'] += time.monotonic()-began

    @torch.inference_mode()
    @stage('video_encoding')
    def append_frame(self, image, timestamp):
        self.check_cancel()
        began = time.monotonic()
        closed_seconds = 0.
        with measure(self.profiler, 'vision_encoding'):
            pixels = self.processor.video_processor(videos=[image], do_sample_frames=False,
                        return_tensors='pt').pixel_values_videos.to(self.device, self.dtype)
            output = self.model.model.get_video_features(pixel_values=pixels,
                        vision_feature_layer=self.model.config.vision_feature_layer,
                        vision_feature_select_strategy=self.model.config.vision_feature_select_strategy,
                        return_dict=True)
            feature = output.pooler_output.reshape(1, -1, self.decoder.config.hidden_size)
            if feature.shape[1] != self.frame_tokens:
                raise ValueError(f'Expected {self.frame_tokens} frame tokens; got {feature.shape[1]}')
        info = dict(index=self.cache.next_id, timestamp=float(timestamp))
        summary_feature = None
        if self.compression_enabled:
            reserve = self.config['summary_max_new_tokens']
            with measure(self.profiler, 'segmentation'):
                raw = selected_vision_features(output, self.model.config.vision_feature_layer,
                                              self.model.config.vision_feature_select_strategy)
                boundary = self.segmenter.push(raw[0])
                context_full = (self.scoring_strategy == 'summary' and
                    self._summary_length(self.features+[feature], self.frame_info+[info]) + reserve > self.main_summarizer.context_limit)
            if self.frame_info and (boundary or context_full):
                close_started = time.monotonic()
                self.close_segment(reset_segmenter=False)
                closed_seconds += time.monotonic()-close_started
                self.segmenter.count = 1
            if (self.scoring_strategy == 'summary' and
                    self._summary_length([feature], [info]) + reserve > self.main_summarizer.context_limit):
                raise ValueError('Even one frame plus summary prompt exceeds the model context')
            if self.summarizer is not None and not self.summarizer_error:
                summary_feature, failure = self._external_feature(image)
                external_full = False
                if not failure:
                    try:
                        length = self.summarizer.context_length(self.summarizer_features+[summary_feature], self.frame_info+[info])
                        external_full = length + reserve > self.summarizer.context_limit
                        if self.summarizer.context_length([summary_feature], [info]) + reserve > self.summarizer.context_limit:
                            raise ValueError('One frame exceeds summarizer context capacity')
                    except (ValueError, RuntimeError) as exc:
                        failure = self._recoverable_error(exc)
                if self.features and external_full:
                    close_started = time.monotonic()
                    self.close_segment(reset_segmenter=False)
                    closed_seconds += time.monotonic()-close_started
                    self.segmenter.count = 1
                if failure:
                    self.summarizer_error = failure
                    self.summarizer_features = []
                    summary_feature = None
        contexts = []
        recent = [self.cache.frames[i] for i in self.cache.recent_ids]
        for layer, (sink_k, sink_v) in enumerate(self.cache.sink):
            parts = [frame.layers[layer].materialize() for frame in recent]
            key = torch.cat([sink_k]+[x[0] for x in parts], 1)[None]
            value = torch.cat([sink_v]+[x[1] for x in parts], 1)[None]
            positions = torch.cat([torch.arange(sink_k.shape[1], device=self.device)] +
                                  [part[2].long()+frame.rope_start for frame, part in zip(recent, parts)])
            contexts.append(EncodeLayer(RotaryCache(key, value, positions),
                                        capture_scores=self.compression_enabled,
                                        capture_frame_query=self.compression_enabled and self.scoring_strategy == 'frame_mean_query'))
        self._set_window('encode')
        self.decoder(inputs_embeds=feature, past_key_values=contexts, use_cache=True, return_dict=True)
        self.cache.append_frame(timestamp, [context.captured for context in contexts],
                                score_keys=[context.score_key for context in contexts] if self.compression_enabled else None,
                                frame_queries=[context.frame_query for context in contexts]
                                    if self.compression_enabled and self.scoring_strategy == 'frame_mean_query' else None,
                                rope_start=contexts[0].position_start)
        del contexts
        if not self.compression_enabled:
            self.cache.close_uncompressed_segment(return_stats=False)
            self.timings['encode_seconds'] += time.monotonic()-began
            return
        if self.scoring_strategy == 'summary':
            self.features.append(feature.detach().clone())
        if summary_feature is not None:
            self.summarizer_features.append(summary_feature)
        self.frame_info.append(info)
        self.timings['encode_seconds'] += time.monotonic()-began-closed_seconds
        if self.segmenter.count >= self.segmenter.max_frames:
            self.close_segment()

    @torch.inference_mode()
    @stage('segment_commit')
    def close_segment(self, reset_segmenter=True):
        self.check_cancel()
        if not self.compression_enabled:
            return self.stats(self.cache.close_uncompressed_segment())
        if not self.frame_info:
            return self.stats()
        if self.scoring_strategy == 'frame_mean_query':
            began = time.monotonic()
            record = dict(segment_id=self.cache.segment_id, scoring_strategy=self.scoring_strategy,
                          summary_enabled=False)
            try:
                cache_stats = self.cache.close_segment(None, scoring_strategy=self.scoring_strategy)
            finally:
                self.clear_segment()
            self.timings['score_compress_seconds'] += time.monotonic()-began
            self.last_summary = record
            if self.cache.last_event is not None:
                self.cache.last_event.update(record)
                cache_stats['last_event'] = dict(self.cache.last_event)
            if reset_segmenter:
                self.segmenter.close()
            return self.stats(cache_stats)
        began = time.monotonic()
        queries, error, text = None, '', ''
        first_error = self.summarizer_error
        retried = False
        source = self.summarizer.model_path if self.summarizer is not None else self.main_summarizer.model_path
        try:
            with measure(self.profiler, 'summary_generation'):
                if self.summarizer is not None:
                    if not first_error:
                        try:
                            text = self.summarizer.generate(self.summarizer_features, self.frame_info,
                                self.config['summary_max_new_tokens'], self.check_cancel).strip()
                            if not text:
                                first_error = 'summary_empty'
                        except (ValueError, RuntimeError) as exc:
                            first_error = self._recoverable_error(exc)
                    if first_error:
                        self.summary_counts['summarizer_failures'] += 1
                        self.summary_counts['summary_retries'] += 1
                        retried = True
                if self.summarizer is None or retried:
                    source = self.main_summarizer.model_path
                    text = self.main_summarizer.generate(self.features, self.frame_info,
                        self.config['summary_max_new_tokens'], self.check_cancel).strip()
                if not text:
                    error = 'summary_empty'
            if text:
                with native_decoder(self.decoder), measure(self.profiler, 'summary_query'):
                    content = self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
                    combined = torch.cat([self.prefix, content], 1)
                    queries = [None]*len(self.decoder.layers)
                    def capture(layer, states):
                        attention = self.decoder.layers[layer].self_attn
                        queries[layer] = states[0, self.prefix.shape[1]:].reshape(-1,
                            attention.config.num_attention_heads, attention.head_dim).detach().clone()
                    with capture_queries(self.decoder, capture):
                        self.decoder(input_ids=combined, use_cache=False, return_dict=True)
        except (ValueError, RuntimeError) as exc:
            error = self._recoverable_error(exc)
            queries = None
        self.timings['summary_seconds'] += time.monotonic()-began
        if error:
            self.summary_counts['summary_failures'] += 1
        record = dict(segment_id=self.cache.segment_id, scoring_strategy=self.scoring_strategy,
                      summary_enabled=True, summary=text, summary_source=source,
                      summarizer_error=first_error, summary_retried=retried,
                      summary_retry_succeeded=bool(text) if retried else None, summary_error=error)
        began = time.monotonic()
        try:
            cache_stats = self.cache.close_segment(queries, error)
        finally:
            self.clear_segment()
        self.timings['score_compress_seconds'] += time.monotonic()-began
        # Store only the latest record; observation copies each commit into its artifact.
        self.last_summary = record
        if self.cache.last_event is not None:
            self.cache.last_event.update(record)
            cache_stats['last_event'] = dict(self.cache.last_event)
        if reset_segmenter:
            self.segmenter.close()
        stats = self.stats(cache_stats)
        stats.update(record)
        return stats

    @torch.inference_mode()
    @stage('answer_generation')
    def answer(self, snapshot, question, on_retrieved, *, prompt=None, max_new_tokens=None):
        """Complete retrieval and generation before allowing further video encoding."""
        began = time.monotonic()
        limit = self.config['max_new_tokens'] if max_new_tokens is None else max_new_tokens
        hits = set()
        contexts = [RetrieveLayer(snapshot, layer, self.config['topk'], hits, getattr(self, 'profiler', None))
                    for layer in range(len(self.decoder.layers))]
        ids = self.tokenizer(question, return_tensors='pt').input_ids.to(self.device)
        self._set_window('qa', max_new_tokens=limit)
        with measure(getattr(self, 'profiler', None), 'retrieval'):
            result = self.decoder(input_ids=ids, past_key_values=contexts, use_cache=True, return_dict=True)
        past = result.past_key_values
        del result, contexts, snapshot
        on_retrieved(hits)
        prompt_ids = (question_inputs(self.tokenizer, question) if prompt is None else
                      self.tokenizer(prompt, return_tensors='pt').input_ids)
        output = self.decoder(input_ids=prompt_ids.to(self.device),
                              past_key_values=past, use_cache=True, return_dict=True)
        del past
        tokens, reason = greedy_token_ids(self.decoder, self.model.lm_head, output,
                         limit, {self.tokenizer.eos_token_id}, getattr(self, 'check_cancel', lambda: None))
        answer = self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return dict(answer=answer,
                    hit_frames=sorted(hits), finish_reason=reason, qa_seconds=time.monotonic()-began,
                    inference_pid=os.getpid(),
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0)

    def stats(self, cache_stats=None):
        result = self.cache.memory_stats() if cache_stats is None else dict(cache_stats)
        result.update(self.timings)
        result['scoring_strategy'] = self.scoring_strategy
        result['inference_pid'] = os.getpid()
        result['model_instances'] = 1 + (self.summarizer is not None)
        extra_bytes = self.summarizer.parameter_bytes if self.summarizer is not None else 0
        result['model_parameter_bytes'] = self._model_parameter_bytes + extra_bytes
        result['summarizer_parameter_bytes'] = extra_bytes
        result['summarizer_model_path'] = self.summarizer.model_path if self.summarizer is not None else None
        result.update(self.summary_counts)
        result['last_summary'] = self.last_summary
        result['feature_working_bytes'] = sum(x.untyped_storage().nbytes()
                                            for x in self.features + self.summarizer_features)
        result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result['gpu_peak_bytes'] = torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0
        return result
