import logging

import torch
from logzero import logger


class Abstract_ReKV:
    processor = None
    kv_cache = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

    def clear_cache(self, *, release_allocator=False):
        """Release video state; optionally return unused allocator memory too."""
        self.kv_cache = None
        if release_allocator:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

    @torch.inference_mode()
    def create_kv_snapshot(self, position_bias):
        from model.rekv.attention.snapshot import KVSnapshot, LayerSnapshot
        layers = tuple(LayerSnapshot.capture(layer) for layer in self.kv_cache)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        return KVSnapshot(layers, ready, position_bias)

    def _retrieve_question(self, input_ids, retrieved_indices=None, kv_snapshot=None):
        if kv_snapshot is not None:
            kv_snapshot.wait()
            if retrieved_indices is not None and any(
                    layer.block_size != self.n_frame_tokens for layer in kv_snapshot.layers):
                raise ValueError('External retrieval requires one block per frame')
            options = dict(rekv_position_bias=kv_snapshot.position_bias)
            output = self.language_model(input_ids=input_ids, use_cache=True,
                                         past_key_values=kv_snapshot.retrieval_contexts(retrieved_indices),
                                         **options)
            return output.past_key_values, options

        try:
            for layer in self.kv_cache:
                layer.set_retrieval()
                if retrieved_indices is not None:
                    assert layer.block_size == self.n_frame_tokens
                    layer.set_retrieved_block_indices(retrieved_indices)
            output = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            return output.past_key_values, {}
        finally:
            for layer in self.kv_cache:
                layer.reset_retrieval()

    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(videos=video_chunk, do_sample_frames=False, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        assert self.n_local >= video_features.shape[1], f'n_local: {self.n_local}, video_features: {video_features.shape[1]}'

        output = self.language_model(inputs_embeds=video_features, past_key_values=self.kv_cache, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        # encode chunk by chunk
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
