"""InfiniPot-V execution core, ported from qwen_inference_ovu.py.

Only compatibility changes: unwrap Transformers 5 video features and use the
4.57.3 RoPE calculation. Compression order and generation positions are intact.
"""
import torch
from transformers import DynamicCache
from .cache import process_kv_cache
from .rope import source_rope_index


class InfiniPotCore:
    MAX_GEN_TOKENS = 300
    DEFAULT_EOS_TOKEN_ID = 151645
    DEFAULT_MAX_PIXELS = 128 * 28 * 28

    def _print(self, message):
        pass

    def _visual_encoding_block(self, pixel_values_videos, video_grid_thw, frame_start, frame_end, total_frames):
        """
        Perform visual encoding for a specific block of frames
        
        This function enables block-wise vision encoding.
        Instead of encoding all frames at once, we encode only a subset of frames.
        
        Args:
            pixel_values_videos: Full pixel values tensor [total_patches, patch_dim]
            video_grid_thw: Original video grid info [1, 3] = [frame_num, H, W]
            frame_start: Starting frame index for this block
            frame_end: Ending frame index for this block (exclusive)
            total_frames: Total number of frames in the video
        
        Returns:
            video_embeds: Vision embeddings for this block of frames
        """
        # Calculate patches per frame
        # pixel_values_videos shape: [total_patches, patch_dim]
        # where total_patches = frame_num * H_patches * W_patches (before patch merger)
        total_patches = pixel_values_videos.shape[0]
        patches_per_frame = total_patches // total_frames
        
        # Extract pixel values for this block
        patch_start = frame_start * patches_per_frame
        patch_end = frame_end * patches_per_frame
        block_pixel_values = pixel_values_videos[patch_start:patch_end]
        
        # Create block-specific video_grid_thw
        block_frame_num = frame_end - frame_start
        block_video_grid_thw = video_grid_thw.clone()
        block_video_grid_thw[0, 0] = block_frame_num  # Update frame count
        
        # Convert to model dtype
        block_pixel_values = block_pixel_values.type(self.model.dtype)
        
        # Run vision tower on this block
        with torch.inference_mode():
            video_embeds = self.model.get_video_features(
                block_pixel_values,
                video_grid_thw=block_video_grid_thw
            )
            # get_video_features returns a list, concatenate
            video_embeds = getattr(video_embeds, "pooler_output", video_embeds)
            video_embeds = torch.cat(video_embeds, dim=0)
        
        return video_embeds

    def _block_wise_prefill(self, inputs, input_ids, past_key_values, system_size, inst_size, 
                             token_per_frame, vision_length, total_frames):
        """
        Perform block-wise prefill with integrated vision encoding
        
        Features:
        - Vision encoding happens within the block loop
        - Each iteration: extract frame pixels -> vision tower -> LLM forward -> KV compression
        
        Args:
            inputs: Original inputs containing pixel_values_videos and video_grid_thw
            input_ids: Input token IDs
            past_key_values: KV cache to update
            system_size: Size of system tokens
            inst_size: Size of instruction tokens
            token_per_frame: Number of tokens per frame (after patch merger)
            vision_length: Total vision token length
            total_frames: Total number of frames in the video
        
        Returns:
            tuple: (past_key_values, position_ids_full) - Updated KV cache and position IDs for generation
        """
        cur_frame = 0  # Track by frame index instead of token position
        
        # Get system token embeddings (these are text embeddings, processed once)
        system_input_ids = input_ids[:, :system_size]
        system_embeds = self.model.model.get_input_embeddings()(system_input_ids)
        
        # Get pixel values and grid info
        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]
        
        # Calculate position_ids for the full sequence (needed for 3D RoPE)
        position_ids_full, _ = source_rope_index(self.model.model, input_ids, video_grid_thw=video_grid_thw)
        
        # Store height_width for KV cache compression
        self.model.config.height_width = (
            position_ids_full[1, :, system_size:system_size + token_per_frame].max() - 
            position_ids_full[1, :, system_size:system_size + token_per_frame].min() + 1
        )
        
        # Calculate how many frames to process per block
        # Considering compression: actual new frames = block_size - compress_frame_num (except first block)
        
        while cur_frame < total_frames:
            # Calculate frame range for this block
            frame_start = cur_frame
        
            # Calculate how many frames can be processed in this block
            # For first block: full block_size frames
            # For subsequent blocks: block_size - compress_frame_num (since compress_frame_num frames are cached)
            if frame_start == 0:
                frames_in_block = min(self.block_size, total_frames)
            else:
                # Account for compressed frames in cache
                vis_cache_frames = self.compress_frame_num
                frames_in_block = min(self.block_size - vis_cache_frames, total_frames - frame_start)
        
            frame_end = frame_start + frames_in_block
        
            # Determine block type
            is_first_block = (frame_start == 0)
            is_last_block = (frame_end >= total_frames)
        
            # Step 1: Vision encoding for this block of frames
            block_video_embeds = self._visual_encoding_block(
                pixel_values_videos, video_grid_thw, 
                frame_start, frame_end, total_frames
            )
            block_video_embeds = block_video_embeds.to(system_embeds.device, system_embeds.dtype)
        
            # Step 2: Prepare inputs_embeds for this block
            if is_first_block:
                # First block: system tokens + vision tokens
                block_inputs_embeds = torch.cat([system_embeds, block_video_embeds.unsqueeze(0)], dim=1)
                # Calculate token range for position_ids
                token_start = 0
                token_end = system_size + frames_in_block * token_per_frame
            else:
                # Subsequent blocks: only vision tokens (system tokens are in cache)
                block_inputs_embeds = block_video_embeds.unsqueeze(0)
                # Token range for position_ids
                token_start = system_size + frame_start * token_per_frame
                token_end = system_size + frame_end * token_per_frame
        
            # Get position_ids for this block from the full position_ids
            position_ids = position_ids_full[:, :, token_start:token_end]
        
            # Step 3: LLM forward for this block
            with torch.inference_mode():
                outputs = self.model(
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=block_inputs_embeds,
                    use_cache=True
                )
        
            # Update cache
            past_key_values = outputs[1]
        
            # Step 4: KV cache compression for non-last blocks
            if not is_last_block:
                # Calculate remaining frames
                res_frame_num = total_frames - frame_end
            
                # Dynamic compression frame calculation
                if (self.compress_frame_num + res_frame_num) >= self.block_size:
                    compress_frame_num = self.compress_frame_num
                else:
                    compress_frame_num = self.block_size - res_frame_num
            
                # Apply KV cache compression
                past_key_values, _ = process_kv_cache(
                    past_key_values=past_key_values,
                    model=self.model,
                    system_size=system_size,
                    inst_size=inst_size,
                    token_per_frame=token_per_frame,
                    compress_frame_num=compress_frame_num,
                    method=self.compression_method,
                    tar_ratio=self.tar_ratio,
                    query_ratio=self.query_ratio,
                    adaptive_pooling=self.adaptive_pooling,
                    is_first_block=is_first_block,
                    is_last_block=is_last_block,
                    per_frame=self.per_frame
                )
        
            # Move to next block
            cur_frame = frame_end
        
        # Return the last position for generation stage
        return past_key_values, position_ids_full

    def _generation_stage(self, inputs, past_key_values, position_ids_full):
        """
        Generation stage: generate tokens autoregressively after prefill
        
        Args:
            inputs: Original inputs (to get instruction tokens)
            past_key_values: KV cache after prefill
            position_ids_full: Position IDs from prefill
        
        Returns:
            Generated response text
        """
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        
        # Extract instruction part after vision tokens
        inst_start = mask_indices[-1] + 1
        post_input_ids = input_ids[:, inst_start:]
        input_len = post_input_ids.shape[-1]
        
        # Initialize for generation
        input_ids_save = post_input_ids
        input_ids_current = post_input_ids
        
        # Calculate 3D position ids starting from prefill end
        position_ids = torch.arange(input_len, device=input_ids_current.device).expand(input_ids_current.shape[0], -1)
        position_ids = (position_ids + position_ids_full[0, 0, -1] + 1).unsqueeze(0).expand(3, -1, -1)
        
        # Get EOS token ID once
        eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id', self.DEFAULT_EOS_TOKEN_ID)
        
        # Generation loop
        for _ in range(self.MAX_GEN_TOKENS):
            with torch.no_grad():
                outputs = self.model(input_ids_current, past_key_values=past_key_values, position_ids=position_ids)
        
            past_key_values = outputs[1]
            next_token = outputs[0][:, -1, :].argmax(dim=-1).unsqueeze(1)
        
            input_ids_save = torch.cat([input_ids_save, next_token], dim=-1)
            input_ids_current = next_token
            position_ids = position_ids[:, :, -1:] + 1
        
            if next_token[0, -1] == eos_token_id:
                break
        
        # Decode generated tokens (excluding instruction)
        output_ids = input_ids_save[:, input_len:]
        return self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def block_process(self, inputs):
        """
        Block processing approach with integrated vision-LLM processing
        
        This is the main entry point for block processing.
        Vision encoding happens inside the block loop.
        
        Args:
            inputs: Preprocessed inputs
        
        Returns:
            Generated response text
        """
        # Extract video tokens and calculate dimensions
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        frame_number = inputs["video_grid_thw"][0, 0].item()
        
        # Calculate token positions and dimensions
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        system_size = input_ids[0, :mask_indices[0] - 1].shape[-1] + 1  # vision_start
        inst_size = input_ids[0, mask_indices[-1] + 2:].shape[-1] + 1  # including vision end token
        token_per_frame = int((input_ids == video_token_id).sum() // frame_number)
        
        # Calculate video-related dimensions
        total_frames = frame_number
        
        token_length = input_ids.shape[1]
        vision_length = token_length - system_size - inst_size
        
        # Initialize past_key_values for caching
        past_key_values = DynamicCache()
        
        self._print(f"Block Processing: {total_frames} frames, {token_per_frame} tokens/frame")
        self._print(f"Block size: {self.block_size}, Compress frames: {self.compress_frame_num}")
        
        # Step 1: Block-wise Prefill with integrated vision encoding
        past_key_values, position_ids_full = self._block_wise_prefill(
            inputs, input_ids, past_key_values,
            system_size, inst_size, token_per_frame, vision_length,
            total_frames
        )
        
        # Step 2: Generation Stage
        response = self._generation_stage(inputs, past_key_values, position_ids_full)
        
        return response
