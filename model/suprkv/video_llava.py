"""Video-LLaVA frontend for SuPr-KV's shared streaming cache lifecycle."""
import torch
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor

from model.video_backbone import preprocess_video, video_features
from .onevision import OneVisionEngine
from .summarizer import VideoLlavaSummarizer


class VideoLlavaEngine(OneVisionEngine):
    family = 'video_llava'
    model_class = VideoLlavaForConditionalGeneration
    processor_class = VideoLlavaProcessor
    summarizer_class = VideoLlavaSummarizer

    def _frame_token_count(self):
        vision = self.model.config.vision_config
        # Video-LLaVA retains CLIP's CLS token for each video frame.
        return (vision.image_size // vision.patch_size) ** 2 + 1

    def _frame_features(self, image):
        pixels = preprocess_video(self.processor, self.family, torch.as_tensor(image)[None])
        feature = video_features(self.model, self.family, pixels.to(self.device, self.dtype))
        return feature, feature[0]
