"""One shared model registry, plus real Video-LLaVA SuPr-KV smoke tests."""
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import (LlamaConfig, VideoLlavaConfig, VideoLlavaForConditionalGeneration,
                          VideoLlavaVideoProcessor)

from model.suprkv.video_llava import VideoLlavaEngine
from model.suprkv.runtime import StreamingConfig, StreamingRuntime
from model.video_backbone import model_family, question_prompt
from video_qa import base, run_eval
from video_qa.model_paths import MODEL_PATHS
from video_qa.methods import METHODS, validate_arguments
from video_qa.methods.suprkv import make_config


class Tokenizer:
    eos_token_id = 31
    bos_token_id = 1

    def __init__(self, bos=False):
        self.bos = bos
        self.texts = []

    def __call__(self, text, add_special_tokens=True, **kwargs):
        self.texts.append(text)
        ids = ([1] if self.bos and add_special_tokens else []) + [3, 4, 5]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, tokens, **kwargs):
        return 'a person walks'


class AllModelMethodsTests(unittest.TestCase):
    def test_every_method_accepts_every_model_in_launcher_and_workers(self):
        for name in MODEL_PATHS:
            for method_name, method in METHODS.items():
                with self.subTest(model=name, method=method_name):
                    args = run_eval.build_parser().parse_args([
                        '--method', method_name, '--model', name, '--dataset', 'rvs_ego'])
                    validate_arguments(args, method)
                    worker = base.build_parser(method_name).parse_args([
                        '--model', name, '--anno_path', 'unused', '--save_dir', 'unused'])
                    validate_arguments(worker, method)
                    if method_name == 'suprkv':
                        config = make_config(args)
                        self.assertEqual(config.model_path, MODEL_PATHS[name])
                        self.assertEqual(config.model_family, model_family(name))
                    elif method_name == 'rekv':
                        with patch('video_qa.methods.rekv.import_module') as imported:
                            method.load(args)
                        self.assertEqual(imported.return_value.load_model.call_args.kwargs['model_path'],
                                         MODEL_PATHS[name])


class VideoLlavaSuPrKVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def tiny_model(self):
        text = dict(vocab_size=32, hidden_size=32, intermediate_size=48,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=4096)
        model = VideoLlavaForConditionalGeneration(VideoLlavaConfig(
            text_config=LlamaConfig(**text), vision_feature_layer=-1,
            vision_config=dict(model_type='clip_vision_model', hidden_size=16, intermediate_size=24,
                num_hidden_layers=2, num_attention_heads=2, image_size=8, patch_size=2))).eval()
        processor = SimpleNamespace(tokenizer=Tokenizer(bos=True), video_processor=VideoLlavaVideoProcessor(
            size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8}))
        return model, processor

    @torch.inference_mode()
    def test_encode_compress_summary_answer_resume_and_reset(self):
        for strategy in ('summary', 'frame_mean_query'):
            for granularity in ('layer', 'head'):
                with self.subTest(strategy=strategy, granularity=granularity):
                    torch.manual_seed(42)
                    model, processor = self.tiny_model()
                    config = StreamingConfig(model_family='video_llava', device='cpu',
                        budget_tokens=1000, recent_frames=1, topk=2, group_size=8,
                        min_segment_frames=2, max_segment_frames=2, summary_max_new_tokens=2,
                        max_new_tokens=2, scoring_strategy=strategy, compression_granularity=granularity)
                    engine = VideoLlavaEngine(asdict(config), model=model, processor=processor)
                    self.assertEqual(engine.frame_tokens, 17)
                    with StreamingRuntime(config, lambda _: engine) as runtime:
                        for index in range(3):
                            runtime.append_frame(np.full((8, 8, 3), index * 50, dtype=np.uint8), index)
                        stats = runtime.close_segment()
                        self.assertEqual(stats['summary_failures'], 0)
                        self.assertLessEqual(stats['persistent_bytes'], stats['budget_bytes'])
                        if strategy == 'summary':
                            self.assertEqual(engine.last_summary['summary'], 'a person walks')
                        self.assertEqual(runtime.get_prompt('which?', mc=True), question_prompt('video_llava', 'which?', mc=True))
                        self.assertEqual(runtime.ask('what?')['answer'], 'a person walks')
                        runtime.append_frame(np.zeros((8, 8, 3), dtype=np.uint8), 3.)
                        runtime.close_segment()
                        self.assertEqual(engine.cache.next_id, 4)
                        self.assertFalse(engine.cache.snapshots)
                        runtime.reset()
                        self.assertEqual(runtime.memory_stats()['retained_frames'], 0)

    @torch.inference_mode()
    def test_video_llava_has_only_one_bos_in_qa_and_summary(self):
        model, processor = self.tiny_model()
        config = StreamingConfig(model_family='video_llava', device='cpu', group_size=8,
                                 summary_max_new_tokens=2, max_new_tokens=2)
        engine = VideoLlavaEngine(asdict(config), model=model, processor=processor)
        self.assertEqual(int(engine.prefix[0, 0]), 1)
        self.assertFalse((engine._question_ids('what?') == 1).any())
        prefix, suffix = engine.main_summarizer.prompt_ids([dict(index=0, timestamp=0.)])
        self.assertEqual(int(prefix[0, 0]), 1)
        self.assertFalse((suffix == 1).any())
        self.assertTrue(processor.tokenizer.texts[-1].endswith(' ASSISTANT:'))

    def test_runtime_loads_once_without_reloading_on_reset(self):
        model, processor = self.tiny_model()
        config = StreamingConfig(model_family='video_llava', model_path='/local/weights', device='cpu',
                                 scoring_strategy='frame_mean_query', group_size=8)
        with patch('transformers.VideoLlavaForConditionalGeneration.from_pretrained', return_value=model) as loader, \
                patch('transformers.VideoLlavaProcessor.from_pretrained', return_value=processor) as proc:
            with StreamingRuntime(config) as runtime:
                self.assertIsInstance(runtime.model, VideoLlavaEngine)
                runtime.reset()
                runtime.reset()
            loader.assert_called_once()
            proc.assert_called_once()
            self.assertEqual(loader.call_args.args, ('/local/weights',))
            self.assertEqual(loader.call_args.kwargs['dtype'], torch.float32)


if __name__ == '__main__':
    unittest.main()
