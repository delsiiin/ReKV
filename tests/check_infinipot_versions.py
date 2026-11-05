"""Compare the actual 4.57.3 source with the installed-version adaptation.

Usage (reference dependencies must already exist in a separate directory):
  python tests/check_infinipot_versions.py --reference-path /path/to/transformers457

No model download, environment installation, or source monkeypatch is needed.
The source's undefined MAX_GEN_TOKENS global is supplied as its class constant;
the generation limit is shortened to three tokens for this CPU check.
"""
import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run_backend(backend, directory):
    import torch
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (DynamicCache, PreTrainedTokenizerFast, Qwen2TokenizerFast, Qwen2VLImageProcessor,
                              Qwen2VLVideoProcessor, Qwen2_5_VLProcessor, Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration)
    torch.set_num_threads(1)
    torch.manual_seed(73)
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, eos_token_id=63,
            max_position_embeddings=4096, rope_theta=1000000.,
            rope_scaling=dict(type='default', mrope_section=[1, 1, 2])),
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
            spatial_merge_size=2, temporal_patch_size=2, window_size=8, out_hidden_size=32,
            fullatt_block_indexes=[1], tokens_per_second=4),
        image_token_id=60, video_token_id=61, vision_start_token_id=58, vision_end_token_id=59)
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    # Use the same CPU attention implementation in both versions.
    model.set_attn_implementation('eager')
    model.generation_config.pad_token_id = 62
    model.generation_config.do_sample = False
    vocab = {f'token{i}': i for i in range(64)}
    special = {0: '[UNK]', 1: '<|im_start|>', 58: '<|vision_start|>', 59: '<|vision_end|>',
               60: '<|image_pad|>', 61: '<|video_pad|>', 62: '[PAD]', 63: '<|im_end|>'}
    for index, name in special.items():
        del vocab[f'token{index}']
        vocab[name] = index
    tok = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tokenizer_cls = Qwen2TokenizerFast if backend == 'source' else PreTrainedTokenizerFast
    tokenizer = tokenizer_cls(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]',
        eos_token='<|im_end|>', additional_special_tokens=list(special.values())[1:])
    options = dict(patch_size=2, temporal_patch_size=2, merge_size=2,
                   size=dict(shortest_edge=64, longest_edge=256))
    processor = Qwen2_5_VLProcessor(tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**options), video_processor=Qwen2VLVideoProcessor(**options))
    processor.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% for item in message['content'] %}{% if item['type']=='video' %}"
        "{{ '<|vision_start|><|video_pad|><|vision_end|>' }}"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    if backend == 'source':
        assert transformers.__version__ == '4.57.3', transformers.__version__
        torch.save(model.state_dict(), directory / 'weights.pt')
        # Import only the reference class, avoiding dataset/download imports.
        sys.path.insert(0, str(ROOT / 'InfiniPot-V'))
        from kvcache_utils import process_kv_cache
        tree = ast.parse((ROOT / 'InfiniPot-V/qwen_inference_ovu.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        scope = dict(torch=torch, DynamicCache=DynamicCache, process_kv_cache=process_kv_cache,
                     MAX_GEN_TOKENS=3)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'source', 'exec'), scope)
        runtime = scope['OfflineVideoEval'].__new__(scope['OfflineVideoEval'])
        runtime.__dict__.update(model=model, processor=processor, block_size=32, compress_frame_num=24,
            compression_method='infinipot-v', tar_ratio=.5, query_ratio=.25, adaptive_pooling=False,
            per_frame=False, verbose=False, max_frames_num=768, MAX_GEN_TOKENS=3)
    else:
        sys.path.insert(0, str(ROOT))
        from model.infinipot.runtime import InfiniPotRuntime
        model.load_state_dict(torch.load(directory / 'weights.pt', weights_only=True), strict=True)
        runtime = InfiniPotRuntime(model, processor, max_new_tokens=3)
    results = {}
    for grids in (2, 32, 33, 41, 49):
        torch.manual_seed(grids)
        # Include non-integer RGB values, as qwen-vl-utils returns float tensors.
        video = torch.rand(grids * 2, 3, 8, 12) * 255
        if backend == 'source':
            runtime._load_or_process_video = lambda messages, text, *unused: processor(
                text=[text], images=None, videos=[video], padding=True, return_tensors='pt')
            inputs = runtime.prepare_video_input('unused.mp4', 'token7 token8')
        else:
            runtime.set_video(video)
            inputs = runtime.prepare_inputs('token7 token8')
        result = {k: inputs[k].clone() for k in ('input_ids', 'attention_mask', 'video_grid_thw', 'pixel_values_videos')}
        positions = model.model.get_rope_index(inputs['input_ids'], video_grid_thw=inputs['video_grid_thw'])[0]
        result['positions'] = positions
        forwards = []
        def record(module, args, kwargs, output):
            forwards.append(dict(logits=output.logits.detach().clone(),
                keys=[layer.keys.detach().clone() for layer in output.past_key_values.layers],
                values=[layer.values.detach().clone() for layer in output.past_key_values.layers]))
        handle = model.register_forward_hook(record, with_kwargs=True)
        with torch.inference_mode():
            model.model.rope_deltas = None
            result['answer'] = runtime.block_process(inputs) if grids > 32 else runtime.generate(inputs)
        handle.remove()
        result['forwards'] = forwards
        results[grids] = result
    torch.save(results, directory / f'{backend}.pt')
    print(f'{backend}: Transformers {transformers.__version__}, completed {len(results)} video lengths')


def compare(directory):
    import torch
    source = torch.load(directory / 'source.pt', weights_only=True)
    adapted = torch.load(directory / 'adapted.pt', weights_only=True)
    largest = 0.
    def check(a, b, label):
        nonlocal largest
        if isinstance(a, torch.Tensor):
            diff = float((a - b).abs().max()) if a.numel() else 0.
            largest = max(largest, diff)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=label)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), label
            for key in a:
                check(a[key], b[key], f'{label}/{key}')
        elif isinstance(a, list):
            assert len(a) == len(b), label
            for i, (x, y) in enumerate(zip(a, b)):
                check(x, y, f'{label}/{i}')
        else:
            assert a == b, (label, a, b)
    check(source, adapted, 'result')
    print(f'PASS: processor tensors, positions, every forward KV/logits and answers; max_abs_diff={largest:g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-path')
    parser.add_argument('--backend', choices=['source', 'adapted'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.backend:
        run_backend(args.backend, args.directory)
    else:
        if not args.reference_path:
            parser.error('--reference-path is required')
        with tempfile.TemporaryDirectory(prefix='infinipot-parity-') as tmp:
            for backend in ('source', 'adapted'):
                env = os.environ.copy()
                env['PYTHONPATH'] = args.reference_path if backend == 'source' else str(ROOT)
                subprocess.run([sys.executable, __file__, '--backend', backend, '--directory', tmp], env=env, check=True)
            compare(Path(tmp))
