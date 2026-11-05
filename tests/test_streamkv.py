"""Differential tests against the supplied StreamKV source, without downloads.

The production cache is CUDA-only, like the reference. CPU execution below
changes only test-time device assertions, allocation and stream/event plumbing;
all cache, attention, selection and generation code runs unchanged on both sides.
"""
import argparse
import ast
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import asdict, replace
import importlib.util
import inspect
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import torch
from transformers import (LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration,
                          LlavaOnevisionVideoProcessor, Qwen2Config)

from model.streamkv import abstract_streamkv as adapted, cache, attention
from model.streamkv.config import StreamKVConfig
from model.streamkv.generation import StreamKVGeneration
from model.streamkv.runtime import StreamKVRuntime
from model.streamkv.prompts import INITIAL_PROMPT, GUIDANCE_PROMPT
from video_qa import base, run_eval
from video_qa.methods import get_method, validate_arguments
from video_qa.streamkv_stream_vqa import StreamKVStreamVQA
from video_qa.streamkv_offline_vqa import StreamKVOfflineVQA

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'StreamKV'


def setUpModule():
    adapted.logger.disabled = True


def tearDownModule():
    adapted.logger.disabled = False


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, SOURCE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = load_file('_reference_abstract', 'model/abstract_streamkv.py')
reference_cache = load_file('model.rekv.attention._reference_cache', 'model/attention/kv_cache_manager.py')


def source_method(relative, cls, name, namespace):
    tree = ast.parse((SOURCE / relative).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE / relative), 'exec'), scope)
    return scope[name]


reference_generation = source_method('model/llava_onevision_streamkv.py', 'LlavaOneVision_StreamKV',
    'question_answering', dict(torch=torch, obtain_cdf_num=reference.obtain_cdf_num,
                              rescale_sum_match=reference.rescale_sum_match))
reference_attention_scope = dict(ContextManager=reference_cache.ContextManager,
    get_multi_stage_dot_production_attention=attention.get_multi_stage_dot_production_attention)
reference_attention_tree = ast.parse((SOURCE/'model/attention/streamkv_attention.py').read_text())
reference_attention_tree.body = [n for n in reference_attention_tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
exec(compile(reference_attention_tree, '<reference attention>', 'exec'), reference_attention_scope)
reference_attention = reference_attention_scope['streamkv_attention_forward']


def normalize_imports(text):
    text = text.replace('from .kv_cache_manager import', 'from .cache import')
    text = text.replace('from .dot_production_attention import', 'from model.rekv.attention.dot_production_attention import')
    return ast.dump(ast.parse(text), include_attributes=False)


class SourceParityTests(unittest.TestCase):
    def test_core_source_bodies_are_unchanged(self):
        self.assertEqual((SOURCE/'model/abstract_streamkv.py').read_bytes(),
                         (ROOT/'model/streamkv/abstract_streamkv.py').read_bytes())
        for src, dst in [('kv_cache_manager.py', 'cache.py'), ('streamkv_attention.py', 'attention.py')]:
            self.assertEqual(normalize_imports((SOURCE/'model/attention'/src).read_text()),
                             normalize_imports((ROOT/'model/streamkv'/dst).read_text()))
        original = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(reference_generation))))
        port = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(StreamKVGeneration.question_answering))))
        self.assertEqual(original, port)

    def test_cdf_rescaling_matches_source_for_compression_retrieval_and_degenerate_scores(self):
        generator = torch.Generator().manual_seed(2026)
        scenarios = [([torch.randn(1,64,generator=generator) for _ in range(28)], 1075, 1, 64),
                     ([torch.randn(1,200+i*7,generator=generator) for i in range(28)],1792,32,96),
                     ([torch.ones(1,1)]*3,1,1,1), ([torch.ones(1,8)]*4,12,1,8)]
        for scores, target, lower, upper in scenarios:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                expected = reference.obtain_cdf_num(scores,target)
                actual = adapted.obtain_cdf_num(scores,target)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(adapted.rescale_sum_match(actual, lower, upper),
                             reference.rescale_sum_match(expected, lower, upper))
        # Preserve the known non-exact reference budget, not the prior heap rule.
        gen = torch.Generator().manual_seed(2026)
        scores = [torch.randn(1,64,generator=gen) for _ in range(28)]
        self.assertEqual(sum(adapted.rescale_sum_match(adapted.obtain_cdf_num(scores,1075),1,64)),1072)

    def test_python_cli_defaults_and_wrapper_profile_match_their_respective_sources(self):
        parser = argparse.ArgumentParser()
        tree = ast.parse((SOURCE/'video_qa/run_eval.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
                flags = [ast.literal_eval(a) for a in node.args]
                kwargs = {}
                for item in node.keywords:
                    kwargs[item.arg] = {'int':int,'float':float,'str':str}[item.value.id] if isinstance(item.value,ast.Name) else ast.literal_eval(item.value)
                parser.add_argument(*flags,**kwargs)
        defaults = vars(parser.parse_args([]))
        for name, value in asdict(StreamKVConfig()).items():
            if name == 'retrieve_chunk_size':
                self.assertEqual(value,1)
            else:
                self.assertEqual(str(value) if isinstance(value,bool) else value, defaults[name])
        shell = (ROOT/'run_streamkv.sh').read_text().replace('\\\n',' ')
        command = next(line for line in shell.splitlines() if line.startswith('python '))
        args = run_eval.build_parser().parse_args(shlex.split(command)[3:-1])
        source_values = {}
        for line in (SOURCE/'run_eval.sh').read_text().splitlines():
            line = line.split('#')[0].strip()
            if '=' in line and not line.startswith('python'):
                key,value = line.split('=',1)
                source_values[key] = value.strip()
        for name, value in source_values.items():
            if name == 'dataset':
                self.assertEqual(args.dataset, 'rvs_movie')
                self.assertIn(args.dataset, run_eval.DATASETS)
                continue
            actual = getattr(args,name)
            self.assertEqual(actual if isinstance(actual,(int,float)) and not isinstance(actual,bool) else str(actual),
                             float(value) if isinstance(actual,(int,float)) and not isinstance(actual,bool) else value)
        self.assertEqual(args.sample_fps,.5)

    def test_prompts_identical(self):
        tree = ast.parse((SOURCE/'model/llava_onevision_streamkv.py').read_text())
        values = {t.id:n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
                  for t in n.targets if isinstance(t,ast.Name)}
        self.assertEqual(INITIAL_PROMPT, values['init_prompt'])
        self.assertEqual(GUIDANCE_PROMPT, values['encode_prompt'])


def segmentation_trace(module, frames, cfg):
    processor = SimpleNamespace(video_processor=lambda x, **kwargs: SimpleNamespace(pixel_values_videos=x))
    runtime = module.Abstract_StreamKV(processor,frames.size(1),[1],None,[2],cfg.compression_ratio,
        cfg.compress_temp,cfg.retrieve_temp,cfg.max_segment_size,cfg.min_segment_size,cfg.segment_theta,
        cfg.segment_mode,str(cfg.use_segment_summary),cfg.n_local,cfg.retrieve_size,cfg.retrieve_chunk_size)
    runtime.device, runtime.dtype = 'cpu',frames.dtype
    batch_sizes, segments = [],[]
    def extract(x):
        batch_sizes.append(len(x))
        return x.reshape(1,-1,x.size(-1))
    runtime._get_video_features = extract
    runtime._encode_video_chunk = lambda x: segments.append(x.clone())
    runtime.calc_memory_usage = lambda: 0
    runtime.encode_video(frames)
    return batch_sizes,segments


class SegmentationTests(unittest.TestCase):
    def test_source_segments_and_vision_batch_boundaries_match(self):
        generator=torch.Generator().manual_seed(0)
        smooth=1+.02*torch.randn(160,2,16,generator=generator)
        changing=torch.cat([smooth[:70],-smooth[70:]])
        for mode in ('semantic','uniform'):
            for dtype in (torch.float32,torch.float16):
                for frames in (smooth,changing,smooth[:1]):
                    cfg=replace(StreamKVConfig(),segment_mode=mode,min_segment_size=4,segment_theta=.99)
                    expected=segmentation_trace(reference,frames.to(dtype),cfg)
                    actual=segmentation_trace(adapted,frames.to(dtype),cfg)
                    self.assertEqual(actual[0],expected[0])
                    self.assertEqual(len(actual[1]),len(expected[1]))
                    for a,b in zip(actual[1],expected[1]):
                        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_empty_semantic_segment_keeps_reference_failure(self):
        cfg=replace(StreamKVConfig(),segment_mode='semantic')
        errors=[]
        for module in (reference,adapted):
            with self.assertRaises((ValueError,RuntimeError)) as raised:
                segmentation_trace(module,torch.empty(0,2,8),cfg)
            errors.append((type(raised.exception),str(raised.exception)))
        self.assertEqual(errors[0],errors[1])


class _Event:
    def record(self,*args): pass
    def wait(self,*args): pass


@contextmanager
def cpu_cache_execution(*modules):
    """Test-only CUDA plumbing shim. No production CPU fallback is installed."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda,'stream',side_effect=lambda stream:nullcontext()))
        stack.enter_context(patch.object(torch.cuda,'Event',_Event))
        stack.enter_context(patch.object(torch.cuda,'current_stream',return_value=None))
        stack.enter_context(patch.object(torch.cuda,'empty_cache'))
        stack.enter_context(patch.object(torch.cuda,'ipc_collect'))
        for module in modules:
            source=textwrap.dedent(inspect.getsource(module.ContextManager.init))
            source=source.replace('        assert _t.is_cuda\n','')
            namespace=dict(module.__dict__)
            exec(compile(source,'<CPU test init>','exec'),namespace)
            stack.enter_context(patch.object(module.ContextManager,'init',namespace['init']))
            def alloc(self,num_units,unit_size,dtype):
                self.num_units,self.unit_size,self.dtype=num_units,unit_size,dtype
                self.data=torch.empty((num_units,unit_size),dtype=dtype)
                self.idle_set=set(range(num_units))
            stack.enter_context(patch.object(module.CudaCache,'__init__',alloc))
        yield


class Tokenizer:
    eos_token_id=None
    def __call__(self,text,**kwargs):
        ids=[3+ord(c)%27 for c in text[:3]]
        return SimpleNamespace(input_ids=torch.tensor([ids]) if kwargs.get('return_tensors') else ids)
    def decode(self,tokens,**kwargs):
        return ' '.join(map(str,tokens))


def tiny_runtime(cfg, source=False, device='cpu'):
    from model.rekv.patch import patch_hf
    torch.manual_seed(24)
    config=LlavaOnevisionConfig(text_config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=48,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2),
        vision_config=dict(model_type='siglip_vision_model',hidden_size=32,intermediate_size=48,
            num_hidden_layers=2,num_attention_heads=4,image_size=8,patch_size=2),
        vision_feature_layer=-1,vision_feature_select_strategy='full')
    model=LlavaOnevisionForConditionalGeneration(config).eval().to(device)
    processor=SimpleNamespace(tokenizer=Tokenizer(),video_processor=LlavaOnevisionVideoProcessor(
        size={'shortest_edge':8},crop_size={'height':8,'width':8}))
    runtime=StreamKVRuntime(model,processor,cfg,n_frame_tokens=4,fattn=False,
                           async_global_stream=False,pin_memory=False)
    if source:
        patch_hf(runtime.decoder,attention_factory=reference_attention,base=config.text_config.rope_parameters['rope_theta'],
                 **runtime.attention_settings)
        for name in ('clear_cache','encode_init_prompt','_encode_video_chunk','encode_video','calc_memory_usage'):
            setattr(runtime,name,getattr(reference.Abstract_StreamKV,name).__get__(runtime))
        runtime.question_answering=reference_generation.__get__(runtime)
    return runtime


def cache_snapshot(runtime):
    return [(c.local_k.clone(),c.local_v.clone(),c.block_k[0].get_data().clone(),
             [(b.cpu_data[0].clone(),b.cpu_data[1].clone()) for b in c.global_blocks[0]],
             c.reserved_num,c.retrieval_num,c.length) for c in runtime.kv_cache]


class RuntimeDifferentialTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def compare(self,left,right):
        for a,b in zip(left,right):
            for x,y in zip(a[:3],b[:3]): torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertEqual(len(a[3]),len(b[3]))
            for (ak,av),(bk,bv) in zip(a[3],b[3]):
                torch.testing.assert_close(ak,bk,rtol=0,atol=0)
                torch.testing.assert_close(av,bv,rtol=0,atol=0)
            self.assertEqual(a[4:],b[4:])

    def test_real_tf5_decoder_matches_source_layers_budgets_kv_and_answers(self):
        with cpu_cache_execution(cache,reference_cache):
            for ratio,summary,mode in ((.6,True,'uniform'),(.6,False,'semantic'),(1.,True,'uniform'),(1.,False,'uniform')):
                cfg=StreamKVConfig(n_local=24,retrieve_size=3,compression_ratio=ratio,compress_temp=1,
                    retrieve_temp=.5,min_segment_size=2,max_segment_size=4,segment_mode=mode,
                    segment_theta=.99,use_segment_summary=summary,retrieve_local_size=1)
                original=tiny_runtime(cfg,source=True)
                port=tiny_runtime(cfg)
                video=torch.arange(8*8*8*3,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(8,8,8,3)
                for runtime in (original,port):
                    runtime.encode_init_prompt()
                    runtime.encode_video(video)
                self.compare(cache_snapshot(port),cache_snapshot(original))
                for question in ('where','what'):
                    inputs=dict(question=question,prompt=port.get_prompt(question))
                    self.assertEqual(port.question_answering(inputs,max_new_tokens=3),
                                     original.question_answering(inputs,max_new_tokens=3))
                    self.compare(cache_snapshot(port),cache_snapshot(original))
                self.assertEqual(port.attention_settings['max_cached_block'],128)
                self.assertEqual(port.attention_settings['block_size'],4)
                # Continue through the same source encoding/cache branch after QA.
                original.encode_video(video[:1]); port.encode_video(video[:1])
                self.compare(cache_snapshot(port),cache_snapshot(original))

    def test_default_cache_parameters_and_unscaled_rope(self):
        cfg=StreamKVConfig()
        runtime=tiny_runtime(cfg)
        settings=runtime.attention_settings
        self.assertEqual(settings['max_cached_block'],128)
        self.assertEqual(settings['chunk_size'],1)
        self.assertEqual(settings['n_encode'],3)
        with patch('model.rekv.patch.patch_hf') as patcher:
            StreamKVRuntime(runtime.model,runtime.processor._processor,cfg)
        kw=patcher.call_args.kwargs
        self.assertTrue(kw['pin_memory'] and kw['async_global_stream'] and kw['fattn'])
        self.assertEqual(kw['block_size'],196)
        self.assertEqual(kw['exc_block_size'],196)
        self.assertEqual(kw['base'],runtime.decoder.config.rope_parameters['rope_theta'])

    @torch.inference_mode()
    def test_vision_api_bridge_matches_source_for_batched_and_single_frames(self):
        runtime=tiny_runtime(StreamKVConfig())
        extract=source_method('model/llava_onevision_streamkv.py','LlavaOneVision_StreamKV',
                              '_get_video_features',dict(torch=torch))
        original=SimpleNamespace(config=runtime.model.config,
            vision_tower=runtime.model.model.vision_tower,
            multi_modal_projector=runtime.model.model.multi_modal_projector,
            apply_pooling=runtime.model.model.apply_pooling)
        for frames in (1,4):
            pixels=torch.randn(1,frames,3,8,8)
            torch.testing.assert_close(runtime._get_video_features(pixels),extract(original,pixels),rtol=0,atol=0)

    def test_fp16_score_reduction_and_recent_block_selection_match_source(self):
        torch.manual_seed(7)
        prompt=torch.randn(1,4,80,16).half()
        vectors=torch.randn(12,64).half()
        for recent in (True,False):
            caches=[]
            for module in (cache,reference_cache):
                manager=module.ContextManager.__new__(module.ContextManager)
                manager.num_units,manager.unit_size,manager.dim_head=1,4,16
                manager.block_size=2
                manager.use_segment_summary='True'
                manager.block_k=[module.VectorTensor(64,torch.float16,'cpu')]
                manager.block_k[0].append(vectors)
                manager.calculate_score(6,prompt)
                torch.testing.assert_close(manager.layer_compression_score[0],
                    vectors[-3:-1].float() @ prompt.mean(2).reshape(-1).float(),rtol=0,atol=0)
                manager.retrieval_prefill=True
                manager.retrieve_size,manager.retrieve_local_size=4,8
                manager.retrieve_local=str(recent)
                manager.num_global_block=12
                manager.global_blocks=[list(range(12))]
                indices=manager._calc_block_topk(prompt)
                caches.append((manager,indices))
            torch.testing.assert_close(caches[0][0].layer_retrieval_score,
                                       caches[1][0].layer_retrieval_score,rtol=0,atol=0)
            self.assertEqual(caches[0][1],caches[1][1])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA hardware unavailable')
    def test_cuda_source_vs_port_attention(self):
        # Full production cache/device path; no CPU shim in this test.
        cfg=StreamKVConfig(n_local=24,retrieve_size=3,max_segment_size=4,min_segment_size=2,
                           compression_ratio=.6,compress_temp=1,retrieve_temp=.5,retrieve_local_size=1)
        original,port=tiny_runtime(cfg,True,'cuda'),tiny_runtime(cfg,False,'cuda')
        video=torch.zeros(4,8,8,3,dtype=torch.uint8)
        for runtime in (original,port):
            runtime.encode_init_prompt(); runtime.encode_video(video)
        self.compare(cache_snapshot(port),cache_snapshot(original))
        inputs=dict(question='where',prompt=port.get_prompt('where'))
        self.assertEqual(port.question_answering(inputs,max_new_tokens=3),original.question_answering(inputs,max_new_tokens=3))


class EvaluationTests(unittest.TestCase):
    def test_stream_rebuilds_full_prefix_even_at_repeated_or_decreasing_cutoffs(self):
        runtime=Mock()
        runtime.get_prompt.side_effect=lambda q,mc=False:q
        runtime.question_answering.return_value='B)'
        sample=dict(video_id='v',video_path='unused',conversations=[
            dict(question='q',answer='a',start_time=0,end_time=t) for t in (4,8,8,2)])
        flow=StreamKVStreamVQA(anno=[sample],save_dir='/tmp',sample_fps=1,qa_model=runtime)
        flow.load_video=lambda _: np.arange(8)[:,None,None,None]
        flow.analyze_a_video(sample)
        self.assertEqual([len(c.args[0]) for c in runtime.encode_video.call_args_list],[4,8,8,2])
        self.assertEqual(runtime.clear_cache.call_count,4)
        self.assertEqual(runtime.encode_init_prompt.call_count,4)
        self.assertTrue(all(c.kwargs['max_new_tokens']==256 for c in runtime.question_answering.call_args_list))

    def test_offline_encodes_video_once_with_no_artificial_question_boundaries(self):
        runtime=Mock(); runtime.question_answering.return_value='A)'
        sample=dict(video_id='v',video_path='unused',conversations=[dict(question='q',answer='a')]*2)
        flow=StreamKVOfflineVQA(anno=[sample],save_dir='/tmp',sample_fps=1,qa_model=runtime)
        flow.load_video=lambda _: np.zeros((8,8,8,3),dtype=np.uint8)
        flow.analyze_a_video(sample)
        runtime.clear_cache.assert_called_once(); runtime.encode_init_prompt.assert_called_once()
        runtime.encode_video.assert_called_once()
        self.assertEqual(runtime.question_answering.call_count,2)

    def test_only_shared_datasets_are_exposed(self):
        parser=run_eval.build_parser()
        self.assertEqual(set(parser._option_string_actions['--dataset'].choices),
                         {'mlvu','qaego4d','egoschema','activitynet_qa','cgbench','rvs_ego','rvs_movie'})
        for dataset in ('streamingbench_omni','streamingbench_real','streamingbench_sqa',
                        'streamingbench_proactive','videomme'):
            with self.subTest(dataset=dataset), patch('sys.stderr',new=io.StringIO()), self.assertRaises(SystemExit) as raised:
                parser.parse_args(['--method','streamkv','--dataset',dataset])
            self.assertEqual(raised.exception.code,2)
        self.assertNotIn('--data_root',parser._option_string_actions)
        self.assertNotIn('--data_root',base.build_parser('streamkv')._option_string_actions)

    def test_shared_datasets_route_and_preserve_explicit_false(self):
        method=get_method('streamkv')
        for dataset,(path,mode,scorer) in run_eval.DATASETS.items():
            args=run_eval.build_parser().parse_args(['--method','streamkv','--dataset',dataset,
                '--use_segment_summary','false','--retrieve_local','false'])
            validate_arguments(args,method)
            with patch.object(run_eval,'load_samples',return_value=[{}]), \
                    patch.object(run_eval.multiprocessing,'Process',return_value=Mock(exitcode=0)) as launch, \
                    patch.object(run_eval,'exec') as execute:
                run_eval.evaluate(args,dataset)
            command=launch.call_args.kwargs['args'][0]
            self.assertEqual(command[2],method.solver(mode))
            self.assertIn(path,command)
            worker=base.build_parser('streamkv').parse_args(command[3:])
            self.assertEqual(method.config(worker),method.config(args))
            self.assertIn('eval_'+scorer+'.py',execute.call_args.args[0])

    def test_source_npy_and_frame_directory_sampling(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Mock()
            sample=dict(video_id='v',video_path='unused',conversations=[dict(question='q',answer='a')])
            flow=StreamKVStreamVQA(anno=[sample],save_dir=tmp,sample_fps=.5,qa_model=runtime)
            path=Path(tmp)/'v.npy'
            np.save(path,np.arange(10)[:,None,None,None])
            np.testing.assert_array_equal(flow.load_video(str(path)).reshape(-1),[0,2,4,6,9])
            images=Path(tmp)/'frames'; images.mkdir()
            for i in range(17): Image.fromarray(np.full((2,2,3),i,np.uint8)).save(images/f'{i:03}.png')
            self.assertEqual(flow.load_video(str(images))[:,0,0,0].tolist(),[0,16])

    def test_launcher_remains_lightweight_and_result_namespace_excludes_previous_variant(self):
        subprocess.run([sys.executable,'-c',"from video_qa.run_eval import build_parser; import sys; build_parser(); assert 'torch' not in sys.modules"],check=True)
        args=run_eval.build_parser().parse_args(['--method','streamkv','--dataset','mlvu'])
        self.assertIn('/streamkv/source/',get_method('streamkv').result_dir(args,'mlvu'))


if __name__=='__main__': unittest.main()
