"""Fixed-work CUDA comparison, independent of answer length and video decoding."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model.rekv.attention.rope import RotaryEmbeddingESM
from model.suprkv.attention import streaming_attention_forward
from model.suprkv.cache import StreamingKVCache


def timed(fn, repeats):
    torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(repeats):
        fn()
        torch.cuda.synchronize()
    return 1000*(time.perf_counter()-start)/repeats


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    torch.manual_seed(73)
    torch.set_num_threads(1)
    dtype=torch.float16; h,d,n=4,128,196
    sink=torch.randn(h,16,d,device='cuda',dtype=dtype)
    cache=StreamingKVCache([(sink,sink)],100000,n,dtype=dtype,compression_granularity='head')
    rope=RotaryEmbeddingESM(d).cuda()
    scores=torch.zeros(h,n,device='cuda')
    for head,count in enumerate((65,96,132,165)): scores[head,:count]=1
    for i in range(64):
        k,v=torch.randn(h,n,d,device='cuda',dtype=dtype),torch.randn(h,n,d,device='cuda',dtype=dtype)
        cache.append_frame(i,[(k,v)],rope_start=17+i%3*n)
        old=cache.frames[i];entry=old.layers[0]
        if i<62:
            entry=entry.classify_scores(scores).drop_low().shrink(128)
            if i%2: entry=entry.shrink(128)
        cache.frames[i]=replace(old,layers=(entry,),closed=True)
    snapshot=cache.acquire_snapshot(1)
    probe=torch.randn(1,28,1,d,device='cuda',dtype=dtype)
    def restore(batched):
        return snapshot.retrieve_layer(0,probe,64,position_bias=rope,headwise=True,batched=batched)[0]
    restore(False);restore(True)
    result={'gpu':torch.cuda.get_device_name(),'restore_ms':{},'attention_ms':{}}
    for batched in (False,True):
        name='batched' if batched else 'legacy'
        result['restore_ms'][name]=timed(lambda:restore(batched),10)
    for length,repeats in ((1,100),(32,10)):
        q=torch.randn(1,28,length,d,device='cuda',dtype=dtype)
        k,v=torch.randn(1,h,length,d,device='cuda',dtype=dtype),torch.randn(1,h,length,d,device='cuda',dtype=dtype)
        flat=lambda x:x.transpose(1,2).flatten(-2)
        fn=streaming_attention_forward(16000,16,fattn=True)
        for batched in (False,True):
            past=restore(batched)
            def step():
                nonlocal past
                out,past=fn(SimpleNamespace(),flat(q),flat(k),rope,True,past,
                    lambda _:flat(q),lambda _:flat(k),lambda _:flat(v),lambda x:x,d,28,h)
                return out
            for _ in range(3):step()
            name='batched' if batched else 'legacy'
            result['attention_ms'].setdefault(str(length),{})[name]=timed(step,repeats)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
