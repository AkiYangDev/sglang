import ast,json,statistics,time
from pathlib import Path
from types import SimpleNamespace as NS
import torch,torch_npu
torch.npu.set_device(0)
root=Path(__file__).resolve().parent
tree=ast.parse((root/'source.py').read_text(encoding='utf-8'))
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_npu_compress_metadata_prefill')
code=ast.unparse(method)
expr='torch.arange(first_k, last_k, device=device, dtype=torch.int64) * ratio'
assert expr in code
newcode=code.replace(expr,'torch.arange(first_k * ratio, last_k * ratio, ratio, device=device, dtype=torch.int64)')
ns={'torch':torch,'ForwardBatch':object}
exec(code,ns);old=ns[method.name]
exec(newcode,ns);new=ns[method.name]
def fixture(bs,length):
    lens=[length+(i%5) for i in range(bs)]
    prefix=[i*127 for i in range(bs)]
    cu=torch.tensor([0]+list(__import__('itertools').accumulate(lens)),dtype=torch.int32,device='npu')
    obj=NS(forward_metadata=NS(actual_seq_lengths_q_pa=cu),_dsv4_unique_compress_ratios=[4,128],_extend_prefix_lens_cpu=lambda _:prefix)
    batch=NS(seq_lens=torch.tensor([p+n for p,n in zip(prefix,lens)],device='npu'),positions=torch.arange(sum(lens),device='npu'),batch_size=bs,
             extend_prefix_lens=torch.tensor(prefix,device='npu'),out_cache_loc_dsv4=None)
    return obj,batch,lens,prefix
def run(fn,obj,batch): fn(obj,batch)
def timed(fn,obj,batch):
    for _ in range(5):run(fn,obj,batch)
    torch.npu.synchronize()
    samples=[]
    for _ in range(3):
        start=time.perf_counter()
        for _ in range(20):run(fn,obj,batch)
        torch.npu.synchronize()
        samples.append((time.perf_counter()-start)*1e6/20)
    return statistics.median(samples)
results=[]
for bs in (1,8,32):
    for length in (32,512,4096):
        obj,batch,lens,prefix=fixture(bs,length)
        old(obj,batch)
        before={k:v.cpu().clone() if isinstance(v,torch.Tensor) else v for k,v in vars(obj.forward_metadata).items()}
        new(obj,batch)
        for k,v in before.items():
            w=getattr(obj.forward_metadata,k)
            assert torch.equal(v,w.cpu()) if isinstance(v,torch.Tensor) else v==w
        for ratio in (4,128):
            expected=[k*ratio for p,n in zip(prefix,lens) for k in range(p//ratio,(p+n)//ratio)]
            actual=getattr(obj.forward_metadata,f'positions_cmp_padding_c{ratio}').cpu().tolist()
            assert actual==expected+[0]*(len(actual)-len(expected))
        runs=[('old',timed(old,obj,batch)),('new',timed(new,obj,batch)),('new',timed(new,obj,batch)),('old',timed(old,obj,batch))]
        results.append(dict(batch=bs,length=length,runs_us=runs,old_us=statistics.median(t for n,t in runs if n=='old'),new_us=statistics.median(t for n,t in runs if n=='new')))
out=dict(device='npu',passed=9,results=results,scope='Actual main prefill helper, only arange-times-ratio replaced; independent position oracle. Helper timings, not model TTFT. Original D2H remains in both arms.')
(root/'RESULT_npu.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out),flush=True)
