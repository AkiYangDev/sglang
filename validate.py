import ast, hashlib, itertools, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch

root=Path(__file__).resolve().parent
device=sys.argv[1] if len(sys.argv)>1 else 'cpu'
if device=='npu':
    import torch_npu
    torch.npu.set_device(0)
def load(name):
    tree=ast.parse((root/name).read_text(encoding='utf-8'))
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_build_npu_compress_metadata_prefill')
    ns={'torch':torch,'ForwardBatch':object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),name,'exec'),ns)
    return ns[method.name],ast.unparse(method)
old,old_code=load('before.py'); new,new_code=load('after.py')
expr='torch.arange(first_k, last_k, device=device, dtype=torch.int64) * ratio'
expected=old_code.replace(expr,'torch.arange(first_k * ratio, last_k * ratio, ratio, device=device, dtype=torch.int64)')
assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(new_code))
passed=0
for p,n,ratios,device_prefix in itertools.product([0,3,4,127,128,255,65535],[0,1,3,4,127,128,129],[[4],[128],[4,128]],[False,True]):
    lens=[n,0,1,129]; prefixes=[p,p+1,p+3,p+127]
    cu=torch.tensor([0]+list(itertools.accumulate(lens)),dtype=torch.int32,device=device)
    objs=[NS(forward_metadata=NS(actual_seq_lengths_q_pa=cu),_dsv4_unique_compress_ratios=ratios,
             _extend_prefix_lens_cpu=lambda _:prefixes) for _ in range(2)]
    bundle=NS(out_c4_loc=torch.arange(3,device=device),out_c128_loc=torch.empty(0,dtype=torch.int64,device=device))
    batch=NS(seq_lens=torch.tensor([a+b for a,b in zip(prefixes,lens)],device=device),positions=torch.arange(sum(lens),device=device),
        batch_size=4,extend_prefix_lens=torch.tensor(prefixes,device=device) if device_prefix else None,out_cache_loc_dsv4=bundle)
    old(objs[0],batch);new(objs[1],batch)
    for key,value in vars(objs[0].forward_metadata).items():
        other=getattr(objs[1].forward_metadata,key)
        assert torch.equal(value,other) if isinstance(value,torch.Tensor) else value==other,key
    for ratio in ratios:
        expected=[k*ratio for pre,length in zip(prefixes,lens) for k in range(pre//ratio,(pre+length)//ratio)]
        actual=getattr(objs[1].forward_metadata,f'positions_cmp_padding_c{ratio}').cpu().tolist()
        assert actual==expected+[0]*(len(actual)-len(expected))
    passed+=1
# Completely empty batches and all-zero requests.
for lens in ([],[0],[0,0,0]):
    bs=len(lens)
    obj=NS(forward_metadata=NS(actual_seq_lengths_q_pa=torch.zeros(bs+1,dtype=torch.int32,device=device)),
           _dsv4_unique_compress_ratios=[4,128],_extend_prefix_lens_cpu=lambda _: [0]*bs)
    batch=NS(seq_lens=torch.zeros(bs,device=device),positions=torch.empty(0,dtype=torch.int64,device=device),batch_size=bs,
             extend_prefix_lens=None,out_cache_loc_dsv4=None)
    new(obj,batch)
    assert obj.forward_metadata.positions_cmp_padding_c4.numel()==0
    assert obj.forward_metadata.positions_cmp_padding_c128.numel()==0
    passed+=1
if device=='npu':torch.npu.synchronize()
result=dict(device=device,passed=passed,source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['before.py','after.py']},
    ast_matches_benchmarked_expression=True,scope='Actual prefill helper, independent Python compressed-position oracle; C4/C128, prefixes, empty and boundary cases. No model/graph test.')
(root/f'RESULT_{device}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
