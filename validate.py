from __future__ import annotations
import ast,json,sys,types,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from typing import *
from abc import ABC,abstractmethod
import torch
import torch.nn.functional as F
import torch_npu
R=Path(__file__).parent
class MoeRunnerCore:
 def __init__(self,config):self.config=config
class FusedMoEMethodBase:pass
ns=dict(globals());ns.update(MoeRunnerConfig=NS,MoeRunnerBackend=NS(ASCEND='ascend'),AscendRunnerOutput=NS)
def load(file,names):
 tree=ast.parse((R/file).read_text()); nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),file,'exec'),ns)
for name in ['NPUW4A8Int8MoEMethod','NPUW4A8MXFP4MoEMethod','NPUMXFP8MoEMethod','_NPUBlockFp8MoEKernel']:ns[name]=type(name,(),{})
m=types.ModuleType('sglang.srt.hardware_backend.npu.quantization.block_fp8_moe');m._NPUBlockFp8MoEKernel=ns['_NPUBlockFp8MoEKernel'];sys.modules[m.__name__]=m
load('quant.py',['BaseHiddenStatesQuant','HiddenStatesDynamicQuant']);load('matmul.py',['BaseMatmul','GroupedMatmul'])
load('moe_methods.py',['_NPUMoEMethodBase','NPUUnquantMoEMethod','NPUW8A8Int8MoEMethod'])
load('activation.py',['BaseActivation','NPUSwiglu','NPUSwigluWithLimit','NPUSwigluQuant','NPUSwigluDeepEPKernel','NPUSwigluMxfp8Quant','NPUSitu','NPUSituMXFP8Quant','AllGatherActivationWrapper'])
results=[]
def oracle(runner,inp,q):
 g=runner.config.layer.w13_kernel.apply(q,inp.hidden_states,inp.expert_tokens,pertoken_scale=None,output_dtype=torch.bfloat16,weight_prefix='w13',group_list_type=inp.group_list_type)
 a,b=g.cpu().float().chunk(2,-1)
 lim=runner.config.swiglu_limit
 if lim is not None:a=a.clamp(max=lim);b=b.clamp(-lim,lim)
 v=(F.silu(a)*b).bfloat16().npu()
 return runner.config.layer.w2_kernel.apply(q,v,inp.expert_tokens,pertoken_scale=None,output_dtype=torch.bfloat16,weight_prefix='w2',group_list_type=inp.group_list_type)
for version in ['before','after']:
 load(version+'.py',['_uses_fused_gmm1','AscendRunnerCore'])
 for transport in (['none'] if version=='before' else ['none','deepep','ascend_ops']):
  ns['get_moe_a2a_backend']=lambda t=transport:NS(is_deepep=lambda:t=='deepep',is_ascend_ops=lambda:t=='ascend_ops')
  for kind in ['bf16','int8']:
   cls=ns['NPUUnquantMoEMethod' if kind=='bf16' else 'NPUW8A8Int8MoEMethod']
   for group_type in [0,1]:
    for limit in ([10.,None] if transport=='none' else [10.]):
     cfg=NS(layer=NS(w13_kernel=cls(),w2_kernel=cls()),activation='silu',swiglu_limit=limit,gemm1_alpha=None,gemm1_clamp_limit=None,use_tp_all_gather_activation=False)
     runner=ns['AscendRunnerCore'](cfg)
     dtype=torch.bfloat16 if kind=='bf16' else torch.int8
     q=NS(w13_weight=torch.eye(128,dtype=dtype,device='npu').repeat(2,1,1),w2_weight=torch.eye(64,dtype=dtype,device='npu').repeat(2,1,1),w13_weight_scale=torch.ones(2,128,dtype=torch.bfloat16,device='npu'),w2_weight_scale=torch.ones(2,64,dtype=torch.bfloat16,device='npu'))
     x=torch.empty(8,128,dtype=torch.bfloat16,device='npu');x[:,:64]=20;x[:,64:]=20;x[:5,1]=-20;x[:5,65]=-20;x[:5,2]=2;x[:5,66]=3;x[5:]=float('nan')
     inp=NS(hidden_states=x,hidden_states_scale=None,expert_tokens=torch.tensor([2,5] if group_type==0 else [2,3],dtype=torch.int64,device='npu'),group_list_type=group_type)
     actual=runner.run(inp,q,{}).hidden_states;expected=oracle(runner,inp,q)
     err=(actual[:5].float()-expected[:5].float()).abs().max().item();ok=torch.allclose(actual[:5],expected[:5],rtol=.01,atol=.02)
     results.append(dict(version=version,transport=transport,kind=kind,group_list_type=group_type,limit=limit,pass_=ok,max_error=err,actual=actual[0,:3].cpu().tolist(),expected=expected[0,:3].cpu().tolist()))
     print(json.dumps(results[-1]),flush=True)
     if version=='after' and transport=='none' and group_type==1 and limit==10.:
      for _ in range(2):runner.run(inp,q,{})
      torch.npu.synchronize(); graph=torch.npu.NPUGraph()
      with torch.npu.graph(graph):graph_out=runner.run(inp,q,{}).hidden_states
      x[:5,:64]=30;x[:5,64:]=-15;graph.replay();torch.npu.synchronize()
      expected=oracle(runner,inp,q)
      ok=torch.allclose(graph_out[:5],expected[:5],rtol=.01,atol=.02)
      results.append(dict(version=version,case='graph_replay',kind=kind,pass_=ok,max_error=(graph_out[:5].float()-expected[:5].float()).abs().max().item()))
r=dict(scope='Source-extracted runner/activation/quant/GMM methods; real NPU ops, mocked transport selector (no distributed communication), CPU activation oracle, no full model',results=results,sha256=hashlib.sha256((R/'after.py').read_bytes()).hexdigest())
(R/'RESULT_npu.json').write_text(json.dumps(r,indent=2))
assert all(v['pass_'] for v in results if v['version']=='after')
