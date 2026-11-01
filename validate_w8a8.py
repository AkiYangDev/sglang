from __future__ import annotations
"""Actual edited methods; synthetic weights/pools, no full model or compressor."""
import gc, json, time, statistics, sys, types
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
from copy import copy
import torch, torch_npu, sgl_kernel_npu, custom_ops
from production_rope import Dsv4NpuRoPE

torch.npu.set_device(0)
torch.manual_seed(73)
root=Path(__file__).parent
records=[]
def record(row):
 records.append(row); print(json.dumps(row),flush=True)
 (root/'tile_results.json').write_text(json.dumps(records,indent=2))
gate=types.ModuleType('sglang.srt.layers.attention.dsv4.unified_kv_gate')
gate.is_unified_kv_fp8=gate.is_unified_kv_triton=lambda:False
sys.modules[gate.__name__]=gate
moe_utils=types.ModuleType('sglang.srt.layers.moe.utils')
moe_utils.should_skip_mlp_all_reduce=lambda:False
sys.modules[moe_utils.__name__]=moe_utils
class Envs:
 def __getattr__(self,name): return NS(get=lambda:False)
class Mode:
 def is_decode_or_idle(self): return False
 def is_target_verify(self): return False
 def is_idle(self): return False
envs=Envs()
ns=dict(torch=torch,torch_npu=torch_npu,Dsv4NpuRoPE=Dsv4NpuRoPE,
        envs=envs,is_cp_active=lambda fb:False,_is_npu=True,_is_hip=False,
        _is_cuda=False,_is_gfx95_supported=False,_is_gfx1250_supported=False,
        contextmanager=contextmanager,copy=copy)
exec((root/'model_methods.py').read_text(),ns)
exec((root/'backend_methods.py').read_text(),ns)
ns['_sparse_attn_ops']=lambda *args,**kw:(torch.ops.custom.npu_sparse_attn_sharedkv_metadata,torch.ops.custom.npu_sparse_attn_sharedkv)
ns['_sparse_attn_kv_quant_kwargs']=lambda:{}
class Backend: pass
for n in ['prefill_query_chunk','_kernel_metadata_from_parts','forward','_forward_swa','_forward_compressed']:
 setattr(Backend,n,ns[n])
def dynamic_apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
    if isinstance(x, tuple):
        'dynamic_scale is calculated in malprolog kernel'
        original_dtype = torch.bfloat16
        quant_out, dynamic_scale = x
    else:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x)
    return torch.ops.npu.npu_quant_matmul(quant_out, layer.weight, layer.weight_scale, pertoken_scale=dynamic_scale.flatten(), bias=bias, output_dtype=original_dtype)

class Model: pass
for n in ['_forward_prepare','_forward_npu_chunked_prefill','_project_wo_b']: setattr(Model,n,ns[n])

with torch.inference_mode():
 wq=torch.randn(32768,1024,device='npu',dtype=torch.bfloat16)/32
 wo=torch.randn(8,1024,4096,device='npu',dtype=torch.bfloat16)/64
 wb=torch.randn(4096,8192,device='npu',dtype=torch.bfloat16)/(8192**.5)
 wk=torch.randn(512,1024,device='npu',dtype=torch.bfloat16)/32
 oneq=torch.ones(1024,device='npu',dtype=torch.bfloat16)
 onek=torch.ones(512,device='npu',dtype=torch.bfloat16)
 qlayer=NS(weight=torch.randint(-127,128,(1024,32768),device='npu',dtype=torch.int8),weight_scale=torch.full((32768,),.0005,device='npu'))
 blayer=NS(weight=torch.randint(-127,128,(8192,4096),device='npu',dtype=torch.int8),weight_scale=torch.full((4096,),.00015,device='npu'))
 del wq,wb
 for T,prefix,ratio,chunk in [(257,123,0,64),(257,255,4,64),(257,255,128,64),(8192,256,4,2048)]:
  if T==16384:
   gc.collect();torch.npu.empty_cache()
   free,totalmem=torch.npu.mem_get_info()
   if free < 4500*1024**2:
    record(dict(kind='skip',tokens=T,reason='free memory below 4500 MiB',free_bytes=free));continue
  x=torch.randn(T,1024,device='npu',dtype=torch.bfloat16)*.1
  positions=torch.arange(prefix,prefix+T,device='npu')
  angle=positions.float()[:,None]/torch.pow(10000.,torch.arange(0,64,2,device='npu')/64.)[None,:]
  cos=angle.cos().repeat_interleave(2,-1).to(x.dtype).view(T,1,1,64)
  sin=angle.sin().repeat_interleave(2,-1).to(x.dtype).view(T,1,1,64)
  nsin=-sin; del angle
  total=T+prefix
  kv=torch.randn((total+127)//128,128,1,512,device='npu',dtype=x.dtype)*.1
  cmp_page=32 if ratio==4 else 16
  compressed=torch.randn(max(1,(total//max(ratio,1)+cmp_page-1)//cmp_page),cmp_page,1,512,device='npu',dtype=x.dtype)*.1
  fm=NS(actual_seq_lengths_q_pa=torch.tensor([0,T],device='npu',dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([total],device='npu',dtype=torch.int32),
        swa_page_table=torch.arange(kv.shape[0],device='npu',dtype=torch.int32)[None],
        c4_page_table=torch.arange(compressed.shape[0],device='npu',dtype=torch.int32)[None],
        c128_page_table=torch.arange(compressed.shape[0],device='npu',dtype=torch.int32)[None])
  ids=torch.arange(512,device='npu',dtype=torch.int32)[None].repeat(T,1)
  ids.masked_fill_(ids >= ((positions+1)//4)[:,None],-1)
  fm.c4_topk_indices=ids
  b=Backend()
  b.__dict__.update(forward_metadata=fm,_is_dspark_draft_worker=False,
      _dsv4_sliding_window_size=128,_dsv4_q_head_num=64,_dsv4_kv_head_num=1,
      _dsv4_head_dim=512,_dsv4_has_c4=ratio==4,_dsv4_has_c128=ratio==128,
      _dsv4_index_topk=512,_dsv4_index_n_heads=64,_dsv4_index_head_dim=128)
  b.token_to_kv_pool=NS(get_swa_buffer=lambda layer:kv,get_compress_buffer=lambda *a:compressed,
                       c128_kv_pool=NS(kernel_page_size=16))
  fb=NS(batch_size=1,forward_mode=Mode())
  fm.kernel_metadata=b._kernel_metadata_from_parts(bs=1,actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,actual_seq_lengths_kv=fm.actual_seq_lengths_kv)
  events=[]
  def store_cache(**kw):
   events.append(('store',kw['swa_k'].shape[0]))
   kv.view(-1,512)[prefix:prefix+T].copy_(kw['swa_k'])
  b.store_cache=store_cache
  b.forward_core_compressor=lambda *a:events.append(('compressor_fixture',T))
  m=Model()
  def wqb(v):
   events.append(('wqb',v.shape[0]));return dynamic_apply(None,qlayer,v),None
  m.__dict__.update(fuse_wqa_wkv=False,wq_a=lambda v:(v,None),q_lora_rank=1024,
      use_fused_qk_norm_rope=False,q_norm=lambda v:torch_npu.npu_rms_norm(v,oneq,1e-6)[0],
      wq_b=wqb,wkv=lambda v:(torch.nn.functional.linear(v,wk),None),
      kv_norm=lambda v:torch_npu.npu_rms_norm(v,onek,1e-6)[0],q_rms_norm_ones=onek,
      eps=1e-6,n_local_heads=64,head_dim=512,qk_nope_head_dim=448,
      compress_ratio=ratio,indexer=None,compressor=object() if ratio else None,
      layer_id=0,n_local_groups=8,o_lora_rank=1024,wo_a=NS(weight=wo),
      attn_mqa=NS(layer_id=0,scaling=512**-.5),attn_tp_size=1,
      wo_b=lambda v,**kw:(dynamic_apply(None,blayer,v),None))
  sink=torch.randn(64,device='npu')
  m._local_attn_sink=lambda h:sink
  m._get_npu_rope_position_cache=lambda fb,p,dtype,inverse=False:(cos,nsin if inverse else sin)
  def full():
   q,_=m._forward_prepare(x,positions,fb,b)
   o=b.forward(q,q,q,m.attn_mqa,fb,compress_ratio=ratio,attn_sink=sink,save_kv_cache=False)
   Dsv4NpuRoPE.apply_rotary_mul_inplace(o,None,cos,nsin,448)
   projected=ns['_apply_wo_a_bf16_matmul'](o.view(T,8,4096),wo,is_decode=False,is_prefill=True)
   return m._project_wo_b(projected,False)
  def tiled(): return m._forward_npu_chunked_prefill(x,positions,fb,b,None,chunk,False)
  def run(fn):
   events.clear();gc.collect();torch.npu.synchronize();torch.npu.empty_cache()
   start=torch.npu.memory_allocated();torch.npu.reset_peak_memory_stats()
   y=fn();torch.npu.synchronize();peak=torch.npu.max_memory_allocated()-start
   cpu=y.cpu();del y
   assert b.forward_metadata is fm
   assert events.count(('store',T))==1
   if ratio: assert events.count(('compressor_fixture',T))==1
   log=list(events)
   times=[]
   if T>1000:
    for _ in range(8):
     begin=time.perf_counter();y=fn();torch.npu.synchronize();times.append((time.perf_counter()-begin)*1000);del y
   return cpu,peak,times,log
  ref,p1,t1,e1=run(full)
  got,p2,t2,e2=run(tiled)
  assert [n for k,n in e2 if k=='wqb']==[min(chunk,T-i) for i in range(0,T,chunk)]
  assert e2.index(('store',T)) < next(i for i,e in enumerate(e2) if e[0]=='wqb')
  got2,_,t22,_=run(tiled)
  ref2,_,t11,_=run(full)
  torch.testing.assert_close(got,got2,rtol=0,atol=0)
  torch.testing.assert_close(ref,ref2,rtol=0,atol=0)
  t1+=t11;t2+=t22
  diff=(ref.float()-got.float()).abs()
  record(dict(kind='production_methods',tokens=T,prefix=prefix,ratio=ratio,chunk=chunk,
      full_peak_bytes=p1,tiled_peak_bytes=p2,full_ms=t1,tiled_ms=t2,
      max_abs=diff.max().item(),rms=diff.square().mean().sqrt().item(),
      reference_rms=ref.float().square().mean().sqrt().item(),exact=torch.equal(ref,got),
      full_events=e1,tiled_events=e2))
  torch.testing.assert_close(ref,got,rtol=.02,atol=.002)
  try:
   with b.prefill_query_chunk(0,min(chunk,T),T,ratio): raise ValueError('expected test exception')
  except ValueError: pass
  assert b.forward_metadata is fm
 print('PRODUCTION_METHODS_PASS',flush=True)
