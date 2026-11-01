from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
from copy import copy
import torch
root=Path(__file__).parent
ns=dict(contextmanager=contextmanager,copy=copy)
exec((root/'backend_methods.py').read_text(),ns)
class Backend: prefill_query_chunk=ns['prefill_query_chunk']
calls=[]
b=Backend()
b._kernel_metadata_from_parts=lambda **kw:(calls.append(kw) or {'c4a_metadata':torch.tensor([len(calls)])})
def metadata(length):
 return NS(actual_seq_lengths_q_pa=torch.tensor([0,8]),actual_seq_lengths_kv=torch.tensor([length]),c4_topk_indices=torch.arange(16).reshape(8,2))
fm=b.forward_metadata=metadata(20)
with b.prefill_query_chunk(2,5,8,4):
 assert b.forward_metadata.actual_seq_lengths_q_pa.tolist()==[0,3]
 assert b.forward_metadata.actual_seq_lengths_kv.tolist()==[17]
 assert torch.equal(b.forward_metadata.c4_topk_indices,fm.c4_topk_indices[2:5])
assert b.forward_metadata is fm
fm.c4_topk_indices=torch.full((8,2),41)
with b.prefill_query_chunk(2,5,8,4):
 assert len(calls)==1
 assert torch.all(b.forward_metadata.c4_topk_indices==41)
fm2=b.forward_metadata=metadata(30)
with b.prefill_query_chunk(2,5,8,4):
 assert len(calls)==2
 assert b.forward_metadata.actual_seq_lengths_kv.tolist()==[27]
assert b.forward_metadata is fm2
def fail(**kw): raise RuntimeError('expected')
b._kernel_metadata_from_parts=fail
try:
 with b.prefill_query_chunk(0,2,8,4): pass
except RuntimeError: pass
else: raise AssertionError('failure not propagated')
assert b.forward_metadata is fm2
print('METADATA_CACHE_AND_FAILURE_RESTORE_PASS')

state=dict(npu=True,chunk=4096,a5=False,cp=False,capture=False,graph=False)
backend=NS(_is_dspark_draft_worker=False)
envs=NS(SGLANG_DSV4_NPU_PREFILL_CHUNK_SIZE=NS(get=lambda:state['chunk']))
ns=dict(torch=torch,_is_npu=True,envs=envs,get_attn_tp_context=lambda:NS(input_scattered=False),
        get_attn_backend=lambda:backend,is_npu_arch35=lambda:state['a5'],
        is_cp_active=lambda fb:state['cp'],get_is_capture_mode=lambda:state['capture'],
        is_in_breakable_cuda_graph=lambda:state['graph'])
exec((root/'dispatch_method.py').read_text(),ns)
def reset():
 state.update(chunk=4096,a5=False,cp=False,capture=False,graph=False)
 ns['_is_npu']=True; backend._is_dspark_draft_worker=False
 m=NS(is_dsv41=False,attn_tp_size=1,n_local_heads=64,wo_a_fp8=False,
      use_npu_arch35_mxfp8_wo_a=False,wo_a=NS(weight=NS(dtype=torch.bfloat16)),
      _forward_npu_chunked_prefill=lambda *a:'tiled')
 x=NS(shape=(8192,4096),dtype=torch.bfloat16)
 fb=NS(batch_size=1,forward_mode=NS(is_extend_without_speculative=lambda:True))
 return m,x,fb
cases=[lambda m,x,f:None,lambda m,x,f:state.update(chunk=0),
 lambda m,x,f:state.update(a5=True),lambda m,x,f:state.update(cp=True),
 lambda m,x,f:state.update(capture=True),lambda m,x,f:state.update(graph=True),
 lambda m,x,f:setattr(m,'is_dsv41',True),lambda m,x,f:setattr(m,'attn_tp_size',2),
 lambda m,x,f:setattr(m,'n_local_heads',128),lambda m,x,f:setattr(m,'wo_a_fp8',True),
 lambda m,x,f:setattr(m,'use_npu_arch35_mxfp8_wo_a',True),
 lambda m,x,f:setattr(f,'batch_size',2),lambda m,x,f:setattr(x,'shape',(4096,4096)),
 lambda m,x,f:setattr(backend,'_is_dspark_draft_worker',True),
 lambda m,x,f:setattr(f.forward_mode,'is_extend_without_speculative',lambda:False),
 lambda m,x,f:ns.update(_is_npu=False),lambda m,x,f:setattr(m,'wo_a',NS()),lambda m,x,f:setattr(x,'dtype',torch.float16)]
for i,change in enumerate(cases):
 m,x,fb=reset();change(m,x,fb)
 assert ns['forward'](m,x,None,fb)==('tiled' if i==0 else 'fallback'),i
print('ACTUAL_DISPATCH_18_CASES_PASS')
