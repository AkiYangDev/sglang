"""Isolated actual-source metadata validation, not a full SGLang model run."""
import ast
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
if DEVICE == 'npu':
    import torch_npu
    torch.npu.set_device(0)

class Base:
    def init_forward_metadata(self, batch):
        self.forward_metadata = NS(block_tables_swa=None, block_tables=None,
            actual_seq_lengths_kv=None, seq_lens_cpu_int=batch.seq_lens_cpu)

    def _init_dspark_sparse_metadata(self, batch):
        pass

    def _compute_kernel_metadata(self, batch):
        return {}

    def _build_npu_compress_metadata(self, batch):
        if batch.forward_mode.is_extend_without_speculative():
            self._build_npu_compress_metadata_prefill(batch)

    def _extend_prefix_lens_cpu(self, batch):
        return batch.prefix_cpu

class Mode:
    def __init__(self, mode): self.mode = mode
    def is_idle(self): return self.mode == 'idle'
    def is_extend(self): return self.mode == 'prefill'
    def is_extend_without_speculative(self): return self.mode == 'prefill'
    def is_draft_extend_v2(self): return self.mode == 'draft'
    def is_target_verify(self): return self.mode == 'verify'
    def is_decode(self): return self.mode == 'decode'

def load(name):
    tree = ast.parse((ROOT / name).read_text(encoding='utf-8'))
    methods = []
    for cls, method in [('CompressorAscendBackendMixin', '_build_npu_compress_metadata_prefill'),
                        ('DeepseekV4AscendAttnBackend', 'init_forward_metadata')]:
        parent = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        methods.append(next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == method))
    module = ast.Module(body=[ast.ClassDef(name='Actual', bases=[ast.Name(id='Base',ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    env = {'torch': torch, 'Base': Base, 'ForwardBatch': object}
    exec(compile(ast.fix_missing_locations(module), name, 'exec'), env)
    obj = env['Actual']()
    obj._dsv4_unique_compress_ratios = [4,128]
    obj._dsv4_compress_ratios = [4,128]
    obj.speculative_num_draft_tokens = 3
    return obj

def host(t): return t.detach().cpu() if isinstance(t, torch.Tensor) else t

def compare(a,b):
    assert vars(a).keys() == vars(b).keys()
    for k,v in vars(a).items():
        w = getattr(b,k)
        if isinstance(v,torch.Tensor):
            assert v.dtype == w.dtype and v.shape == w.shape, k
            assert torch.equal(host(v),host(w)), k
        else: assert v == w, k

def oracle(fm,lens,prefix):
    assert fm.actual_seq_lengths_q_pa_cpu.tolist() == [0] + list(itertools.accumulate(lens))
    assert fm.dsv4_max_input_capacity == max(1,max(lens,default=0))
    assert host(fm.start_pos).tolist() == prefix
    assert host(fm.seqused).tolist() == lens
    for ratio in (4,128):
        expected = [i*ratio for p,n in zip(prefix,lens) for i in range(p//ratio,(p+n)//ratio)]
        width = min(sum(lens),sum(lens)//ratio+len(lens))
        expected += [0]*(width-len(expected))
        assert host(getattr(fm,f'positions_cmp_padding_c{ratio}')).tolist() == expected

random.seed(31)
old,new = load('before.py'),load('after.py')
cpu_calls = {'before':0,'after':0}
d2h_calls = {'before':0,'after':0}
original_cpu = torch.Tensor.cpu
cases = [([0],[0]),([1,3,4,127,128,129],[0,1,3,127,128,255]),([0,0,0],[3,127,128])]
cases += [([random.randrange(0,260) for _ in range(bs)], [random.randrange(0,512) for _ in range(bs)])
          for bs in [1,2,4,8,16] for _ in range(5)]
passed=0
for lens,prefix in cases:
    for rep in ('list','int32','int64'):
        for device_prefix in (True,False):
            seq_cpu = torch.tensor([p+n for p,n in zip(prefix,lens)],dtype=torch.int32)
            batch = NS(seq_lens=seq_cpu.to(DEVICE), seq_lens_cpu=seq_cpu,
                extend_seq_lens_cpu=lens if rep=='list' else torch.tensor(lens,dtype=getattr(torch,rep)),
                positions=torch.arange(sum(lens),device=DEVICE), batch_size=len(lens),
                forward_mode=Mode('prefill'), prefix_cpu=prefix,
                extend_prefix_lens=torch.tensor(prefix,device=DEVICE) if device_prefix else None,
                out_cache_loc_dsv4=NS(out_c4_loc=torch.arange(2,device=DEVICE),out_c128_loc=torch.empty(0,device=DEVICE,dtype=torch.int64)))
            for label,backend in [('before',old),('after',new)]:
                def counted(t,*args,**kwargs):
                    cpu_calls[label]+=1
                    d2h_calls[label]+=int(t.device.type!='cpu')
                    return original_cpu(t,*args,**kwargs)
                with patch.object(torch.Tensor,'cpu',counted):
                    backend.init_forward_metadata(batch)
                oracle(backend.forward_metadata,lens,prefix)
            compare(old.forward_metadata,new.forward_metadata)
            passed+=1
# Reuse the same backend instances across non-prefill modes and changing batches.
controls=0
for mode in ('decode','verify','draft','idle'):
    batch.forward_mode=Mode(mode)
    batch.spec_info=NS(draft_token_num=3)
    old.init_forward_metadata(batch)
    new.init_forward_metadata(batch)
    compare(old.forward_metadata,new.forward_metadata)
    controls+=1
assert cpu_calls == {'before':passed,'after':0}
if DEVICE=='npu':
    assert d2h_calls == {'before':passed,'after':0}
    torch.npu.synchronize()
result = dict(device=DEVICE,torch=torch.__version__,prefill_cases_passed=passed,mode_controls_passed=controls,
    tensor_cpu_calls=cpu_calls,device_to_host_cpu_calls=d2h_calls,
    source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['before.py','after.py']},
    scope='Actual init and prefill methods; parent metadata, unrelated kernels and outer dispatch scaffolded. Independent CPU position oracle. No model, graph or timing test.')
(ROOT/f'RESULT_{DEVICE}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
