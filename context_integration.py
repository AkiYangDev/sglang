import json
import sys
import types
from pathlib import Path
import torch
import torch_npu
import context_init
from context_freqs import precompute_freqs_cis
from context_rope import get_npu_rope_cache_length

class Env:
    def __init__(self, value): self.value = value
    def get(self): return self.value

env = types.ModuleType('sglang.srt.environ')
env.envs = types.SimpleNamespace(SGLANG_SPEC_EXPANSION_SAFETY_FACTOR=Env(2),
    SGLANG_ROPE_CACHE_SAFETY_MARGIN=Env(256), SGLANG_ROPE_CACHE_ALIGN=Env(128))
runtime = types.ModuleType('sglang.srt.runtime_context')
runtime.get_spec = lambda: types.SimpleNamespace(speculative_num_steps=6)
runtime.max_speculative_num_draft_tokens = lambda: 6
sys.modules[env.__name__] = env
sys.modules[runtime.__name__] = runtime
context_init.precompute_freqs_cis = precompute_freqs_cis
context_init.get_npu_rope_cache_length = get_npu_rope_cache_length
context_init.get_rope_config = lambda config: (config.rope_theta, config.rope_scaling)
context_init.torch = torch
records=[]
for npu in (True, False):
    context_init._is_npu = npu
    for ratio in (0,4,128):
        cfg=types.SimpleNamespace(max_position_embeddings=8192,context_len=2048,
            rope_theta=10000,compress_rope_theta=160000,
            rope_scaling=dict(original_max_position_embeddings=65536,factor=16,beta_fast=32,beta_slow=1))
        obj=torch.nn.Module()
        obj.qk_rope_head_dim=64
        obj.compress_ratio=ratio
        context_init.initialize(obj,cfg)
        expected=2432 if npu else 8192
        assert obj.freqs_cis.shape==(expected,32)
        assert obj.rope_kwargs['original_seq_len']==(65536 if ratio else 0)
        assert obj.rope_kwargs['factor']==16
        assert cfg.max_position_embeddings==8192
        records.append([npu,ratio,expected])

# Exercise production frequency generation with the NPU default device too,
# covering the loader's device context, rather than only CPU -> NPU buffers.
torch.npu.set_device(0)
precompute_freqs_cis.cache_clear()
with torch.device('npu'):
    full=precompute_freqs_cis(64,8192,65536,160000,16,32,1)
    short=precompute_freqs_cis(64,2432,65536,160000,16,32,1)
assert torch.equal(full[:2432].cpu(),short.cpu())
result=dict(initializer_cases=records,npu_default_device_prefix_exact=True,
            scope='production initializer RoPE section; unrelated projections omitted')
Path(__file__).with_name('integration.json').write_text(json.dumps(result,indent=2))
print('INTEGRATION_PASS',json.dumps(result))
