import gc
import json
import sys
import types
from pathlib import Path

import torch
import torch_npu

from context_freqs import precompute_freqs_cis
from context_rope import Dsv4NpuRoPE, get_npu_rope_cache_length

torch.npu.set_device(0)
torch.set_num_threads(1)
root = Path(__file__).parent
result = {"versions": [torch.__version__, torch_npu.__version__]}

# Execute the production length selector with controlled runtime configuration.
class Env:
    def __init__(self, value): self.value = value
    def get(self): return self.value

env = types.ModuleType('sglang.srt.environ')
env.envs = types.SimpleNamespace(
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR=Env(2),
    SGLANG_ROPE_CACHE_SAFETY_MARGIN=Env(256),
    SGLANG_ROPE_CACHE_ALIGN=Env(128),
)
runtime = types.ModuleType('sglang.srt.runtime_context')
saved = {k: sys.modules.get(k) for k in (env.__name__, runtime.__name__)}
sys.modules[env.__name__] = env
sys.modules[runtime.__name__] = runtime
controls = []
for context, steps, draft, expected in [
    (None, 0, 0, 1048576), (1048576, 6, 6, 1048576),
    (2097152, 6, 6, 1048576), (32768, 0, 0, 33024),
    (139264, 6, 6, 139648), (139264, 6, 16, 140032),
    (1048500, 6, 6, 1048576), (1, 0, 0, 384),
]:
    runtime.get_spec = lambda: types.SimpleNamespace(speculative_num_steps=steps)
    runtime.max_speculative_num_draft_tokens = lambda: draft
    config = types.SimpleNamespace(max_position_embeddings=1048576, context_len=context)
    before = vars(config).copy()
    got = get_npu_rope_cache_length(config)
    assert got == expected, (context, got, expected)
    assert vars(config) == before
    controls.append([context, steps, draft, got])
for k, v in saved.items():
    if v is None: sys.modules.pop(k)
    else: sys.modules[k] = v
result['controls'] = controls

def freqs(length, compressed):
    # Same production precompute function; move the resulting complex buffer
    # to NPU before running the actual NPU table builder and gather.
    return precompute_freqs_cis(64, length, 65536 if compressed else 0,
                               160000 if compressed else 10000, 16, 32, 1)

def clean():
    precompute_freqs_cis.cache_clear()
    Dsv4NpuRoPE._instances.clear()
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()

checks = 0
for compressed in (False, True):
    full_cpu = freqs(1048576, compressed)
    for length in (384, 33024, 139648):
        short_cpu = freqs(length, compressed)
        assert torch.equal(full_cpu[:length], short_cpu)
        # Full table on CPU is an independent prefix oracle for every row;
        # compare all NPU cache rows, not just a few gathered positions.
        short = short_cpu.to('npu')
        rope = Dsv4NpuRoPE(short, torch.nn.Module())
        for dtype in (torch.float32, torch.bfloat16):
            tables = rope.ensure_tables(dtype)
            oracle = [v.repeat_interleave(2, dim=-1).to(dtype)
                      for v in (full_cpu[:length].real, full_cpu[:length].imag)]
            assert all(torch.equal(x.cpu(), y) for x, y in zip(tables, oracle))
            for inverse in (False, True):
                ids = [0, min(127, length-1), length-1, length-2, 0]
                positions = torch.tensor(ids, device='npu')
                got = rope.get_cos_sin(positions, dtype, inverse=inverse, allow_build=False)
                ref = [x[ids] for x in oracle]
                if inverse: ref[1] = -ref[1]
                assert all(torch.equal(x.cpu(), y) for x,y in zip(got, ref))
                checks += 1
        if length == 139648:
            for _ in range(3): rope.get_cos_sin(positions, torch.bfloat16, allow_build=False)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                outputs = rope.get_cos_sin(positions, torch.bfloat16, allow_build=False)
            for ids in ([139263,139264,139269,139647,0], [1,127,128,65535,139646]):
                positions.copy_(torch.tensor(ids,device='npu'))
                graph.replay()
                torch.npu.synchronize()
                expected = rope.get_cos_sin(positions, torch.bfloat16, allow_build=False)
                assert all(torch.equal(x.cpu(),y.cpu()) for x,y in zip(outputs,expected))
            del graph, outputs, expected
        del rope, short, tables, got, positions
    del full_cpu, short_cpu
clean()
result['gather_cases_exact'] = checks
result['full_table_pairs_exact'] = 12
result['changed_position_graph_replays_exact'] = 4
print('PRECISION_PASS', flush=True)

# Compare isolated lifetimes with two shared frequency families, using the
# production main table builder. Also measure stacking the previous PR by
# releasing its otherwise retained real/imag copies after initialization.
measurements = []
for length in (1048576, 139648, 33024):
    clean()
    start = torch.npu.memory_allocated()
    keep = []
    for compressed in (False, True):
        f = freqs(length, compressed).to('npu')
        rope = Dsv4NpuRoPE(f, torch.nn.Module())
        rope.ensure_tables(torch.float32)
        rope.ensure_tables(torch.bfloat16)
        keep.append(rope)
    torch.npu.synchronize()
    main_bytes = torch.npu.memory_allocated() - start
    for rope in keep: rope._real_imag = None
    torch.npu.synchronize()
    no_duplicate_bytes = torch.npu.memory_allocated() - start
    measurements.append(dict(rows=length, main_bytes=main_bytes,
                             without_real_imag_copies_bytes=no_duplicate_bytes))
    del keep, rope, f
    print(measurements[-1], flush=True)
result['allocation'] = measurements
result['scope'] = 'RoPE tables only; synthetic runtime config; no full model or compressor execution'
(root/'result.json').write_text(json.dumps(result, indent=2))
print('ALL_PASS', json.dumps(result), flush=True)
