import gc
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
if device == 'npu':
    import torch_npu
    torch.npu.set_device(0)
root = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Dsv4NpuRoPE

old, new = load('before'), load('after')
torch.manual_seed(123)
checks = 0
for length in (17, 257):
    for dim in (8, 64):
        freq_cpu = torch.complex(torch.randn(length, dim // 2), torch.randn(length, dim // 2))
        freq = freq_cpu.to(device)
        for owned in (False, True):
            a = old(freq, torch.nn.Module() if owned else None)
            b = new(freq, torch.nn.Module() if owned else None)
            try:
                b.ensure_tables(torch.float32, allow_build=False)
                raise AssertionError('missing table did not fail')
            except RuntimeError:
                pass
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                aa = a.ensure_tables(dtype)
                bb = b.ensure_tables(dtype)
                for x, y in zip(aa, bb):
                    assert torch.equal(x.cpu(), y.cpu())
                again = b.ensure_tables(dtype, allow_build=False)
                assert all(x.data_ptr() == y.data_ptr() for x, y in zip(bb, again))
                for inverse in (False, True):
                    for ids in ([0, length - 1, 0, 3], [2, 1, 4, 2], []):
                        pos = torch.tensor(ids, dtype=torch.long, device=device)
                        got = b.get_cos_sin(pos, dtype, inverse=inverse, view_4d=True, allow_build=False)
                        ref = a.get_cos_sin(pos, dtype, inverse=inverse, view_4d=True, allow_build=False)
                        assert all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(got, ref))
                        checks += 1
            if owned:
                assert not b.rotary_emb.state_dict()
            assert not hasattr(b, '_real_imag')
del a, b, aa, bb, again, got, ref, freq, pos
gc.collect()

result = dict(device=device, checks=checks, exact=True, memory=[], graph=None)
if device == 'npu':
    (root / 'allocator_api.txt').write_text(inspect.getsource(torch.npu.memory_allocated) + inspect.getsource(torch.npu.memory_reserved))
    # Same process, isolated lifetimes; compare allocated bytes, not reserved cache.
    for cls in (old, new):
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        start = torch.npu.memory_allocated()
        keep = []
        for table in range(2):
            f = torch.complex(torch.randn(262144, 32), torch.randn(262144, 32)).to(device)
            instance = cls(f, torch.nn.Module())
            instance.ensure_tables(torch.float32)
            instance.ensure_tables(torch.bfloat16)
            keep.append(instance)
        torch.npu.synchronize()
        live = torch.npu.memory_allocated() - start
        result['memory'].append(dict(implementation=cls.__module__, positions=262144, tables=2, live_bytes=live))
        del instance, f, keep
        gc.collect()
        torch.npu.synchronize()
    saving = result['memory'][0]['live_bytes'] - result['memory'][1]['live_bytes']
    expected_payload = 2 * 262144 * 64 * 4
    # The NPU allocator pads each released allocation; retain the measured delta.
    assert expected_payload <= saving <= expected_payload + 4096, saving
    result['saved_payload_bytes'] = expected_payload
    result['allocator_padding_delta_bytes'] = saving - expected_payload
    result['measured_saved_bytes'] = saving
    result['projected_saved_bytes_1M_two_tables'] = 2 * 1048576 * 64 * 4

    f = torch.complex(torch.randn(257, 32), torch.randn(257, 32)).to(device)
    b = new(f, torch.nn.Module())
    b.ensure_tables(torch.float32)
    b.ensure_tables(torch.bfloat16)
    positions = torch.tensor([0, 1, 256, 3], device=device)
    # Warm up before capture. Capture only the unchanged no-build forward.
    for _ in range(3):
        b.get_cos_sin(positions, torch.bfloat16, allow_build=False)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = b.get_cos_sin(positions, torch.bfloat16, allow_build=False)
    for ids in ([2, 4, 255, 1], [256, 0, 17, 17]):
        positions.copy_(torch.tensor(ids, device=device))
        graph.replay()
        torch.npu.synchronize()
        expected = b.get_cos_sin(positions, torch.bfloat16, allow_build=False)
        assert all(torch.equal(x.cpu(), y.cpu()) for x, y in zip(outputs, expected))
    result['graph'] = '2 changed-position replays exact'
    result['versions'] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__)

(root / ('RESULT_' + device + '.json')).write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
