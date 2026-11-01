import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

root = Path(__file__).resolve().parent
device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
if device == 'npu':
    import torch_npu
    torch.npu.set_device(0)

names = ['_build_dsv4_graph_replay_ctx', '_refresh_graph_seq_metadata']
arms = []
for filename in ['before.py', 'after.py']:
    tree = ast.parse((root / filename).read_text(encoding='utf-8'))
    ns = {'torch': torch, 'ForwardBatch': object, 'SimpleNamespace': NS}
    for name in names:
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, 'exec'), ns)
    arms.append(ns)

class Mode:
    def __init__(self, name): self.name = name
    def is_target_verify(self): return self.name == 'verify'
    def is_draft_extend_v2(self): return self.name == 'draft'
    def is_idle(self): return self.name == 'idle'

native_to = torch.Tensor.to
passed = saved = nonempty_saved = 0
for mode, algorithm, bs, width, compression, idle in itertools.product(
    ['verify', 'decode', 'draft'], ['eagle', 'dspark', 'explicit'], [0, 1, 4],
    [1, 5], ['none', 'c4', 'c128', 'both'], [False, True]
):
    live = torch.tensor([1, 127, 128, 0][:bs], dtype=torch.int64)
    final = live + width
    host = final if mode == 'verify' and algorithm != 'eagle' else live
    batch = NS(forward_mode=Mode(mode), actual_forward_mode=Mode('idle') if idle else None,
               batch_size=bs, num_padding=int(bs == 4), seq_lens_cpu=host,
               seq_lens=host.to(device), spec_info=NS(live_seq_lens_cpu=live if algorithm == 'explicit' else None))
    counts = []
    for arm in arms:
        obj = NS(_dsv4_graph_tokens_per_req=width, _is_dspark_algorithm=algorithm == 'dspark',
                 _dsv4_has_c4=compression in ['c4', 'both'], _dsv4_has_c128=compression in ['c128', 'both'],
                 forward_metadata=NS(actual_seq_lengths_kv=torch.full((bs,), -99, dtype=torch.int32, device=device)))
        counter = [0]
        def counted(t, *args, **kwargs):
            if t.device.type == 'cpu' and kwargs.get('device') is not None:
                counter[0] += 1
            return native_to(t, *args, **kwargs)
        with patch.object(torch.Tensor, 'to', counted):
            ctx = arm[names[0]](obj, batch)
            arm[names[1]](obj, ctx)
        expected = live
        if mode == 'verify':
            expected = torch.zeros_like(live) if idle else torch.where(live > 0, final, live)
        expected_host = expected if mode == 'verify' else expected.clamp(min=1)
        assert torch.equal(obj.forward_metadata.actual_seq_lengths_kv.cpu(), expected.clamp(min=1).int()), (mode, algorithm, bs, width, compression, idle)
        assert torch.equal(obj.forward_metadata.seq_lens_cpu_int, expected_host.int())
        counts.append(counter[0])
    reduction = int(mode == 'verify' and compression != 'none' and not idle)
    assert counts[0] - counts[1] == reduction
    saved += reduction
    nonempty_saved += reduction * int(bs > 0)
    passed += 1
out = dict(device=device, torch_version=torch.__version__, cases_passed=passed,
           conversion_calls_removed=saved, nonempty_cases_with_one_call_removed=nonempty_saved,
           source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in ['before.py', 'after.py']},
           scope='Actual before/after methods with metadata fixtures and independent length oracle. CPU conversion-call counts are not physical H2D measurements. No captured graph, model or timing test.')
(root / f'RESULT_fix_{device}.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out), flush=True)
