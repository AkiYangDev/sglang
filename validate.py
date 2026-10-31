import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
if DEVICE == 'npu':
    import torch_npu
    torch.npu.set_device(0)
else:
    torch.npu = SimpleNamespace(synchronize=lambda: None)


def load(name):
    tree = ast.parse((ROOT / name).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUMLATokenToKVPool')
    cls.bases = []
    methods = {'_copy_indices_for_buffer', '_cpu_offload_indices', '_get_cpu_offload_layer_buffers',
               '_chunk_copy_npu_to_cpu', 'get_cpu_copy', 'load_cpu_copy'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_mla_fia_nz_scatter_indices')
    dcp = next(n for n in ast.parse((ROOT/'layout.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'localize_dcp_indices')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    ns = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, helper, dcp, cls], type_ignores=[])), name, 'exec'), ns)
    return ns['NPUMLATokenToKVPool']


def logical(buf, nz):
    pages, page_size, _, dim = buf.shape
    if nz:
        return buf.view(pages, dim//16, page_size, 16).permute(0, 2, 1, 3).reshape(pages*page_size, 1, dim)
    return buf.flatten(0, 1).clone()


def physical(buf, nz, pages=24, page_size=128):
    dim = buf.shape[-1]
    if nz:
        return buf.view(pages, page_size, dim//16, 16).permute(0, 2, 1, 3).contiguous().view(pages, page_size, 1, dim)
    return buf.view(pages, page_size, 1, dim)


def case(cls, nz, mode, length):
    p = cls()
    p.page_size, p.start_layer, p.layer_num = 128, 8, 2
    p.kv_lora_rank, p.qk_rope_head_dim = 512, 64
    p.dcp_size = 2 if mode in ('dcp0', 'dcp1', 'draft') else 1
    p.dcp_rank = 1 if mode == 'dcp1' else 0
    p.is_draft_worker = mode == 'draft'
    p.dsa_kv_cache_store_fp8 = mode == 'packed'
    p.use_fia_nz = nz
    p.indexer_layer_id_to_slot = {9: 0}
    p.cpu_offloading_chunk_size = 64
    packed = p.dsa_kv_cache_store_fp8
    dims = [576, 0] if packed else [512, 64]
    def alloc(dim, dtype=torch.bfloat16):
        return torch.randint(0, 64, (24, 128, 1, dim)).to(dtype).to(DEVICE)
    p.k_buffer = [alloc(dims[0], torch.uint8 if packed else torch.bfloat16) for _ in range(2)]
    p.v_buffer = [alloc(dims[1]) for _ in range(2)]
    p.index_k_buffer = [alloc(128, torch.float8_e4m3fn if packed else torch.bfloat16)]
    p.index_k_scale_buffer = [alloc(4, torch.float32)] if packed else None
    src = torch.arange(2*p.dcp_size*128+3, 2*p.dcp_size*128+3+length)
    # DCP restore preserves ownership in every CPU chunk.
    dst = (src if mode in ('dcp0', 'dcp1') else src.flip(0)) + 6*p.dcp_size*128
    targets = []
    for layer in range(2):
        bufs = [p.k_buffer[layer], p.v_buffer[layer]]
        if layer == 1:
            bufs += p.index_k_buffer
            if packed:
                bufs += p.index_k_scale_buffer
        for i, buf in enumerate(bufs):
            tiled = nz and not packed and i < 2
            raw = buf.cpu()
            if raw.dtype == torch.float8_e4m3fn:
                raw = raw.view(torch.uint8)
            values = logical(raw, tiled)
            # Independent DCP owner math, not the implementation under test.
            si, di = src, dst
            if i < 2 and not p.is_draft_worker and p.dcp_size > 1:
                def local(ids):
                    ids = ids[(ids//128) % p.dcp_size == p.dcp_rank]
                    return (ids//(128*p.dcp_size))*128 + ids%128
                si, di = local(src), local(dst)
            expected = torch.full_like(values, 7)
            expected[di] = values[si]
            targets.append((buf, physical(expected, tiled)))
    backup = p.get_cpu_copy(src.to(DEVICE))
    for buf, _ in targets:
        if buf.dtype == torch.float8_e4m3fn:
            buf.view(torch.uint8).fill_(7)
        else:
            buf.fill_(7)
    p.load_cpu_copy(backup, dst.to(DEVICE))
    torch.npu.synchronize()
    mismatches = []
    for buf, expected in targets:
        actual = buf.cpu()
        if actual.dtype == torch.float8_e4m3fn:
            actual = actual.view(torch.uint8)
        mismatches.append(int((actual != expected).sum()))
    return {'nz': nz, 'mode': mode, 'tokens': length, 'mismatches': mismatches,
            'pass': not any(mismatches)}


torch.manual_seed(20261031)
torch.set_num_threads(1)
result = {'device': DEVICE, 'scope': 'actual source copy methods; independent whole-buffer layout oracle; no model or distributed runtime', 'runs': {}}
for source in ('before.py', 'after.py'):
    cls = load(source)
    rows = [case(cls, nz, mode, length) for nz in (False, True)
            for mode in ('plain', 'dcp0', 'dcp1', 'draft', 'packed')
            for length in (0, 1, 17, 127, 128, 129, 257)]
    result['runs'][source] = {'sha256': hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                              'passed': sum(r['pass'] for r in rows), 'total': len(rows), 'cases': rows}
    print(source, result['runs'][source]['passed'], '/', len(rows), flush=True)
(ROOT/f'RESULT_{DEVICE}.json').write_text(json.dumps(result, indent=2))
assert result['runs']['after.py']['passed'] == result['runs']['after.py']['total']
assert result['runs']['before.py']['passed'] < result['runs']['before.py']['total']
print('PASS', flush=True)
