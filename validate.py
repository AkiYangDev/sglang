import os
import ast
import contextlib
import hashlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace as NS

os.environ['OMP_NUM_THREADS'] = '1'
import torch

ROOT = Path(__file__).parent
DEVICE = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
if DEVICE == 'npu':
    import torch_npu
    from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor, slot_map_lookup, unidex_copy_inplace
    import uuid
    _create_shm_tensor = create_shm_tensor
    def create_shm_tensor(**kw):
        kw['name'] = 'prefix_probe_' + uuid.uuid4().hex
        return _create_shm_tensor(**kw)
else:
    class Event:
        pass
    class Stream:
        def record_event(self, event): pass
        def wait_event(self, event): pass
    torch.npu = NS(Stream=Stream, Event=Event, current_stream=Stream, current_device=lambda: 0,
                   stream=lambda s: contextlib.nullcontext(), synchronize=lambda: None)
    def create_shm_tensor(shape, dtype, **kw):
        return torch.empty(shape, dtype=dtype), 0, 0
    def unidex_copy_inplace(src, dst, si, di, valid, sd, dd, **kw):
        dst.flatten(0, dd-1)[di[valid]] = src.flatten(0, sd-1)[si[valid]]
    def slot_map_lookup(table, reqs, tokens):
        valid = (tokens >= 0) & (tokens < table.shape[1])
        pos = table[reqs.long()[:, None], tokens.long().clamp(0, table.shape[1]-1)]
        return valid & (pos >= 0), pos

class Pool:
    start_layer = 0
    layer_num = 1
    head_num = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    store_dtype = torch.bfloat16
    size = 64
    page_size = 16
    kv_page_padding = 1

def load(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    names = {'SparseKVCacheManager', '_record_stream_event', '_wait_stream_event',
             'normalize_batch_topk_indices', '_build_hit_src_dst_index', '_build_miss_src_dst_index'}
    nodes = [n for n in tree.body if getattr(n, 'name', '') in names]
    ns = dict(torch=torch, logger=logging.getLogger('probe'), MLATokenToKVPool=Pool,
              TorchMemorySaverAdapter=NS(create=lambda **kw: NS(region=lambda *a: contextlib.nullcontext())),
              GPU_MEMORY_TYPE_KV_CACHE='kv', create_shm_tensor=create_shm_tensor,
              slot_map_lookup=slot_map_lookup, unidex_copy_inplace=unidex_copy_inplace)
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)
    return ns['SparseKVCacheManager']

def tensor(x): return torch.tensor(x, device=DEVICE, dtype=torch.long)
def make_batch(rid, prefix, slots, decode=False):
    n = len(slots)
    return NS(req_pool_indices=tensor([rid]), seq_lens=tensor([prefix+n]),
              forward_mode=NS(is_decode=lambda: decode), out_cache_loc=tensor(slots),
              extend_seq_lens=tensor([n]), extend_prefix_lens=tensor([prefix]), extend_seq_lens_cpu=[n])

def run(source):
    cls = load(source)
    table = torch.zeros((3,2048), dtype=torch.int32, device=DEVICE)
    req_pool = NS(req_to_token=table, max_context_len=2048, device=DEVICE, alloc=lambda reqs: [1]*len(reqs))
    m = cls(req_pool, NS(get_kvcache=lambda: Pool()), sparse_context_len=2048)
    m.host_kv_buffer[0].fill_(-99)
    layer = NS(layer_id=0)
    stream = torch.npu.current_stream()
    def write(rid, prefix, slots, values, decode=False):
        b = make_batch(rid, prefix, slots, decode)
        table[rid,prefix:prefix+len(slots)] = tensor(slots).int()
        k = tensor(values).to(torch.bfloat16)[:,None,None].expand(-1,1,512).contiguous()
        pe = (tensor(values)+100).to(torch.bfloat16)[:,None,None].expand(-1,1,64).contiguous()
        m.offload_v2(k,pe,layer,b,stream)
        torch.npu.synchronize()
        return b
    def read(b):
        k,p = m.get_forward_kv(layer,b,stream)
        torch.npu.synchronize()
        return k[:,0,0].float().cpu().tolist(), p[:,0,0].float().cpu().tolist()
    rows=[]
    b=write(0,0,[16,17,18,19],[10,11,12,13])
    rows.append(dict(case='fresh_prefill',actual=read(b)[0],expected=[10,11,12,13]))
    b=write(0,4,[20],[14])
    rows.append(dict(case='same_request_chunk',actual=read(b)[0],expected=[10,11,12,13,14]))
    # Keep radix-owned slots 16..19 while another request uses a different row.
    table[1,:4]=table[0,:4]
    m.reset_requests([1])
    b=write(1,4,[32],[24],decode=True)
    rows.append(dict(case='shared_prefix_new_request',actual=read(b)[0],expected=[10,11,12,13,24]))
    for name in ['decode_host_misses','decode_device_hits']:
        selected=torch.zeros((1,2048,1,576),device=DEVICE,dtype=torch.bfloat16)
        topk = torch.full((1,2048), -1, device=DEVICE, dtype=torch.long)
        topk[0,:3] = tensor([0,3,4])
        m.materialize_selected_kv(layer,b,topk,selected,stream)
        stream.wait_event(m.hit_done);stream.wait_event(m.miss_done)
        stream.wait_event(m.refill_done);stream.wait_event(m.slot_map_done)
        torch.npu.synchronize()
        rows.append(dict(case=name,actual=selected[0,:4,0,0].float().cpu().tolist(),expected=[10,13,24,0]))
        assert torch.count_nonzero(selected[:,3:]).item() == 0
    # Original request row is reused without changing radix-owned host slots.
    m.reset_requests([0])
    write(0,0,[40,41,42,43],[80,81,82,83])
    m.reset_requests([1])
    rows.append(dict(case='source_request_row_reused',actual=read(b)[0],expected=[10,11,12,13,24]))
    # Model allocator release/reallocation: overwrite freed physical slots.
    m.reset_requests([1])
    b=write(1,0,[16,17,18,19],[30,31,32,33])
    rows.append(dict(case='physical_slots_reallocated',actual=read(b)[0],expected=[30,31,32,33]))
    try:
        req_pool.alloc([NS(kv=NS(req_pool_idx=None))])
        allocation_ok=True
    except AttributeError:
        allocation_ok=False
    for r in rows:r['pass']=r['actual']==r['expected']
    graph_cases = []
    if DEVICE == 'npu' and source.name == 'after.py':
        # Capture only the changed decode write path, replay with a new slot.
        gb = make_batch(1,4,[32],True)
        gk = torch.full((1,1,512), 41., device=DEVICE, dtype=torch.bfloat16)
        gp = torch.full((1,1,64), 141., device=DEVICE, dtype=torch.bfloat16)
        gs = torch.npu.Stream()
        gs.wait_stream(stream)
        with torch.npu.stream(gs):
            m.offload_v2(gk,gp,layer,gb,gs)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=gs):
            m.offload_v2(gk,gp,layer,gb,gs)
        gb.out_cache_loc.fill_(33)
        gk.fill_(42);gp.fill_(142)
        graph.replay()
        torch.npu.synchronize()
        flat = m.host_kv_buffer[0].flatten(0,1)
        actual=[flat[32,0,0].item(),flat[33,0,0].item()]
        graph_cases.append(dict(case='graph_write_replays_new_physical_slot',actual=actual,expected=[41,42],pass_=actual==[41,42]))
        assert actual==[41,42]
    return dict(cases=rows, allocation_hook_pass=allocation_ok,
                graph_cases=graph_cases,host_shape=list(m.host_kv_buffer[0].shape),sha256=hashlib.sha256(source.read_bytes()).hexdigest())

result=dict(device=DEVICE,base='67bd749ada5bebf3f1ed9be16a814a3243f348bd',
            scope='Actual source constructors and methods; allocator/model scaffolding. Real Ascend shared-memory copy and slot lookup on NPU; reference operations on CPU. No full model run.',
            before=run(ROOT/'before.py'),after=run(ROOT/'after.py'))
assert all(r['pass'] for r in result['after']['cases'])
assert result['after']['allocation_hook_pass']
(ROOT/f'RESULT_{DEVICE}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

