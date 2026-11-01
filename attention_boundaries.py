"""Isolated real-op probe, not a full-model benchmark or production patch."""
import json
import math
import time
from pathlib import Path
import torch
import torch_npu
import sgl_kernel_npu, custom_ops

torch.npu.set_device(0)
torch.manual_seed(31)
root = Path(__file__).resolve().parent
results = []

def record(row):
    results.append(row)
    print(json.dumps(row), flush=True)
    (root / 'tile_results.json').write_text(json.dumps(results, indent=2))

def fixture(t, prefix, ratio):
    length = t + prefix
    kv = torch.randn(math.ceil(length / 128), 128, 1, 512, device='npu', dtype=torch.bfloat16) * .1
    bt = torch.arange(kv.shape[0], device='npu', dtype=torch.int32)[None]
    ck = cb = ids = None
    if ratio > 1:
        page = 32 if ratio == 4 else 16
        ck = torch.randn(max(1, math.ceil(length / ratio / page)), page, 1, 512, device='npu', dtype=torch.bfloat16) * .1
        cb = torch.arange(ck.shape[0], device='npu', dtype=torch.int32)[None]
        if ratio == 4:
            ids = torch.arange(512, device='npu', dtype=torch.int32).view(1, 1, 512).repeat(t, 1, 1)
            valid = (torch.arange(prefix + 1, prefix + t + 1, device='npu') // 4).view(t, 1, 1)
            ids.masked_fill_(ids >= valid, -1)
    sink = torch.randn(64, device='npu')
    def attention(q, begin, end):
        cu = torch.tensor([0, end - begin], dtype=torch.int32, device='npu')
        seq = torch.tensor([prefix + end], dtype=torch.int32, device='npu')
        metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=64, num_heads_kv=1, head_dim=512,
            layout_q='TND', layout_kv='PA_ND', cu_seqlens_q=cu.to('npu'),
            seqused_kv=seq, batch_size=1, cmp_topk=512 if ratio == 4 else 0,
            cmp_ratio=ratio, has_ori_kv=True, has_cmp_kv=ratio > 1,
            ori_win_left=127)
        return torch.ops.custom.npu_sparse_attn_sharedkv(
            q, ori_kv=kv, ori_block_table=bt, cmp_kv=ck, cmp_block_table=cb,
            cmp_sparse_indices=None if ids is None else ids[begin:end],
            cu_seqlens_q=cu.to('npu'), seqused_kv=seq.to('npu'), sinks=sink,
            metadata=metadata, softmax_scale=1 / math.sqrt(512), cmp_ratio=ratio,
            ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127,
            ori_win_right=0, layout_q='TND', layout_kv='PA_ND')[0]
    return attention

with torch.inference_mode():
    for ratio in [1, 4, 128]:
        for t, prefix, chunk in [(17, 0, 7), (129, 123, 32), (257, 255, 64)]:
            try:
                attn = fixture(t, prefix, ratio)
                q = torch.randn(t, 64, 512, dtype=torch.bfloat16, device='npu') * .1
                full = attn(q, 0, t)
                tiled = torch.cat([attn(q[b:b+chunk], b, min(b+chunk, t)) for b in range(0, t, chunk)])
                torch.npu.synchronize()
                err = (full.float() - tiled.float()).abs().max().item()
                exact = torch.equal(full, tiled)
                record(dict(kind='attention', ratio=ratio, tokens=t, prefix=prefix,
                            chunk=chunk, exact=exact, max_abs=err))
                torch.testing.assert_close(full, tiled, rtol=0, atol=0)
                del q, full, tiled, attn
            except Exception as exc:
                record(dict(kind='failure', ratio=ratio, tokens=t, error=str(exc)))
                raise
    print('ATTENTION_SEMANTICS_PASS', flush=True)
