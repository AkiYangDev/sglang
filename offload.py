"""Usage: python offload.py /path/to/sglang (standard library only)."""
from __future__ import annotations

import ast
import csv
import statistics
import sys
import time
import tracemalloc
from array import array
from pathlib import Path
from types import SimpleNamespace as NS

src = Path(sys.argv[1]) / "python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py"
original = src.read_text(encoding="utf8")
copy_line = "        all_tokens = req.origin_input_ids + req.output_ids[:-1]"
length_line = "len(all_tokens) - prefill_offloaded_len"
slice_line = "        incremental_tokens = all_tokens[start:end]"
assert all(original.count(line) == 1 for line in (copy_line, length_line, slice_line))
control = original.replace(copy_line, "").replace(length_line,
    "len(req.origin_input_ids) + len(req.output_ids) - 1 - prefill_offloaded_len").replace(
    slice_line, copy_line + "\n" + slice_line)

def load(source):
    tree = ast.parse(source)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and n.name in {"offload_kv_cache", "_prefill_offloaded_len"}]
    assert len(nodes) == 2
    scope = dict(torch=NS(int64=int), time=time, OffloadedState=lambda last_hash: NS(inc_len=0, last_hash=last_hash))
    exec(compile(ast.Module(nodes, []), str(src), "exec"), scope)
    return scope

class Req:
    pass

class IndexRow:
    def dim(self):
        return 1
    def numel(self):
        return 1_000_000
    def __getitem__(self, key):
        return NS(to=lambda dtype: (key.start, key.stop))

def fixture(scope, length, outputs, submitted, fresh=False):
    req = Req()
    req.kv = NS(req_pool_idx=0)
    req.origin_input_ids = array("q", range(length))
    req.output_ids = array("q", range(length, length + outputs))
    manager = NS(page_size=64, offload_stride=64, decode_host_mem_pool=True,
        offloaded_state={} if fresh else {req: NS(inc_len=submitted, last_hash="seed")},
        req_to_token_pool=NS(req_to_token=[IndexRow()]), request_counter=0, ongoing_offload={})
    manager._prefill_offloaded_len = lambda r: scope["_prefill_offloaded_len"](manager, r)
    manager._compute_prefix_hash = lambda r, ids: ["hash"] * (len(ids) // manager.page_size)
    manager._resolve_offload_transfers = lambda indices: (indices, None)
    manager._mark_offload_started = lambda r: None
    manager.cache_controller = NS(write=lambda **kwargs: kwargs["device_indices"])
    return manager, req

upstream, delayed = load(original), load(control)
def run(scope, length, outputs, submitted=0, fresh=False):
    manager, req = fixture(scope, length, outputs, submitted, fresh)
    tracemalloc.start()
    did_write = scope["offload_kv_cache"](manager, req)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    chunk = manager.ongoing_offload.get(1)
    return (did_write, manager.offloaded_state[req].inc_len, manager.offloaded_state[req].last_hash,
            (chunk[1], list(chunk[2])) if chunk else None), current, peak

if __name__ == "__main__":
    # Check the final-token exclusion, an unaligned input tail, and a second stride.
    for length in (127, 128, 129):
        for outputs in (1, 2, 64, 65, 66, 130):
            for submitted, fresh in [(0, True), (0, False), (64, False)]:
                start = (length // 64) * 64 + submitted
                available = length + outputs - 1 - start
                if available < 0:
                    continue
                end = start + available // 64 * 64
                result = run(upstream, length, outputs, submitted, fresh)[0]
                assert result == run(delayed, length, outputs, submitted, fresh)[0]
                assert result[0] == (end > start)
                assert result[3] == (((start, end), list(range(start, end))) if end > start else None)
    with open("offload.csv", "w", newline="") as dest:
        writer = csv.writer(dest)
        writer.writerow(["prompt_tokens", "submitted_tokens", "output_tokens", "variant", "repeat", "retained_bytes", "peak_bytes"])
        for length in (4096, 16384, 65536, 131072):
            for submitted in (0, 4096, 16384):
                for name, scope in [("upstream", upstream), ("defer_concat", delayed)]:
                    for repeat in range(1, 4):
                        result, current, peak = run(scope, length, submitted + 2, submitted)
                        assert not result[0] and result[1] == submitted
                        writer.writerow([length, submitted, submitted + 2, name, repeat, current, peak])
    print("Boundary cases passed. Wrote offload.csv")
