# NPU DeepSeek-V4 compression-position range

Base: 25c3fd83085e76d7ba3694900a402f964e1533bf.

`python validate.py cpu` / `python validate.py npu` extracts the actual prefill
metadata helper from before.py and after.py. 297 cases cover C4/C128 individually
and together, prefix/compression boundaries, device/host prefix inputs, empty
queries and entirely empty batches. Uses an independent Python position oracle.
Only the helper runs, with small metadata fixtures; no serving stack or model.

BENCH_npu.json reuses the preceding same-source microbenchmark. validate.py checks
that after.py has exactly the benchmarked expression change. benchmark.py and
source.py are the original benchmark inputs. To rerun, copy them to a separate
directory: benchmark.py writes RESULT_npu.json (do not overwrite correctness
results here). On one Ascend NPU: ABBA order, each arm 5 warmups and 3 groups of20
calls; sync at group boundaries, median groups then median the two same-name arms.
Batch1/8/32, base query lengths32/512/4096 with heterogeneous lengths+i%5 and
prefixes i*127. Original main CPU readback remains in BOTH arms; PR47663 is not
stacked. No whole-model TTFT/TPOT or graph result is claimed.

validation.png is a report generated from saved results, not a terminal screenshot.
Evidence is separate from the production PR. CPU torch2.5.1; NPU torch2.10.0+cpu,
torch-npu2.10.0.post6.
