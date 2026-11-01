# DeepSeek-V4 NPU prefill tiling validation

Production commit: b3528878bf340ec450eb0a21c37b8887f4489054; base: 07efafb546364e7ddee1b46f9c4ae01da417f90b.
Extracted production methods keep their complete function bodies (AST-unparsed).
The harness supplies synthetic BF16/INT8 weights and KV/index fixtures, real target
NPU attention, production RoPE and actual NPU dynamic INT8 linear apply.

Run in a compatible Ascend environment with its existing custom_ops and vendor OPP
libraries configured: `python attention_boundaries.py`, `python benchmark_bf16.py`,
`python validate_w8a8.py`, `python check_control.py`.
Dependencies: torch2.10.0+cpu,torch_npu2.10.0.post6,sgl-kernel-npu2026.9.0,
custom_ops,CANN9.1.0; Ascend 910C. No packages were replaced for these runs.

BF16:16K tokens,prefix256,C4topk512,4096-token tiles,Q rank1024,64heads,head512,
WO_A8x1024x4096,WO_B4096x8192. Warm median 38.8869->39.2791ms (+1.01%).
ABBA order,eight timed calls per phase,16samples per arm. First-use metadata cost
excluded from warm timings; allocations included in peak sample.
Incremental peak allocated 3489665024->905996800 bytes;
2.4062GiB saved. Not total device HBM or reserved-cache memory.
W8A8 tests use actual dynamic INT8 Q_B/WO_B kernels, BF16 elsewhere:4cases pass.
Attention-only SWA/C4/C128:9cases bitwise equal, including prefixes and uneven tiles.
CPU controls:18dispatch cases,cache reuse/current top-k/new batch/exception restore.

Projection outputs are not bitwise equal. BF16 final max absolute error
0.0009765625,RMS error9.746305295266211e-06,referenceRMS0.047648850828409195.
Harness tolerance:rtol=.02,atol=.002. This is not full-model accuracy evidence.
Q_A is identity in the harness; compressor/indexer are fixtures. No whole-model
load/generation,real compressor/indexer integration,TTFT,TPOT or graph/CP/TP claim.
Draft PR stays opt-in until model integration and accuracy are validated.

Earlier experiments:top-k64 fixture rejected by metadata (corrected to512);
initial custom-op call lacked vendor OPP search paths (child environment corrected).
Building metadata per tile/layer added~9% stage latency; per-batch reuse removed
most of it. Larger8192tiles and direct bmm(out=) did not offer a useful improvement.
The final extra memory saving comes from releasing Q before WO_A.

validation.png is a result image rendered from results.json, not a terminal screenshot.
Test artifacts are kept on this separate branch; production PR contains no test files.
