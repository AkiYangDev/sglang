# DeepSeek-V4 NPU RoPE context allocation

Production commit: 099c2c23047f41cfa251881d7961f1e3effbd572. Base: bebee560cc2ba9a322b0ca79ee6da3c13160cf39.

Run `python context_probe.py` and `python context_integration.py` in a compatible
Ascend environment. Tested on Ascend 910C, torch 2.10.0+cpu / torch_npu 2.10.0.post6,
CANN 9.1.0. Runtime config is synthetic; no full model is loaded.

context_rope.py contains the tested production module (final commit differs only
in comments). context_freqs.py extracts the production precompute function.
context_init.py extracts the RoPE portion of the production initializer, omitting
unrelated projection weights. The initializer harness supplies its config accessor.

Two frequency families, rotary dimension 64. Compare isolated lifetimes using
memory_allocated, not memory_reserved or total device HBM:

| Table rows | Main bytes | Without retained real/imag copies, bytes |
| --- | ---: | ---: |
| 1048576 | 2684361728 | 2147488768 |
| 139648 (context 139264, draft 6) | 365497344 | 289999872 |
| 33024 (context 32768, no speculation) | 84548608 | 67638272 |

136K saves 2.15961 GiB against main. The extra saving after excluding the copies
addressed by PR47718 is 1.72992 GiB. That second column is measured by releasing
those copies after initialization, not by running a complete model with PR47718.
Generic rotary embedding tables are excluded from these measurements.

PASS: 8 length-selection cases; 12 full table pairs exact; 24 forward/inverse
position gathers exact; 4 changed-position NPU graph replays exact; 6 extracted
initializer cases; actual precompute under the NPU default device prefix exact.
Full model generation, actual compressor execution, end-to-end DSpark boundaries,
TTFT and TPOT have not been tested. No claim of zero performance regression.

validation.png is rendered from saved results, not a terminal screenshot.
Test artifacts live on this evidence branch, outside the production PR.
