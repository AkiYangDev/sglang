# DeepSeek-V4 NPU prefill metadata validation

Base: sgl-project/sglang 25c3fd83085e76d7ba3694900a402f964e1533bf.

Run `python validate.py cpu` or `python validate.py npu` with PyTorch (and torch-npu for NPU).
The harness extracts the unchanged method bodies of init_forward_metadata and
_build_npu_compress_metadata_prefill from the frozen before/after source. Parent
metadata creation, unrelated kernels and the outer compression dispatcher are
scaffolded. Actual tensor operations run on the selected device.

168 prefill cases compare every produced metadata field and check compressed
positions against an independent Python oracle. Covers list/int32/int64 host
lengths, device/host prefix sources, empty queries and C4/C128 boundaries, reusing
backend objects across changing batches. Four isolated non-prefill mode controls
also match. This is not a full serving-stack or graph test.

Tensor.cpu calls are counted with a delegating wrapper: operations still execute.
On NPU, prefill source-method device-to-host calls drop from168 to0; fixture and
comparison transfers are outside the counting window. No latency, throughput or
model-accuracy claim. CPU torch2.5.1; NPU torch2.10.0+cpu, torch-npu2.10.0.post6.

validation.png is a generated report of these results, not a terminal screenshot.
Artifacts live on a separate evidence branch, not in the production PR.
