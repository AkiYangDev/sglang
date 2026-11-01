# DeepSeek V4 NPU RoPE cache memory

Base: c108a656f138a17ba3601a311a0d8d6fb8896eea. Production commit: 9380310d.
Run `python probe.py cpu` or `python probe.py npu` in this directory.
Actual before/after modules are included. CPU and NPU each pass 144 exact
comparisons covering FP32/BF16/FP16, dimensions 8/64, owner/no-owner,
inverse, 4D, empty/repeated/changed positions. Cold no-build rejection,
warm table pointer stability and nonpersistent buffers are checked.
Two NPU graph replays with changed positions match eager results exactly.

NPU: torch 2.10.0+cpu, torch_npu 2.10.0.post6.
Two frequency tables, 262144 positions, rotary dimension 64, each holding
FP32 and BF16 final caches. Allocated bytes above baseline:
- Before: 671095808
- After: 536876032
- Saved: 134219776 (128 MiB payload plus 2048 bytes allocator padding)

The probe releases each implementation before measuring the other. These are
active allocator bytes, not reserved memory or total device HBM. For two
1048576-position tables the removed payload is 512 MiB by calculation;
that is not a whole-model measurement. No serving accuracy or speedup claim.
Building a new dtype regenerates the temporary copies at initialization;
the warmed forward path is unchanged.

The first NPU run rejected the additional 2048-byte allocator padding under
an exact-payload assertion. The assertion now accepts bounded padding and
reports the actual delta. Production code did not change between runs.

validation.png is generated from saved results, not a terminal screenshot.
This evidence branch is separate from the production PR.

Tested after.py SHA256: 55f46e0c02df9b674a655062fabf94620002ba5f9544c6ce8c7b7fb1ad624498
