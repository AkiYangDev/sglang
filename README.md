# DeepSeek-V4 verify sequence-length reuse

Base: ee83f5a2e4fa6a0251fea32ccba075ac28515e2b.

Run `python validate_fix.py cpu` with PyTorch. The script extracts the actual
context builder and sequence metadata updater from before.py and after.py.
432 cases cover verify/decode/draft, EAGLE/DSpark/explicit live lengths,
C4/C128/both/neither, empty batches, padding and idle replay. Each arm is
checked against an independent expected-length calculation. CPU conversion
call counts are not measurements of physical H2D transfers.

The optimized active-verify branch reuses the int32 device tensor already
created by the context builder. Other branches retain their existing copy.
36 nonempty cases remove one conversion call each; 18 empty cases also
remove a call. No numerical difference was found in these fixture tests.

No NPU run was performed because available devices had resident workloads.
No captured-graph replay, full model, TTFT/TPOT or timing validation is claimed.
The script also accepts `npu` for a future device run with torch_npu.

validation.png is a report generated from saved results, not a terminal screenshot.
This evidence branch is separate from the production PR.
