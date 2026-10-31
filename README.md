# MLA FIA-NZ CPU backup / restore validation

The image is a report generated from the JSON results, not a terminal screenshot.

Run `python validate.py cpu` or `python validate.py npu` in this directory. The NPU run requires torch-npu. The harness extracts the actual pool methods from the frozen source files, constructs small cache fixtures, and compares every destination byte/element against an independent reshape/permute oracle. No full serving stack is imported.

CPU and NPU: old code 49/70 pass; fixed code 70/70 pass. The 70 cases cover two layouts, five modes (plain, DCP rank 0, DCP rank 1, draft worker, packed FP8), and seven lengths (0, 1, 17, 127, 128, 129, 257). Page size 128, transfer chunk 64, two layers starting at layer 8. DCP is index mapping on one device, not distributed execution. No full-model accuracy or performance result is claimed.

The production PR changes one file only. These validation artifacts are stored on a separate evidence branch.
