# Person C job scripts

The exact sequential job queues used for Person C's compression study, kept for provenance (copies of the
scripts that ran from `runs/person_c/`). Run them from the repository root with
`powershell -ExecutionPolicy Bypass -File jobs\person_c\<script>`; each step logs to `runs\person_c\logs\`.
They assume the earlier steps documented in `results/person_c/person_c_results.md` (offline sweeps, the KD student
and the TTA evaluation) have run.

| Script | What it ran |
|---|---|
| `1_gpu_compression_queue.ps1` | After the KD student: label-only control, 50% pruning + fine-tune, FP16 export, six GPU benchmarks, per-variant post-processing checks |
| `2_pruning30_and_cpu_queue.ps1` | Declared follow-up after 50% pruning collapsed: 30% pruning + fine-tune and its benchmark; ONNX export, INT8 quantisation and CPU benchmarks |
| `3_pruned30_fp16_and_cpu_retiming.ps1` | 30% pruned model stored in FP16 and benchmarked; CPU variants re-timed in a quiet window (the first CPU timings overlapped slide rendering and are kept under `runs/person_c/compression/superseded/`) |
