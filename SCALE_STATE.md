# Scaling State

## Current status

- Phase: implementation/correctness complete; CUDA characterization pending.
- Host used for repository construction: CPU-only (`torch.cuda.is_available() == False`).
- Evidence policy: CPU smoke and fault tests validate behavior but are not scaling measurements.
- Fixed comparison invariant: global batch 32 sequences; sequence length 1,024 for scale models.

## Iteration 0 — establish the measurement harness

- Bottleneck selected: unknown until a real 1-GPU baseline exists.
- Prediction: the 150M model will have the lowest 2/4-GPU scaling efficiency because its short per-step compute gives gradient collectives and launch latency less work to hide behind.
- Smallest benchmark: 150M, 1 GPU single then 2 GPU DDP, 20 warmup + 50 measured steps, global batch 32.
- Profile: collect per-phase CUDA events, DDP hook communication time, and peak allocated HBM. Escalate to Nsight only if the phase counters cannot explain the gap.
- Result: `NOT RUN` — no CUDA device is available on the construction host.
- Prediction comparison: pending.
- Decision: run `PEAK_TFLOPS=<actual> MODELS=150m GPU_COUNTS='1 2' make benchmark`, update this file, then commit before changing a tuning variable.

## Iteration template

Copy this block for every change; vary one primary factor at a time.

- Bottleneck selected:
- Evidence:
- Predicted effect (direction and approximate magnitude):
- Smallest benchmark and fixed controls:
- Profile artifact:
- Observed result:
- Prediction error/explanation:
- Next decision:
- Commit:
