# Scaling State

## Current status

- Phase: implementation/correctness complete; CUDA characterization pending.
- Host used for repository construction: CPU-only (`torch.cuda.is_available() == False`).
- Evidence policy: CPU smoke and fault tests validate behavior but are not scaling measurements.
- Fixed comparison invariant: global batch 32 sequences; sequence length 1,024 for scale models.
- Controlled CPU/Gloo check (`make correctness`): 1/2/4 processes completed the same four-step smoke workload at global batch 8; maximum loss deviation versus one process was 0 (2 processes) and 1.19e-7 (4 processes). The command writes a machine-readable attestation under `results/correctness/`.

## Iteration 0 — establish the measurement harness

- Bottleneck selected: unknown until a real 1-GPU baseline exists.
- Prediction: the 150M model will have the lowest 2/4-GPU scaling efficiency because its short per-step compute gives gradient collectives and launch latency less work to hide behind.
- Smallest benchmark: 150M, 1 GPU single then 2 GPU DDP, 20 warmup + 50 measured steps, global batch 32.
- Profile: collect per-phase CUDA events, DDP hook communication time, and peak allocated HBM. Escalate to Nsight only if the phase counters cannot explain the gap.
- Result: `NOT RUN` — no CUDA device is available on the construction host.
- Prediction comparison: pending.
- Decision: run `PEAK_TFLOPS=<actual> MODELS=150m GPU_COUNTS='1 2' make benchmark`, update this file, then commit before changing a tuning variable.

## Iteration 1 — make distributed evidence fail closed

- Bottleneck selected: rank-local setup and checkpoint failures could leave healthy ranks blocked in a later collective, while an overlapping generated-data exclusion could omit tracked source from experiment identity.
- Evidence: adversarial review of provenance discovery and each distributed checkpoint phase; regression tests reproduce tracked-source exclusion, rank-payload write failure, and state-restore failure.
- Predicted effect: no throughput change; failures should surface with the responsible rank and phase within the configured process-group/orchestration timeout, and every tracked source edit should change experiment identity.
- Smallest benchmark and fixed controls: focused CPU tests for provenance/checkpoint/trainer behavior, followed by the unchanged 1/2/4-process correctness workload and 1/2-process hard-crash resume workflow.
- Profile artifact: machine-readable correctness and fault-test attestations under `results/`; subprocess stdout/stderr and checkpoint hashes retained by the fault-test attestation.
- Observed result: focused suite passes (19 tests); complete acceptance rerun is the release gate.
- Prediction error/explanation: none observed in focused validation; DDP/FSDP constructor collectives can only be bounded by the process-group timeout because a peer may fail inside the collective itself.
- Next decision: do not tune throughput until real CUDA measurements identify a bottleneck.
- Commits: `c97705d`, `387e245`.

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
