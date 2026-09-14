# MiniScale Pretraining Report

## Status

This report is generated only from completed CUDA `benchmark.jsonl` artifacts. CPU diagnostics are excluded. `NOT RUN` means no compatible measurement exists; it is not an estimate. No CUDA benchmark records were found; all scaling cells remain pending.

## Scaling results

| model | GPUs | strategy | batch (micro×accum×DP) | tokens/s | tokens/s/GPU | peak HBM/GPU | efficiency | MFU |
|---|---:|---|---|---:|---:|---:|---:|---:|
| scale-150m | 1 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-150m | 2 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-150m | 4 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-400m | 1 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-400m | 2 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-400m | 4 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-1b | 1 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-1b | 2 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| scale-1b | 4 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

## How to interpret the configurations

- Single GPU avoids gradient communication and establishes the denominator for scaling efficiency.
- DDP replicates model and optimizer state, overlaps bucketed gradient all-reduce with backward computation, and should win when the model has enough computation per bucket to hide communication. At fixed global batch, increasing DP reduces accumulation, so small models can expose launch latency and collective overhead.
- FSDP full-shards parameters, gradients, and optimizer state. It lowers persistent HBM per GPU but adds parameter all-gathers and gradient reduce-scatter traffic. It is expected to help only when memory pressure enables a useful microbatch/model that DDP cannot fit, or when its communication can be hidden.
- Activation checkpointing trades extra forward recomputation for lower activation memory. It should reduce HBM while increasing step time; whether throughput improves depends on whether the saved memory permits a larger, more efficient microbatch.
- BF16 reduces tensor-core compute and memory traffic relative to FP32 on supported GPUs. MFU is reported only when the operator supplies the GPU's dense BF16 peak through `PEAK_TFLOPS`.

## Measurement protocol

All comparisons hold global batch constant. Each configuration runs configured warmup steps followed by a measured window. Step duration is the slowest rank; forward, backward, and optimizer phases use CUDA events; DDP communication uses an asynchronous all-reduce completion hook. FSDP communication remains `N/A` unless collected with the supplied profiler workflow. Tokens/s counts input tokens (`global_batch × sequence_length`). Peak HBM is the maximum allocated bytes over ranks. Scaling efficiency is `throughput_N / (N × throughput_1)` for the matching model.

## Fault recovery

`make fault-test` runs a baseline, hard-exits a second job after a step that was not checkpointed, resumes the last checkpoint, then compares final model/optimizer/scheduler/RNG contents, global step, data cursor, and post-resume losses. A timestamped JSON attestation is written under `results/fault-tests/` only after all checks pass.

## Diagnosis protocol

Interpret results from the compute/communication/memory balance rather than selecting the highest raw number blindly. Inspect per-phase time and DDP communication first. If per-GPU throughput falls as GPUs increase while HBM is comfortable, communication or launch latency is the likely limiter; test larger buckets/microbatches or compilation with one small benchmark. If DDP does not fit, compare checkpointing and FSDP separately so recomputation and sharding costs are not conflated. Record every prediction and result in `SCALE_STATE.md` before changing another variable.
