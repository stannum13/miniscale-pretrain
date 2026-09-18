# miniscale-pretrain

An explicit, reproducible miniature LLM pretraining system for studying how one workload behaves on 1, 2, and 4 GPUs. The goal is distributed-systems evidence, not chatbot quality. The implementation follows the compute/communication/memory framework in Hugging Face's [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/main/ultra_blog.md), uses [Picotron](https://github.com/huggingface/picotron) as a readability reference, and [Nanotron](https://github.com/huggingface/nanotron) as a production checkpoint/data reference.

## What is implemented

- A standard decoder-only Transformer: RMSNorm, RoPE, causal SDPA, optional GQA, SwiGLU, tied embeddings, and block activation checkpointing.
- Approximate 150M, 400M, and 1B configurations plus a CPU smoke model. Parameter estimates are tested.
- Stream → tokenize → immutable uint32 shards → SHA-256 manifest → O(1) deterministic shuffled sampling.
- Single GPU, DDP with asynchronous bucket all-reduce timing, and FSDP `FULL_SHARD` with transformer-block wrapping.
- Constant-global-batch derivation, gradient accumulation with `no_sync`, BF16, gradient clipping, and cosine decay.
- Per-step and measured-window throughput, per-GPU throughput, peak HBM, MFU, phase timings, DDP communication, loss, and gradient norm.
- Rank-local topology-specific checkpoints containing model, optimizer, scheduler, RNG, global step, and exact global sample cursor.
- Immutable `run.json` provenance containing the complete normalized configuration, corpus/tokenizer manifest and hashes, source commit/dirty state, topology, device, and software environment. Existing run IDs require explicit resume and matching provenance.
- A hard-crash recovery test and report generation that leaves missing experiments as `NOT RUN`.

## Setup

Python 3.10+ and PyTorch 2.2+ are required. Install the core and optional corpus tooling:

```bash
pip install -e '.[test,data]'
```

Prepare a pinned FineWeb-Edu subset. Revisions are required so a moving dataset or tokenizer cannot silently change the experiment:

```bash
export DATASET_REVISION=<immutable-hugging-face-commit>
export TOKENIZER_REVISION=<immutable-hugging-face-commit>
scripts/prepare_data.sh --max-tokens 100000000
```

The default tokenizer is `HuggingFaceTB/SmolLM2-135M`; the scale configurations match its 49,152-token vocabulary. Adjust `vocab_size` if using another tokenizer. The manifest records names, revisions, token counts, and every shard hash.

## Commands

```bash
make smoke
make correctness
make fault-test

PEAK_TFLOPS=989 MODEL=150m make single-gpu
PEAK_TFLOPS=989 MODEL=150m GPU_COUNT=2 STRATEGY=ddp make distributed
PEAK_TFLOPS=989 MODEL=400m GPU_COUNT=4 STRATEGY=fsdp make distributed

PEAK_TFLOPS=989 make benchmark
make report
```

`989` is only an example; use the dense BF16 peak for the actual GPU and power mode. `make benchmark` covers all three models at 1/2/4 GPUs and both DDP/FSDP on multi-GPU. Narrow an exploratory run with `MODELS='150m' GPU_COUNTS='1 2'`. All launchers derive accumulation from `global_batch_size / (micro_batch_size × world_size)` and reject non-integral configurations.

`make correctness` runs the same controlled four-step workload at world sizes 1, 2, and 4, verifies identical step/data cursors, checks loss agreement, and emits a JSON attestation under `results/correctness/`. It uses CUDA automatically when four GPUs are visible and otherwise exercises the same DDP logic through CPU/Gloo.

For a longer run, first use the benchmark table to choose one model/configuration, increase `max_steps`, and launch `train.py` through `torchrun`. Resume with the same topology and configuration:

```bash
torchrun --standalone --nproc-per-node=4 train.py --config configs/400m.yaml \
  --strategy fsdp --run-id long-400m
torchrun --standalone --nproc-per-node=4 train.py --config configs/400m.yaml \
  --strategy fsdp --run-id long-400m \
  --resume results/runs/long-400m/checkpoints/step-00000050
```

## Measurement cautions

The benchmark window excludes configured warmup steps. Step time uses the slowest rank. CUDA phase timings are event-based. The DDP communication value measures each asynchronous gradient all-reduce from enqueue to completion; it overlaps backward and therefore must not be added to phase times. FSDP collective attribution should be obtained with Nsight Systems or `torch.profiler`, since FSDP does not expose a comparable public communication hook. Save traces under `profiles/` and derived plots under `figures/`.

CPU runs validate plumbing only: they are ignored by the three-model GPU report table. No CUDA result is bundled or inferred by this CPU-only development environment.

## Guarded GCloud run

The repository includes a Spot G2 launcher for the full CUDA evidence run. It is intentionally fail-closed: the launch requires four available global GPUs, four regional preemptible L4 GPUs, 48 regional Spot CPUs, a clean Git HEAD published to `origin`, and immutable source/dataset/tokenizer commits. The VM is a four-L4 `g2-standard-48`, self-deletes after at most six hours, and uses a conservative $4.50/hour guard plus a $1 disk/network allowance. The worst case is therefore $28, below the authorized $30 ceiling. Actual Spot cost should be lower, but capacity is not guaranteed.

```bash
make cloud-status    # exits nonzero until all required quotas are available
make cloud-launch    # creates/updates the 7-day artifact bucket, then launches
make cloud-logs
make cloud-download
make cloud-delete    # manual early stop; the six-hour deletion remains the backstop
```

The startup job pins the repository and FineWeb-Edu/tokenizer revisions, prepares deterministic shards, runs 1/2/4-rank loss agreement, 2/4-rank CUDA fault recovery, a bounded two-GPU PyTorch profile, the full 1/2/4-GPU benchmark matrix, and an intentional crash/resume of the selected 400M FSDP run. Terminal results, profiler traces, logs, and status are uploaded even when a stage fails. Artifact objects expire after seven days.

Set `MINISCALE_PROFILE_DIR`, optionally `MINISCALE_PROFILE_WAIT` and `MINISCALE_PROFILE_ACTIVE`, to collect gzip-compressed per-rank PyTorch traces for a targeted run. Do not use profiled throughput as a scaling result; the cloud workflow deliberately profiles a separate run.

## Layout

`configs/`, `data/`, `model/`, `distributed/`, `train.py`, `checkpoint.py`, `bench.py`, `scripts/`, `results/`, `profiles/`, `figures/`, `SCALE_STATE.md`, and `REPORT.md` follow the repository contract. Tests in `tests/` cover model causality, deterministic rank sampling, metric math, full-state recovery, and loss-equivalent resume.
