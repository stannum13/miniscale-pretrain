# MiniScale Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible decoder-only Transformer pretraining harness that measures and explains 1/2/4-GPU DDP and FSDP scaling, including deterministic data and exact fault recovery.

**Architecture:** A small `miniscale` Python package owns configuration, the model, binary-token data, distributed wrapping, metrics, checkpointing, and the explicit training loop. Root CLI files remain thin. GPU experiments write one JSONL record per rank-zero run; report generation refuses to fabricate missing measurements.

**Tech Stack:** Python 3.10+, PyTorch 2.2+, PyYAML; optional Hugging Face `datasets` and `transformers` only for corpus preparation.

## Global Constraints

- Do not use Hugging Face Trainer.
- Keep global batch size constant across scaling comparisons.
- Support approximately 150M, 400M, and 1B parameter configurations.
- Do not add tensor or pipeline parallelism.
- Persist model, optimizer, scheduler, RNG, global step, and exact data position.
- Record step time, throughput, per-GPU throughput, scaling efficiency, peak memory, MFU, phase timings, loss, and gradient norm.
- Never report synthetic GPU benchmark numbers as measurements.

---

### Task 1: Configuration and batch invariants

**Files:** Create `miniscale/config.py`, `configs/{smoke,150m,400m,1b}.yaml`, `tests/test_config.py`.

**Interfaces:** Produces `load_config(path) -> TrainConfig`, `TrainConfig.validate(world_size)`, and `TrainConfig.gradient_accumulation_steps(world_size)`.

- [ ] Write tests asserting exact model-family parameter bands and `global_batch_size == micro_batch_size * world_size * gradient_accumulation_steps`.
- [ ] Run `pytest tests/test_config.py -q`; expect failure because `miniscale.config` is missing.
- [ ] Implement typed dataclasses, YAML loading, validation, and derived accumulation.
- [ ] Re-run the test; expect pass.
- [ ] Commit config and tests.

### Task 2: Decoder-only Transformer

**Files:** Create `model/transformer.py`, `model/__init__.py`, `tests/test_model.py`.

**Interfaces:** Consumes `ModelConfig`; produces `Transformer(config)`, logits shaped `[B,T,V]`, scalar shifted-token loss, and `estimate_model_flops`.

- [ ] Write tests for causal behavior, tied embeddings, output/loss shapes, and activation-checkpoint backward.
- [ ] Run `pytest tests/test_model.py -q`; expect missing-module failure.
- [ ] Implement RMSNorm, RoPE, causal SDPA attention, SwiGLU blocks, tied output weights, and per-block non-reentrant checkpointing.
- [ ] Re-run the test; expect pass.
- [ ] Commit model and tests.

### Task 3: Reproducible corpus pipeline

**Files:** Create `data/prepare.py`, `data/dataset.py`, `data/__init__.py`, `scripts/prepare_data.sh`, `tests/test_data.py`.

**Interfaces:** Produces a manifest with dataset/tokenizer revisions and SHA-256 hashes; `TokenShardDataset.sample(global_index)` deterministically maps a global sample to tokens; `rank_batch(step, rank, world_size, ...)` partitions samples without overlap.

- [ ] Write tests that build tiny shards, compare repeated samples, verify rank disjointness, and verify resume from a sample cursor.
- [ ] Run `pytest tests/test_data.py -q`; expect missing-module failure.
- [ ] Implement optional streamed FineWeb-Edu download/tokenization, atomic shard writes, hashes, manifest validation, and counter-based deterministic sampling.
- [ ] Re-run the test; expect pass.
- [ ] Commit data pipeline and tests.

### Task 4: Distributed strategies and measurements

**Files:** Create `distributed/runtime.py`, `distributed/strategies.py`, `distributed/__init__.py`, `miniscale/metrics.py`, `tests/test_distributed.py`, `tests/test_metrics.py`.

**Interfaces:** Produces `DistributedContext`, `wrap_model(model, strategy, ...)`, timed all-reduce accounting, peak-memory reset/read, MFU, and rank-zero JSONL output.

- [ ] Write CPU/Gloo tests for initialization defaults, DDP wrapping, metric formulas, and scaling-efficiency joins.
- [ ] Run those tests; expect missing-module failure.
- [ ] Implement single/DDP/FSDP selection, BF16 autocast helpers, barriers/reductions, CUDA events with wall-clock fallback, and structured run records.
- [ ] Re-run tests; expect pass.
- [ ] Commit distributed runtime and metrics.

### Task 5: Checkpoint and exact recovery

**Files:** Create `checkpoint.py`, `tests/test_checkpoint.py`.

**Interfaces:** Produces `save_checkpoint(...) -> Path` and `load_checkpoint(...) -> ResumeState`, including Python/NumPy/Torch/CUDA RNG and per-rank state.

- [ ] Write a test that advances RNG and data cursor, restores, and proves identical next values/model/optimizer/scheduler state.
- [ ] Run `pytest tests/test_checkpoint.py -q`; expect missing-module failure.
- [ ] Implement atomic rank-local state plus rank-zero metadata, barriers, config fingerprint checks, and strict topology checks.
- [ ] Re-run test; expect pass.
- [ ] Commit checkpoint recovery and tests.

### Task 6: Explicit training and benchmark CLIs

**Files:** Create `miniscale/trainer.py`, `train.py`, `bench.py`, `tests/test_trainer.py`.

**Interfaces:** `run_training(config, benchmark=False, stop_after=None)` runs accumulation, optimizer/scheduler, clipping, measurements, checkpoint/save/resume, and intentional termination; `bench.py` emits measured-window summaries only.

- [ ] Write a controlled CPU test comparing one-rank loss with an interrupted/resumed run.
- [ ] Run `pytest tests/test_trainer.py -q`; expect missing-module failure.
- [ ] Implement the explicit loop with warmup exclusion, synchronization around timings, phase measurements, token-weighted reductions, JSONL logs, and SIGTERM checkpoint handling.
- [ ] Re-run test; expect pass.
- [ ] Commit loop and CLIs.

### Task 7: Experiment orchestration and reports

**Files:** Create `Makefile`, `scripts/{single_gpu,distributed,benchmark,fault_test}.sh`, `scripts/render_report.py`, `results/README.md`, `profiles/README.md`, `figures/README.md`, `SCALE_STATE.md`, `REPORT.md`, `README.md`, `tests/test_report.py`.

**Interfaces:** Required Make targets invoke reproducible commands; report renderer reads JSONL, computes one-GPU-relative efficiency, and prints missing cells as `NOT RUN`.

- [ ] Write tests for JSONL aggregation and missing-result handling.
- [ ] Run `pytest tests/test_report.py -q`; expect failure.
- [ ] Implement launch scripts, Make targets, table renderer, experimental protocol, fault-test assertions, and honest initial state/report documentation.
- [ ] Re-run test; expect pass.
- [ ] Commit orchestration and reports.

### Task 8: End-to-end verification

**Files:** Modify only files implicated by failures.

**Interfaces:** All repository commands and requirements.

- [ ] Run `make smoke`; expect a completed CPU smoke run with checkpoint and metrics.
- [ ] Run `pytest -q`; expect all tests pass.
- [ ] Run `python bench.py --help`, `python train.py --help`, and `make report`; expect exit zero.
- [ ] Inspect the requirement matrix and confirm GPU-only acceptance items remain explicitly pending rather than claimed.
- [ ] Commit verified final state.
