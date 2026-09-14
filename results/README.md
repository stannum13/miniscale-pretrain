# Results

Each run writes `steps.jsonl`, `benchmark.jsonl`, and topology-specific checkpoints below its run ID. JSONL is append-only so a hard failure preserves completed records. `scripts/render_report.py` scans benchmark artifacts and computes efficiency only against a matching measured one-GPU baseline. Generated run data is intentionally ignored; publish selected immutable artifacts explicitly.

Correctness and fault-test attestations are written under `correctness/` and `fault-tests/`. They identify CPU versus CUDA execution and are never ingested as throughput evidence.
