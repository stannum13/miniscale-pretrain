from scripts.render_report import build_rows, render_table


def record(model: str, gpus: int, throughput: float) -> dict:
    return {
        "model": model, "gpu_count": gpus, "strategy": "single" if gpus == 1 else "ddp",
        "micro_batch_size": 2, "global_batch_size": 32,
        "gradient_accumulation_steps": 16 // gpus, "sequence_length": 1024,
        "tokens_per_second": throughput, "tokens_per_second_per_gpu": throughput / gpus,
        "peak_memory_bytes": 1024, "mfu": 0.2,
        "device_type": "cuda", "backend": "nccl", "hardware": "fixture-gpu",
        "torch_version": "fixture", "cuda_version": "fixture", "experiment_key": "same",
    }


def test_report_computes_efficiency_from_matching_single_gpu_baseline() -> None:
    rows = build_rows([record("scale-150m", 1, 100), record("scale-150m", 4, 360)])
    assert rows[1]["scaling_efficiency"] == 0.9


def test_report_marks_missing_experiments_without_inventing_values() -> None:
    table = render_table(build_rows([]), models=("scale-150m",), gpu_counts=(1, 2, 4))
    assert sum("| NOT RUN |" in line for line in table.splitlines()) == 5
    assert "scale-150m" in table
    assert "| scale-150m | 2 | ddp (NOT RUN) |" in table
    assert "| scale-150m | 2 | fsdp (NOT RUN) |" in table


def test_report_excludes_cpu_diagnostics_and_incompatible_baselines() -> None:
    cpu = record("scale-150m", 1, 100)
    cpu["device_type"] = "cpu"
    distributed = record("scale-150m", 2, 190)
    distributed["experiment_key"] = "different-hardware"
    rows = build_rows([cpu, record("scale-150m", 1, 100), distributed])
    assert len(rows) == 2
    assert rows[-1]["scaling_efficiency"] is None
