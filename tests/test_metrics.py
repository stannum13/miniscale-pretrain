import json

import pytest

from miniscale.metrics import BenchmarkRecord, mfu, scaling_efficiency, write_jsonl


def test_metric_formulas() -> None:
    assert mfu(model_flops=2_000, step_seconds=0.5, gpu_count=2, peak_tflops_per_gpu=4e-9) == pytest.approx(0.5)
    assert scaling_efficiency(single_gpu_tokens_per_second=100, distributed_tokens_per_second=360, gpu_count=4) == pytest.approx(0.9)


def test_jsonl_contains_required_metrics(tmp_path) -> None:
    record = BenchmarkRecord(
        run_id="x", model="smoke", gpu_count=1, strategy="single",
        micro_batch_size=2, global_batch_size=4, gradient_accumulation_steps=2,
        sequence_length=32, measured_steps=2, step_time_ms=10, tokens_per_second=100,
        tokens_per_second_per_gpu=100, scaling_efficiency=None, peak_memory_bytes=0,
        mfu=None, forward_ms=2, backward_ms=4, optimizer_ms=1, communication_ms=None,
        loss=3.0, gradient_norm=1.2, device_type="cuda", backend="nccl",
        hardware="fixture-gpu", torch_version="fixture", cuda_version="fixture",
        experiment_key="controlled-fixture",
    )
    path = tmp_path / "metrics.jsonl"
    write_jsonl(path, record)
    payload = json.loads(path.read_text())
    assert set(BenchmarkRecord.__dataclass_fields__) == set(payload)


def test_scaling_efficiency_rejects_invalid_gpu_count() -> None:
    with pytest.raises(ValueError):
        scaling_efficiency(100, 200, 0)
