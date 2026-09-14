import torch
from datetime import timedelta

from distributed.runtime import DistributedContext
from distributed.strategies import unwrap_model, wrap_model


def test_context_defaults_to_single_cpu(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    context = DistributedContext.from_environment()
    assert context.rank == context.local_rank == 0
    assert context.world_size == 1
    assert context.device.type == "cpu"


def test_single_strategy_preserves_model() -> None:
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    model = torch.nn.Linear(2, 2)
    wrapped, communication = wrap_model(model, "single", context, bf16=False)
    assert wrapped is model
    assert unwrap_model(wrapped) is model
    assert communication.elapsed_ms == 0


def test_process_group_uses_bounded_timeout(monkeypatch) -> None:
    captured = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("MINISCALE_DIST_TIMEOUT_SECONDS", "17")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: captured.update(kwargs))
    context = DistributedContext.from_environment()
    assert captured["timeout"] == timedelta(seconds=17)
    assert context.owns_process_group
