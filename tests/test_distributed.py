import torch

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
