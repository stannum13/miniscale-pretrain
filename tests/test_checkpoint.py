import random
from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint, save_checkpoint
from distributed.runtime import DistributedContext
from miniscale.config import load_config


def test_checkpoint_restores_training_and_rng_state(tmp_path: Path) -> None:
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    optimizer.zero_grad()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    saved_weight = model.weight.detach().clone()
    path = save_checkpoint(tmp_path, model, optimizer, scheduler, context,
                           step=7, sample_cursor=28, config=load_config("configs/smoke.yaml"))
    expected = (random.random(), np.random.random(), torch.rand(2))
    with torch.no_grad():
        model.weight.zero_()
    random.random(), np.random.random(), torch.rand(10)
    state = load_checkpoint(path, model, optimizer, scheduler, context,
                            config=load_config("configs/smoke.yaml"))
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert state.step == 7 and state.sample_cursor == 28
    torch.testing.assert_close(model.weight, saved_weight)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
    assert (path / "COMPLETE").exists()
