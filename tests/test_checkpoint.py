import random
from pathlib import Path

import numpy as np
import pytest
import torch

import checkpoint
from checkpoint import load_checkpoint, save_checkpoint
from data.dataset import ensure_synthetic_dataset
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
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    path = save_checkpoint(tmp_path, model, optimizer, scheduler, context,
                           step=7, sample_cursor=28, config=cfg)
    expected = (random.random(), np.random.random(), torch.rand(2))
    with torch.no_grad():
        model.weight.zero_()
    random.random(), np.random.random(), torch.rand(10)
    state = load_checkpoint(path, model, optimizer, scheduler, context, config=cfg)
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert state.step == 7 and state.sample_cursor == 28
    torch.testing.assert_close(model.weight, saved_weight)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])
    assert (path / "COMPLETE").exists()


def test_checkpoint_rejects_changed_dataset_manifest(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    manifest = ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    path = save_checkpoint(tmp_path / "checkpoints", model, optimizer, scheduler, context,
                           step=1, sample_cursor=8, config=cfg)
    manifest.write_text(manifest.read_text().replace("generator-v1", "generator-v2", 1))
    with pytest.raises(ValueError, match="dataset manifest"):
        load_checkpoint(path, model, optimizer, scheduler, context, config=cfg)


def test_checkpoint_rejects_foreign_run_lineage(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    path = save_checkpoint(tmp_path / "checkpoints", model, optimizer, scheduler, context,
                           step=1, sample_cursor=8, config=cfg, run_uuid="run-a")
    with pytest.raises(ValueError, match="lineage"):
        load_checkpoint(path, model, optimizer, scheduler, context,
                        config=cfg, run_uuid="run-b")


def test_checkpoint_save_labels_rank_local_serialization_failure(tmp_path: Path, monkeypatch) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="checkpoint rank payload failed on rank 0: disk full"):
        save_checkpoint(tmp_path / "checkpoints", model, optimizer, None, context,
                        step=1, sample_cursor=8, config=cfg)


def test_checkpoint_load_labels_rank_local_state_application_failure(tmp_path: Path, monkeypatch) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    cfg = load_config("configs/smoke.yaml")
    cfg.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(cfg.data.directory, cfg.model.vocab_size)
    path = save_checkpoint(tmp_path / "checkpoints", model, optimizer, None, context,
                           step=1, sample_cursor=8, config=cfg)
    monkeypatch.setattr(
        checkpoint, "_load_states",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad state")),
    )
    with pytest.raises(RuntimeError, match="checkpoint state restore failed on rank 0: bad state"):
        load_checkpoint(path, model, optimizer, None, context, config=cfg)
