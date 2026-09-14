from pathlib import Path

import pytest

from miniscale.config import load_config


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [("150m", 120_000_000, 190_000_000), ("400m", 330_000_000, 500_000_000), ("1b", 850_000_000, 1_150_000_000)],
)
def test_model_family_is_in_documented_parameter_band(name: str, low: int, high: int) -> None:
    cfg = load_config(Path("configs") / f"{name}.yaml")
    assert low <= cfg.model.estimated_parameters() <= high


def test_global_batch_derives_integer_accumulation() -> None:
    cfg = load_config("configs/150m.yaml")
    accumulation = cfg.gradient_accumulation_steps(world_size=4)
    assert cfg.global_batch_size == cfg.micro_batch_size * 4 * accumulation


def test_invalid_global_batch_is_rejected() -> None:
    cfg = load_config("configs/smoke.yaml")
    cfg.global_batch_size = 7
    with pytest.raises(ValueError, match="divisible"):
        cfg.validate(world_size=2)


def test_smoke_batch_supports_four_way_distributed_run() -> None:
    cfg = load_config("configs/smoke.yaml")
    assert cfg.gradient_accumulation_steps(world_size=4) == 1


@pytest.mark.parametrize("name", ("150m", "400m", "1b"))
def test_scale_family_matches_default_tokenizer_vocabulary(name: str) -> None:
    assert load_config(Path("configs") / f"{name}.yaml").model.vocab_size == 49_152
