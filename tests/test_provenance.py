from data.dataset import ensure_synthetic_dataset
from miniscale.config import load_config
from miniscale.provenance import experiment_key


def test_compiled_and_eager_runs_have_different_experiment_keys(tmp_path) -> None:
    eager = load_config("configs/smoke.yaml")
    eager.data.directory = str(tmp_path / "tokens")
    ensure_synthetic_dataset(eager.data.directory, eager.model.vocab_size)
    compiled = load_config("configs/smoke.yaml")
    compiled.data.directory = eager.data.directory
    compiled.compile = True
    assert experiment_key(eager, "fixture-gpu") != experiment_key(compiled, "fixture-gpu")
