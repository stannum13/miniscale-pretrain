from pathlib import Path
import json

import torch

from data.dataset import TokenShardDataset, write_token_shards


def make_dataset(tmp_path: Path) -> TokenShardDataset:
    write_token_shards(
        [list(range(41)), list(range(41, 97))],
        tmp_path,
        dataset={"name": "fixture", "revision": "abc"},
        tokenizer={"name": "fixture-tokenizer", "revision": "def"},
    )
    return TokenShardDataset(tmp_path, sequence_length=8, seed=19)


def test_manifest_hashes_and_sampling_are_reproducible(tmp_path: Path) -> None:
    first = make_dataset(tmp_path)
    second = TokenShardDataset(tmp_path, sequence_length=8, seed=19)
    torch.testing.assert_close(first.sample(11), second.sample(11))
    assert first.manifest["dataset"]["revision"] == "abc"
    assert all(len(shard["sha256"]) == 64 for shard in first.manifest["shards"])
    assert first.manifest["min_token_id"] == 0
    assert first.manifest["max_token_id"] == 96


def test_rank_batches_are_disjoint_and_resume_from_cursor(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    rank0, idx0 = dataset.rank_batch(0, 0, rank=0, world_size=2, micro_batch_size=2)
    rank1, idx1 = dataset.rank_batch(0, 0, rank=1, world_size=2, micro_batch_size=2)
    assert set(idx0).isdisjoint(idx1)
    resumed, resumed_indices = dataset.rank_batch(4, 0, rank=0, world_size=2, micro_batch_size=2)
    direct = torch.stack([dataset.sample(index) for index in resumed_indices])
    torch.testing.assert_close(resumed, direct)


def test_manifest_tampering_is_detected(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    shard = next(tmp_path.glob("*.bin"))
    shard.write_bytes(shard.read_bytes() + b"bad")
    try:
        TokenShardDataset(tmp_path, sequence_length=8, seed=19)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("corrupt shard was accepted")


def test_dataset_rejects_tokens_outside_model_vocabulary(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    try:
        dataset.validate_vocab_size(64)
    except ValueError as exc:
        assert "token id 96" in str(exc)
    else:
        raise AssertionError("out-of-vocabulary token shard was accepted")


def test_manifest_token_bounds_are_verified_against_shards(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["max_token_id"] = 1
    path.write_text(json.dumps(manifest))
    try:
        TokenShardDataset(tmp_path, sequence_length=8, seed=19)
    except ValueError as exc:
        assert "token bounds mismatch" in str(exc)
    else:
        raise AssertionError("falsified token bounds were accepted")
