from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

from data.dataset import write_token_shards


def prepare(
    output: Path,
    dataset_name: str,
    dataset_config: str,
    dataset_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    max_tokens: int,
    shard_tokens: int,
) -> Path:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install data dependencies with: pip install -e '.[data]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
    stream = load_dataset(
        dataset_name, dataset_config, split="train", streaming=True, revision=dataset_revision
    )

    def token_shards() -> Iterator[list[int]]:
        buffer: list[int] = []
        consumed = 0
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer must define eos_token_id")
        for row in stream:
            encoded = tokenizer.encode(row["text"], add_special_tokens=False) + [eos]
            take = min(len(encoded), max_tokens - consumed)
            buffer.extend(encoded[:take])
            consumed += take
            while len(buffer) >= shard_tokens:
                yield buffer[:shard_tokens]
                del buffer[:shard_tokens]
            if consumed >= max_tokens:
                break
        if buffer:
            yield buffer

    return write_token_shards(
        token_shards(),
        output,
        dataset={"name": dataset_name, "config": dataset_config, "revision": dataset_revision},
        tokenizer={"name": tokenizer_name, "revision": tokenizer_revision},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream, tokenize, and shard an immutable corpus subset")
    parser.add_argument("--output", type=Path, default=Path("data/processed/fineweb-edu"))
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-tokens", type=int, default=100_000_000)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    print(prepare(args.output, args.dataset, args.dataset_config, args.dataset_revision,
                  args.tokenizer, args.tokenizer_revision, args.max_tokens, args.shard_tokens))


if __name__ == "__main__":
    main()
