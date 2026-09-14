from __future__ import annotations

import argparse

import torch

from miniscale.config import load_config
from miniscale.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a warmup + measured scaling benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", required=True, choices=("single", "ddp", "fsdp"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--peak-tflops", required=True, type=float)
    parser.add_argument("--allow-cpu", action="store_true", help="diagnostics only; not a GPU scaling result")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("benchmark requires CUDA; pass --allow-cpu only for diagnostic plumbing")
    config = load_config(args.config)
    config.strategy = args.strategy
    config.output_dir = args.output_dir
    config.hardware_peak_tflops = args.peak_tflops
    result = run_training(config, run_id=args.run_id)
    print(result.last_checkpoint.parent.parent / "benchmark.jsonl")


if __name__ == "__main__":
    main()
