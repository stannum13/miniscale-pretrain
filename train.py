from __future__ import annotations

import argparse

from miniscale.config import load_config
from miniscale.trainer import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit miniature LLM pretraining loop")
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", choices=("single", "ddp", "fsdp"))
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after", type=int, help="checkpoint and stop after this completed step")
    parser.add_argument("--peak-tflops", type=float, help="per-GPU BF16 dense peak used for MFU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for argument, attribute in (
        (args.strategy, "strategy"), (args.resume, "resume"),
        (args.output_dir, "output_dir"), (args.max_steps, "max_steps"),
        (args.peak_tflops, "hardware_peak_tflops"),
    ):
        if argument is not None:
            setattr(config, attribute, argument)
    result = run_training(config, run_id=args.run_id, stop_after=args.stop_after)
    print(f"step={result.end_step} checkpoint={result.last_checkpoint} stopped={result.stopped_early}")


if __name__ == "__main__":
    main()
