from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from miniscale.metrics import scaling_efficiency


def load_records(root: str | Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(Path(root).glob("**/benchmark.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def build_rows(records: Iterable[dict]) -> list[dict]:
    rows = [dict(record) for record in records if record.get("device_type") == "cuda"]
    baseline_groups: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["gpu_count"] == 1 and row["strategy"] == "single":
            baseline_groups.setdefault((row["model"], row["experiment_key"]), []).append(row["tokens_per_second"])
    baselines = {key: sum(values) / len(values) for key, values in baseline_groups.items()}
    for row in rows:
        baseline = baselines.get((row["model"], row["experiment_key"]))
        row["scaling_efficiency"] = (
            scaling_efficiency(baseline, row["tokens_per_second"], row["gpu_count"])
            if baseline is not None else None
        )
    return sorted(rows, key=lambda row: (row["model"], row["gpu_count"], row["strategy"]))


def _number(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_table(
    rows: list[dict],
    *,
    models: tuple[str, ...] = ("scale-150m", "scale-400m", "scale-1b"),
    gpu_counts: tuple[int, ...] = (1, 2, 4),
) -> str:
    header = "| model | GPUs | strategy | batch (micro×accum×DP) | tokens/s | tokens/s/GPU | peak HBM/GPU | efficiency | MFU |\n|---|---:|---|---|---:|---:|---:|---:|---:|"
    lines = [header]
    for model in models:
        for gpu_count in gpu_counts:
            matches = [row for row in rows if row["model"] == model and row["gpu_count"] == gpu_count]
            if not matches:
                lines.append(f"| {model} | {gpu_count} | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |")
                continue
            for row in matches:
                batch = f'{row["micro_batch_size"]}×{row["gradient_accumulation_steps"]}×{gpu_count}={row["global_batch_size"]}'
                memory = row["peak_memory_bytes"] / 2**30
                lines.append(
                    f'| {model} | {gpu_count} | {row["strategy"]} | {batch} | '
                    f'{row["tokens_per_second"]:.1f} | {row["tokens_per_second_per_gpu"]:.1f} | '
                    f'{memory:.2f} GiB | {_number(row["scaling_efficiency"])} | {_number(row.get("mfu"))} |'
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render measured scaling results without filling missing cells")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("REPORT.md"))
    args = parser.parse_args()
    rows = build_rows(load_records(args.results))
    table = render_table(rows)
    status = (
        "CUDA benchmark records were found and are shown below."
        if rows
        else "No CUDA benchmark records were found; all scaling cells remain pending."
    )
    report = f"""# MiniScale Pretraining Report

## Status

This report is generated only from completed CUDA `benchmark.jsonl` artifacts. CPU diagnostics are excluded. `NOT RUN` means no compatible measurement exists; it is not an estimate. {status}

## Scaling results

{table}

## How to interpret the configurations

- Single GPU avoids gradient communication and establishes the denominator for scaling efficiency.
- DDP replicates model and optimizer state, overlaps bucketed gradient all-reduce with backward computation, and should win when the model has enough computation per bucket to hide communication. At fixed global batch, increasing DP reduces accumulation, so small models can expose launch latency and collective overhead.
- FSDP full-shards parameters, gradients, and optimizer state. It lowers persistent HBM per GPU but adds parameter all-gathers and gradient reduce-scatter traffic. It is expected to help only when memory pressure enables a useful microbatch/model that DDP cannot fit, or when its communication can be hidden.
- Activation checkpointing trades extra forward recomputation for lower activation memory. It should reduce HBM while increasing step time; whether throughput improves depends on whether the saved memory permits a larger, more efficient microbatch.
- BF16 reduces tensor-core compute and memory traffic relative to FP32 on supported GPUs. MFU is reported only when the operator supplies the GPU's dense BF16 peak through `PEAK_TFLOPS`.

## Measurement protocol

All comparisons hold global batch constant. Each configuration runs configured warmup steps followed by a measured window. Step duration is the slowest rank; forward, backward, and optimizer phases use CUDA events; DDP communication uses an asynchronous all-reduce completion hook. FSDP communication remains `N/A` unless collected with the supplied profiler workflow. Tokens/s counts input tokens (`global_batch × sequence_length`). Peak HBM is the maximum allocated bytes over ranks. Scaling efficiency is `throughput_N / (N × throughput_1)` for the matching model.

## Fault recovery

`make fault-test` runs a baseline, hard-exits a second job after a step that was not checkpointed, resumes the last checkpoint, then compares final model/optimizer/scheduler/RNG contents, global step, data cursor, and post-resume losses. A timestamped JSON attestation is written under `results/fault-tests/` only after all checks pass.

## Diagnosis protocol

Interpret results from the compute/communication/memory balance rather than selecting the highest raw number blindly. Inspect per-phase time and DDP communication first. If per-GPU throughput falls as GPUs increase while HBM is comfortable, communication or launch latency is the likely limiter; test larger buckets/microbatches or compilation with one small benchmark. If DDP does not fit, compare checkpointing and FSDP separately so recomputation and sharding costs are not conflated. Record every prediction and result in `SCALE_STATE.md` before changing another variable.
"""
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
