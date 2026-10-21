from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from multicell_shared_pool.analysis import summarize_condition, summarize_reporter
from multicell_shared_pool.config import CONDITIONS, SimulationConfig, derive_seed
from multicell_shared_pool.models.autocrine_nc6 import AutocrineNC6Model
from multicell_shared_pool.models.autocrine_nc3 import AutocrineNC3Model
from multicell_shared_pool.models.paracrine_c3 import ParacrineC3Model
from multicell_shared_pool.models.paracrine_c6 import ParacrineC6Model
from multicell_shared_pool.ssa import run_trajectory


MODEL_VERSION = "multicell-shared-pool-v1"
MODEL_CLASSES = {
    "autocrine_nc3": AutocrineNC3Model,
    "paracrine_c3": ParacrineC3Model,
    "autocrine_nc6": AutocrineNC6Model,
    "paracrine_c6": ParacrineC6Model,
}


@dataclass(frozen=True)
class ReplicateTask:
    config: SimulationConfig
    replicate_id: int
    seed: int


@dataclass
class ReplicateOutput:
    replicate_id: int
    seed: int
    times: np.ndarray
    reporters: dict[str, np.ndarray]
    final_time: float
    event_count: int
    hit_event_cap: bool
    runtime_seconds: float


def create_model(config: SimulationConfig):
    return MODEL_CLASSES[config.condition](config)


def build_replicate_tasks(
    config: SimulationConfig,
    base_seed: int,
    replicate_ids: Iterable[int],
) -> list[ReplicateTask]:
    return [
        ReplicateTask(
            config=config,
            replicate_id=replicate_id,
            seed=derive_seed(base_seed, config.condition, config.copies, replicate_id),
        )
        for replicate_id in replicate_ids
    ]


def _run_replicate(task: ReplicateTask) -> ReplicateOutput:
    started = time.perf_counter()
    model = create_model(task.config)
    result = run_trajectory(model, task.config, task.seed)
    return ReplicateOutput(
        replicate_id=task.replicate_id,
        seed=task.seed,
        times=result.times,
        reporters=result.reporters,
        final_time=result.final_time,
        event_count=result.event_count,
        hit_event_cap=result.hit_event_cap,
        runtime_seconds=time.perf_counter() - started,
    )


def _metadata(config: SimulationConfig, base_seed: int, target_N: int) -> dict[str, object]:
    metadata = asdict(config)
    metadata["window"] = list(config.window)
    metadata.update(
        {
            "base_seed": base_seed,
            "target_N": target_N,
            "model_version": MODEL_VERSION,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }
    )
    return metadata


def _scientific_metadata(metadata: dict[str, object]) -> dict[str, object]:
    excluded = {"target_N", "python_version", "numpy_version", "pandas_version"}
    return {key: value for key, value in metadata.items() if key not in excluded}


def _ensure_metadata(
    outdir: Path,
    config: SimulationConfig,
    base_seed: int,
    target_N: int,
    resume: bool,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metadata_path = outdir / "metadata.json"
    requested = _metadata(config, base_seed, target_N)
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _scientific_metadata(existing) != _scientific_metadata(requested):
            raise ValueError(f"Existing metadata in {outdir} do not match this run")
        if not resume:
            raise ValueError(f"Output directory already contains a run: {outdir}")
        if existing.get("target_N") != target_N:
            existing["target_N"] = target_N
            metadata_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return
    if (outdir / "batches").exists() and any((outdir / "batches").glob("*.npz")):
        raise ValueError("Batch files exist without metadata")
    metadata_path.write_text(json.dumps(requested, indent=2), encoding="utf-8")


def _batch_path(outdir: Path, outputs: list[ReplicateOutput]) -> Path:
    replicate_ids = sorted(output.replicate_id for output in outputs)
    return outdir / "batches" / f"batch_{replicate_ids[0]:05d}_{replicate_ids[-1]:05d}.npz"


def _write_batch(outdir: Path, outputs: list[ReplicateOutput]) -> Path:
    if not outputs:
        raise ValueError("Cannot write an empty batch")
    outputs = sorted(outputs, key=lambda output: output.replicate_id)
    reporter_keys = tuple(outputs[0].reporters)
    for output in outputs[1:]:
        if tuple(output.reporters) != reporter_keys:
            raise ValueError("Reporter keys differ within a batch")
        if not np.array_equal(output.times, outputs[0].times):
            raise ValueError("Sample times differ within a batch")

    path = _batch_path(outdir, outputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, np.ndarray] = {
        "replicate_ids": np.asarray([output.replicate_id for output in outputs], dtype=np.int64),
        "seeds": np.asarray([output.seed for output in outputs], dtype=np.uint64),
        "times": outputs[0].times,
        "final_times": np.asarray([output.final_time for output in outputs], dtype=float),
        "event_counts": np.asarray([output.event_count for output in outputs], dtype=np.int64),
        "hit_event_caps": np.asarray([output.hit_event_cap for output in outputs], dtype=bool),
        "runtime_seconds": np.asarray([output.runtime_seconds for output in outputs], dtype=float),
        "reporter_keys": np.asarray(reporter_keys),
    }
    for reporter_key in reporter_keys:
        payload[f"reporter__{reporter_key}"] = np.stack(
            [output.reporters[reporter_key] for output in outputs], axis=0
        )
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary_path, path)
    return path


def _load_batches(outdir: Path) -> tuple[list[ReplicateOutput], tuple[str, ...]]:
    outputs: list[ReplicateOutput] = []
    reporter_keys: tuple[str, ...] | None = None
    seen: set[int] = set()
    for batch_path in sorted((outdir / "batches").glob("batch_*.npz")):
        with np.load(batch_path, allow_pickle=False) as batch:
            current_keys = tuple(str(value) for value in batch["reporter_keys"].tolist())
            if reporter_keys is None:
                reporter_keys = current_keys
            elif current_keys != reporter_keys:
                raise ValueError(f"Reporter schema mismatch in {batch_path}")
            replicate_ids = batch["replicate_ids"].astype(int)
            for row_index, replicate_id in enumerate(replicate_ids):
                if int(replicate_id) in seen:
                    raise ValueError(f"Duplicate replicate_id {replicate_id}")
                seen.add(int(replicate_id))
                outputs.append(
                    ReplicateOutput(
                        replicate_id=int(replicate_id),
                        seed=int(batch["seeds"][row_index]),
                        times=batch["times"].copy(),
                        reporters={
                            key: batch[f"reporter__{key}"][row_index].copy()
                            for key in current_keys
                        },
                        final_time=float(batch["final_times"][row_index]),
                        event_count=int(batch["event_counts"][row_index]),
                        hit_event_cap=bool(batch["hit_event_caps"][row_index]),
                        runtime_seconds=float(batch["runtime_seconds"][row_index]),
                    )
                )
    outputs.sort(key=lambda output: output.replicate_id)
    return outputs, reporter_keys or ()


def _within_population_row(
    values: np.ndarray,
    times: np.ndarray,
    window: tuple[int, int],
    valid: bool,
) -> dict[str, float]:
    if not valid:
        return {
            "within_population_cv2_mean": np.nan,
            "within_population_cv2_sd": np.nan,
            "mean_cell_correlation": np.nan,
        }
    summary = summarize_reporter(
        values[np.newaxis, :, :],
        times,
        window,
        np.array([True]),
    )
    return {
        "within_population_cv2_mean": float(summary["within_population_cv2_mean"]),
        "within_population_cv2_sd": float(summary["within_population_cv2_sd"]),
        "mean_cell_correlation": float(summary["mean_cell_correlation"]),
    }


def aggregate_outputs(outdir: Path, config: SimulationConfig, expected_N: int) -> None:
    outputs, reporter_keys = _load_batches(outdir)
    replicate_ids = [output.replicate_id for output in outputs]
    if replicate_ids != list(range(expected_N)):
        raise ValueError(
            f"Expected replicate IDs 0..{expected_N - 1}, found {replicate_ids}"
        )
    if not outputs:
        raise ValueError("No completed batches found")
    times = outputs[0].times
    expected_times = config.sample_times
    if not np.array_equal(times, expected_times):
        raise ValueError("Batch sample times do not match configuration")
    for output in outputs[1:]:
        if not np.array_equal(output.times, times):
            raise ValueError("Batch sample times are inconsistent")

    reporter_data = {
        key: np.stack([output.reporters[key] for output in outputs], axis=0)
        for key in reporter_keys
    }
    valid = np.asarray(
        [
            not output.hit_event_cap
            and output.final_time >= config.t_end
            and all(np.all(output.reporters[key] >= 0) for key in reporter_keys)
            for output in outputs
        ],
        dtype=bool,
    )
    model = create_model(config)
    summary_rows = summarize_condition(
        reporter_data,
        model.reporter_labels,
        times,
        config.window,
        valid,
    )
    for row in summary_rows:
        row.update(
            {
                "condition": config.condition,
                "copies": config.copies,
                "cells_per_population": config.cells_per_population,
                "N_requested": expected_N,
                "window_start": config.window[0],
                "window_end": config.window[1],
            }
        )
    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)

    time_columns = [f"t{int(value)}" for value in times]
    timelapse_rows: list[dict[str, object]] = []
    for output_index, output in enumerate(outputs):
        for reporter_key in reporter_keys:
            row: dict[str, object] = {
                "condition": config.condition,
                "copies": config.copies,
                "replicate_id": output.replicate_id,
                "reporter_key": reporter_key,
                "reporter_label": model.reporter_labels[reporter_key],
                "valid_replicate": bool(valid[output_index]),
            }
            row.update(
                dict(zip(time_columns, np.mean(reporter_data[reporter_key][output_index], axis=0)))
            )
            timelapse_rows.append(row)
    pd.DataFrame(timelapse_rows).to_csv(outdir / "population_mean_timelapse.csv", index=False)

    window_mask = (times >= config.window[0]) & (times <= config.window[1])
    cell_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []
    for output_index, output in enumerate(outputs):
        for reporter_key in reporter_keys:
            cell_values = reporter_data[reporter_key][output_index]
            for cell_index in range(cell_values.shape[0]):
                cell_rows.append(
                    {
                        "condition": config.condition,
                        "copies": config.copies,
                        "replicate_id": output.replicate_id,
                        "reporter_key": reporter_key,
                        "reporter_label": model.reporter_labels[reporter_key],
                        "cell_index": cell_index,
                        "valid_replicate": bool(valid[output_index]),
                        "steady_time_mean": (
                            float(np.mean(cell_values[cell_index, window_mask]))
                            if valid[output_index]
                            else np.nan
                        ),
                    }
                )
            within = _within_population_row(
                cell_values,
                times,
                config.window,
                bool(valid[output_index]),
            )
            within.update(
                {
                    "condition": config.condition,
                    "copies": config.copies,
                    "replicate_id": output.replicate_id,
                    "reporter_key": reporter_key,
                    "reporter_label": model.reporter_labels[reporter_key],
                    "valid_replicate": bool(valid[output_index]),
                }
            )
            within_rows.append(within)
    pd.DataFrame(cell_rows).to_csv(outdir / "steady_cell_time_means.csv", index=False)
    pd.DataFrame(within_rows).to_csv(outdir / "within_population_noise.csv", index=False)

    pd.DataFrame(
        [
            {
                "condition": config.condition,
                "copies": config.copies,
                "replicate_id": output.replicate_id,
                "seed": output.seed,
                "final_time": output.final_time,
                "event_count": output.event_count,
                "hit_event_cap": output.hit_event_cap,
                "valid_replicate": bool(valid[index]),
                "runtime_seconds": output.runtime_seconds,
            }
            for index, output in enumerate(outputs)
        ]
    ).to_csv(outdir / "run_qc.csv", index=False)

    steady_payload = {"replicate_ids": np.asarray(replicate_ids), "times": times[window_mask]}
    for reporter_key in reporter_keys:
        steady_payload[reporter_key] = reporter_data[reporter_key][:, :, window_mask]
    temporary = outdir / "steady_cell_trajectories.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **steady_payload)
    os.replace(temporary, outdir / "steady_cell_trajectories.npz")


def run_condition_copy(
    config: SimulationConfig,
    N: int,
    base_seed: int,
    outdir: Path,
    batch_size: int = 10,
    workers: int = 1,
    resume: bool = True,
) -> None:
    if N <= 0 or batch_size <= 0 or workers <= 0:
        raise ValueError("N, batch_size, and workers must be positive")
    outdir = Path(outdir)
    _ensure_metadata(outdir, config, base_seed, N, resume)
    existing, _ = _load_batches(outdir)
    completed = {output.replicate_id for output in existing}
    if any(replicate_id < 0 or replicate_id >= N for replicate_id in completed):
        raise ValueError("Existing replicate IDs fall outside the requested range")
    pending = [replicate_id for replicate_id in range(N) if replicate_id not in completed]
    if completed:
        print(f"{config.condition} copies={config.copies}: resuming with {len(completed)}/{N} complete")

    for batch_start in range(0, len(pending), batch_size):
        replicate_ids = pending[batch_start : batch_start + batch_size]
        tasks = build_replicate_tasks(config, base_seed, replicate_ids)
        outputs: list[ReplicateOutput] = []
        if workers == 1:
            for task in tasks:
                output = _run_replicate(task)
                outputs.append(output)
                print(
                    f"{config.condition} copies={config.copies} replicate "
                    f"{output.replicate_id + 1}/{N}: events={output.event_count} "
                    f"runtime={output.runtime_seconds:.1f}s",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_replicate, task): task for task in tasks}
                for future in as_completed(futures):
                    output = future.result()
                    outputs.append(output)
                    print(
                        f"{config.condition} copies={config.copies} replicate "
                        f"{output.replicate_id + 1}/{N}: events={output.event_count} "
                        f"runtime={output.runtime_seconds:.1f}s",
                        flush=True,
                    )
        _write_batch(outdir, outputs)

    aggregate_outputs(outdir, config, N)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run explicit multicell shared-pool Gillespie population simulations."
    )
    parser.add_argument("--conditions", "--condition", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--copies", nargs="+", type=int, choices=(5, 40), default=[5, 40])
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--cells", type=int, default=10)
    parser.add_argument("--T", type=int, default=3500)
    parser.add_argument("--w0", type=int, default=3000)
    parser.add_argument("--w1", type=int, default=3500)
    parser.add_argument("--RNAP", type=int, default=50)
    parser.add_argument("--Ribo", type=int, default=200)
    parser.add_argument("--tau-ahl", type=float, default=240.0)
    parser.add_argument("--tau-s", type=float, default=240.0)
    parser.add_argument("--max-events", type=int, default=100_000_000)
    parser.add_argument("--validation-interval", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.N = 2
        args.cells = 2
        args.T = 20
        args.w0 = 10
        args.w1 = 20
        args.max_events = min(args.max_events, 1_000_000)

    summary_paths: list[Path] = []
    for condition in args.conditions:
        for copies in args.copies:
            config = SimulationConfig(
                condition=condition,
                copies=copies,
                cells_per_population=args.cells,
                rnap_total=args.RNAP,
                ribo_total=args.Ribo,
                t_end=args.T,
                window=(args.w0, args.w1),
                transfer_time_ahl=args.tau_ahl,
                transfer_time_s=args.tau_s,
                volume_scale=float(args.cells),
                max_events=args.max_events,
                validation_interval=args.validation_interval,
            )
            outdir = args.out / condition / f"copies_{copies}"
            run_condition_copy(
                config=config,
                N=args.N,
                base_seed=args.seed,
                outdir=outdir,
                batch_size=args.batch_size,
                workers=args.workers,
                resume=args.resume,
            )
            summary_paths.append(outdir / "summary.csv")

    combined = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out / "all_conditions_summary.csv", index=False)
    print(f"Completed. Combined summary: {(args.out / 'all_conditions_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
