#!/usr/bin/env python3
"""Binned SSA event counters for Autocrine (nc-6) and Paracrine (c-6).

The counted SSA kernels are generated from the Autocrine (nc-6) and Paracrine
(c-6) models distributed with this repository. This keeps the reaction network
identical to the plasmid-sweep simulations while adding per-event counts.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTOCRINE_NC6_SOURCE = REPOSITORY_ROOT / "plasmid_sweep" / "models" / "autocrine_nc6_model.py"
PARACRINE_C6_SOURCE = REPOSITORY_ROOT / "plasmid_sweep" / "models" / "paracrine_c6_model.py"


def _extract_events(paths: List[Path]) -> List[str]:
    pattern = re.compile(r"rxns\.append\(\([^,]+,\s*['\"]([^'\"]+)['\"]\)\)")
    events: List[str] = []
    seen = set()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for event in pattern.findall(text):
            if event not in seen:
                seen.add(event)
                events.append(event)
    return events


EVENTS = _extract_events([AUTOCRINE_NC6_SOURCE, PARACRINE_C6_SOURCE])
EVENT_INDEX = {name: idx for idx, name in enumerate(EVENTS)}


def n_bins(T_end: float, bin_size: float) -> int:
    return int(math.ceil(T_end / bin_size))


def record_event(counts: np.ndarray, t: float, T_end: float, bin_size: float, event: str) -> None:
    if event not in EVENT_INDEX:
        raise KeyError(f"Unknown event name: {event}")
    if t <= 0:
        bin_idx = 0
    else:
        bin_idx = min(int((t - 1e-12) // bin_size), counts.shape[0] - 1)
    if 0 <= bin_idx < counts.shape[0]:
        counts[bin_idx, EVENT_INDEX[event]] += 1


def _load_counted_cond4():
    source = AUTOCRINE_NC6_SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        "def gillespie_cond4(P: Params4, seed: int | None = None):",
        "def gillespie_cond4_counted(P, seed=None, bin_size=10, max_events=5000000):",
    )
    source = source.replace(
        "    max_it = int(3e6)\n",
        "    max_it = int(max_events)\n"
        "    event_count = 0\n"
        "    counts = np.zeros((n_bins(P.T_end, bin_size), len(EVENTS)), dtype=np.int64)\n",
    )
    source = source.replace(
        "        who, rx = rxns[idx]\n",
        "        who, rx = rxns[idx]\n"
        "        record_event(counts, t, P.T_end, bin_size, rx)\n"
        "        event_count += 1\n",
        1,
    )
    source = source.replace(
        "    return np.array(times), np.array(G_las_trace), np.array(G_lux_trace)\n",
        "    qc = {'last_t': t, 'event_count': event_count, 'hit_event_cap': float(event_count >= max_events)}\n"
        "    return np.array(times), np.array(G_las_trace), np.array(G_lux_trace), counts, qc\n",
    )
    module = types.ModuleType("counted_cond4")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "EVENTS": EVENTS,
        "record_event": record_event,
        "n_bins": n_bins,
    })
    exec(compile(source, str(AUTOCRINE_NC6_SOURCE), "exec"), namespace)
    return namespace["Params4"], namespace["gillespie_cond4_counted"]


def _load_counted_cond6():
    source = PARACRINE_C6_SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        "def gillespie_cond6(P: Params6, seed: int | None = None):",
        "def gillespie_cond6_counted(P, seed=None, bin_size=10, max_events=5000000):",
    )
    source = source.replace(
        "    max_it = int(3e6)\n",
        "    max_it = int(max_events)\n"
        "    event_count = 0\n"
        "    counts = np.zeros((n_bins(P.T_end, bin_size), len(EVENTS)), dtype=np.int64)\n",
    )
    source = source.replace(
        "        who, rx = rxns[idx]\n",
        "        who, rx = rxns[idx]\n"
        "        record_event(counts, t, P.T_end, bin_size, rx)\n"
        "        event_count += 1\n",
        1,
    )
    source = source.replace(
        "    return np.array(times), np.array(GA), np.array(GB)\n",
        "    qc = {'last_t': t, 'event_count': event_count, 'hit_event_cap': float(event_count >= max_events)}\n"
        "    return np.array(times), np.array(GA), np.array(GB), counts, qc\n",
    )
    module = types.ModuleType("counted_cond6")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({
        "EVENTS": EVENTS,
        "record_event": record_event,
        "n_bins": n_bins,
    })
    exec(compile(source, str(PARACRINE_C6_SOURCE), "exec"), namespace)
    return namespace["Params6"], namespace["gillespie_cond6_counted"]


Params4, gillespie_cond4_counted = _load_counted_cond4()
Params6, gillespie_cond6_counted = _load_counted_cond6()


def sample_trace(times: np.ndarray, values: np.ndarray, T_end: int) -> np.ndarray:
    t_grid = np.arange(T_end + 1)
    idx = np.searchsorted(times, t_grid, side="right") - 1
    idx[idx < 0] = 0
    return values[idx]


def stats(arr: np.ndarray) -> Dict[str, float]:
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = float(sd / mean) if mean > 0 else float("nan")
    return {"mean": mean, "sd": sd, "CV": cv, "CV2": float(cv * cv) if mean > 0 else float("nan")}


def write_reporter_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_event_bins(path: Path, condition_rows: List[Tuple[str, int, int, np.ndarray]], bin_size: int) -> None:
    fieldnames = ["condition", "copies", "replicate_id", "bin_start_s", "bin_end_s"] + EVENTS
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, copies, rep_id, counts in condition_rows:
            for bin_idx in range(counts.shape[0]):
                row = {
                    "condition": condition,
                    "copies": copies,
                    "replicate_id": rep_id,
                    "bin_start_s": bin_idx * bin_size,
                    "bin_end_s": (bin_idx + 1) * bin_size,
                }
                row.update({event: int(counts[bin_idx, EVENT_INDEX[event]]) for event in EVENTS})
                writer.writerow(row)


def write_event_summaries(
    outdir: Path,
    condition_rows: List[Tuple[str, int, int, np.ndarray]],
    bin_size: int,
    window: Tuple[int, int],
) -> None:
    by_condition: Dict[Tuple[str, int], List[np.ndarray]] = {}
    for condition, copies, _rep_id, counts in condition_rows:
        by_condition.setdefault((condition, copies), []).append(counts)

    with (outdir / "event_bins_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "copies", "bin_start_s", "bin_end_s", "event", "mean_count", "sd_count", "total_count"],
        )
        writer.writeheader()
        for (condition, copies), matrices in sorted(by_condition.items()):
            stack = np.stack(matrices, axis=0)
            for bin_idx in range(stack.shape[1]):
                for event in EVENTS:
                    vals = stack[:, bin_idx, EVENT_INDEX[event]]
                    writer.writerow({
                        "condition": condition,
                        "copies": copies,
                        "bin_start_s": bin_idx * bin_size,
                        "bin_end_s": (bin_idx + 1) * bin_size,
                        "event": event,
                        "mean_count": float(vals.mean()),
                        "sd_count": float(vals.std(ddof=0)),
                        "total_count": int(vals.sum()),
                    })

    w0, w1 = window
    first_bin = int(w0 // bin_size)
    last_bin = int(math.ceil(w1 / bin_size)) - 1
    with (outdir / "event_totals_by_replicate.csv").open("w", newline="") as handle:
        fieldnames = ["condition", "copies", "replicate_id", "period"] + EVENTS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition, copies, rep_id, counts in condition_rows:
            for period, block in [
                ("full", counts),
                ("steady_window", counts[first_bin : last_bin + 1, :]),
            ]:
                totals = block.sum(axis=0)
                row = {"condition": condition, "copies": copies, "replicate_id": rep_id, "period": period}
                row.update({event: int(totals[EVENT_INDEX[event]]) for event in EVENTS})
                writer.writerow(row)


def run_condition(
    condition: str,
    copies: int,
    N: int,
    T_end: int,
    window: Tuple[int, int],
    bin_size: int,
    RNAP_total: int,
    Ribo_total: int,
    tau_ahl: float,
    tau_s: float,
    seed: int,
    max_events: int,
) -> Tuple[List[Dict[str, object]], List[Tuple[str, int, int, np.ndarray]], List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31 - 1, size=N, dtype=np.int64)
    las_curves = np.zeros((N, T_end + 1))
    lux_curves = np.zeros((N, T_end + 1))
    event_rows = []
    qc_rows = []

    for rep_id, rep_seed in enumerate(seeds):
        if condition == "autocrine_nc6":
            params = Params4(
                RNAP_total=RNAP_total,
                Ribo_total=Ribo_total,
                n_pCon_luxI=copies,
                n_pCon_lasR=copies,
                n_pCon_luxR=copies,
                n_pLas_gfp=copies,
                n_pLux_gfp=copies,
                n_pLux_lasI=copies,
                T_end=T_end,
            )
            times, p_las_trace, p_lux_trace, counts, qc = gillespie_cond4_counted(
                params, seed=int(rep_seed), bin_size=bin_size, max_events=max_events
            )
        elif condition == "paracrine_c6":
            params = Params6(
                RNAP_total=RNAP_total,
                Ribo_total=Ribo_total,
                n_pCon_luxI_A=copies,
                n_pCon_lasR_A=copies,
                n_pLas_gfp_A=copies,
                n_pCon_luxR_B=copies,
                n_pLux_gfp_B=copies,
                n_pLux_lasI_B=copies,
                T_end=T_end,
                distance_cm=1.0,
                transfer_time_ahl=tau_ahl,
                transfer_time_s=tau_s,
            )
            times, p_las_trace, p_lux_trace, counts, qc = gillespie_cond6_counted(
                params, seed=int(rep_seed), bin_size=bin_size, max_events=max_events
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        las_curves[rep_id, :] = sample_trace(times, p_las_trace, T_end)
        lux_curves[rep_id, :] = sample_trace(times, p_lux_trace, T_end)
        event_rows.append((condition, copies, rep_id, counts))
        qc_rows.append({
            "condition": condition,
            "copies": copies,
            "replicate_id": rep_id,
            "last_t": qc["last_t"],
            "event_count": int(qc["event_count"]),
            "hit_event_cap": int(qc["hit_event_cap"]),
        })

    w0, w1 = window
    las_block = las_curves[:, w0 : w1 + 1].mean(axis=1)
    lux_block = lux_curves[:, w0 : w1 + 1].mean(axis=1)
    total_block = las_block + lux_block
    las_stats = stats(las_block)
    lux_stats = stats(lux_block)
    total_stats = stats(total_block)
    summary_row = {
        "condition": condition,
        "copies": copies,
        "mean_total": total_stats["mean"],
        "sd_total": total_stats["sd"],
        "CV_total": total_stats["CV"],
        "CV2_total": total_stats["CV2"],
        "mean_pLasGFP": las_stats["mean"],
        "sd_pLasGFP": las_stats["sd"],
        "CV_pLasGFP": las_stats["CV"],
        "CV2_pLasGFP": las_stats["CV2"],
        "mean_pLuxGFP": lux_stats["mean"],
        "sd_pLuxGFP": lux_stats["sd"],
        "CV_pLuxGFP": lux_stats["CV"],
        "CV2_pLuxGFP": lux_stats["CV2"],
        "N": N,
        "T_end": T_end,
        "window_start": w0,
        "window_end": w1,
        "bin_size_s": bin_size,
        "RNAP_total": RNAP_total,
        "Ribo_total": Ribo_total,
        "transfer_time_ahl_s": tau_ahl if condition == "paracrine_c6" else "",
        "transfer_time_s_s": tau_s if condition == "paracrine_c6" else "",
        "max_events": max_events,
    }
    return [summary_row], event_rows, qc_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run binned SSA event counters for Autocrine (nc-6) vs Paracrine (c-6)."
    )
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--T", type=int, default=3500)
    parser.add_argument("--w0", type=int, default=3000)
    parser.add_argument("--w1", type=int, default=3500)
    parser.add_argument("--bin-size", type=int, default=10)
    parser.add_argument("--copies", nargs="+", type=int, default=[5, 40])
    parser.add_argument("--RNAP", type=int, default=50)
    parser.add_argument("--Ribo", type=int, default=200)
    parser.add_argument("--tau-ahl", type=float, default=240.0)
    parser.add_argument("--tau-s", type=float, default=240.0)
    parser.add_argument("--max-events", type=int, default=5_000_000)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reporter_rows: List[Dict[str, object]] = []
    all_event_rows: List[Tuple[str, int, int, np.ndarray]] = []
    qc_rows: List[Dict[str, object]] = []

    for copy_index, copies in enumerate(args.copies):
        for condition_index, condition in enumerate(["autocrine_nc6", "paracrine_c6"]):
            condition_seed = args.seed + copy_index * 1000 + condition_index * 100
            print(f"Running {condition}, copies={copies}, N={args.N}")
            summaries, event_rows, qc = run_condition(
                condition=condition,
                copies=copies,
                N=args.N,
                T_end=args.T,
                window=(args.w0, args.w1),
                bin_size=args.bin_size,
                RNAP_total=args.RNAP,
                Ribo_total=args.Ribo,
                tau_ahl=args.tau_ahl,
                tau_s=args.tau_s,
                seed=condition_seed,
                max_events=args.max_events,
            )
            reporter_rows.extend(summaries)
            all_event_rows.extend(event_rows)
            qc_rows.extend(qc)

    write_reporter_summary(args.out / "reporter_summary.csv", reporter_rows)
    write_event_bins(args.out / "event_bins_by_replicate.csv", all_event_rows, args.bin_size)
    write_event_summaries(args.out, all_event_rows, args.bin_size, (args.w0, args.w1))
    with (args.out / "run_qc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "copies", "replicate_id", "last_t", "event_count", "hit_event_cap"])
        writer.writeheader()
        writer.writerows(qc_rows)

    print(f"Wrote outputs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
