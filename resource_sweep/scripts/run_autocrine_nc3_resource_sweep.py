from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import os
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

RNAP_VALUES = [20, 35, 50, 75, 100, 150, 200]
RIBO_VALUES = [50, 100, 200, 300, 400, 500]


def _stats(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    cv = float(sd / mean) if mean > 0 else float("nan")
    return mean, sd, cv, float(cv * cv)


def _run_point(job: tuple[int, int, int, int, int, int, int, str]) -> dict:
    rnap, ribo, n, t_end, w0, w1, seed, outdir_s = job
    from autocrine_nc3_model import Params, gillespie_cond1

    outdir = Path(outdir_s) / f"RNAP{rnap}_Ribo{ribo}"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "autocrine_nc3_raw.csv"

    if not raw_path.exists():
        params = Params(
            RNAP_total=rnap,
            Ribo_total=ribo,
            n_pLux_gfp=10,
            n_pCon_luxI=10,
            n_pCon_luxR=10,
            T_end=t_end,
        )
        t_grid = np.arange(t_end + 1)
        curves = np.zeros((n, len(t_grid)), dtype=np.float32)
        rng = np.random.default_rng(seed)
        seeds = rng.integers(1, 2**31 - 1, size=n, dtype=np.int64)
        for i, rep_seed in enumerate(seeds):
            times, gfp = gillespie_cond1(params, seed=int(rep_seed))
            idx = np.searchsorted(times, t_grid, side="right") - 1
            idx[idx < 0] = 0
            curves[i, :] = gfp[idx]
        pd.DataFrame(curves, columns=[f"t{t}" for t in t_grid]).assign(
            replicate_id=lambda d: np.arange(n)
        ).to_csv(raw_path, index=False)

    cols = [f"t{t}" for t in range(w0, w1 + 1)]
    block = pd.read_csv(raw_path, usecols=cols).to_numpy(dtype=float).mean(axis=1)
    mean, sd, cv, cv2 = _stats(block)
    return {
        "condition": "Autocrine (nc-3)",
        "RNAP": rnap,
        "Ribo": ribo,
        "copies": 10,
        "N": n,
        "T_end": t_end,
        "window_start": w0,
        "window_end": w1,
        "mean": mean,
        "sd": sd,
        "CV": cv,
        "CV2": cv2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--T", type=int, default=4000)
    parser.add_argument("--w0", type=int, default=3000)
    parser.add_argument("--w1", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "autocrine_nc3")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (rnap, ribo, args.N, args.T, args.w0, args.w1, 101000 + i * 1009, str(args.out))
        for i, (rnap, ribo) in enumerate((r, b) for r in RNAP_VALUES for b in RIBO_VALUES)
    ]
    summaries = []
    summary_path = args.out / "summary_autocrine_nc3.csv"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_point, job): (job[0], job[1]) for job in jobs}
        for future in as_completed(futures):
            rnap, ribo = futures[future]
            summaries.append(future.result())
            pd.DataFrame(summaries).sort_values(["RNAP", "Ribo"]).to_csv(summary_path, index=False)
            print(f"Autocrine (nc-3): RNAP={rnap}, Ribo={ribo}")


if __name__ == "__main__":
    main()
