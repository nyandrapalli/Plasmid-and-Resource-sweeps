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
    from autocrine_nc6_model import Params4, gillespie_cond4

    outdir = Path(outdir_s) / f"RNAP{rnap}_Ribo{ribo}"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "autocrine_nc6_raw.csv"

    if not raw_path.exists():
        params = Params4(
            RNAP_total=rnap,
            Ribo_total=ribo,
            n_pCon_luxI=10,
            n_pCon_luxR=10,
            n_pCon_lasR=10,
            n_pLas_gfp=10,
            n_pLux_gfp=10,
            n_pLux_lasI=10,
            T_end=t_end,
        )
        t_grid = np.arange(t_end + 1)
        las_curves = np.zeros((n, len(t_grid)), dtype=np.float32)
        lux_curves = np.zeros((n, len(t_grid)), dtype=np.float32)
        rng = np.random.default_rng(seed)
        seeds = rng.integers(1, 2**31 - 1, size=n, dtype=np.int64)
        for i, rep_seed in enumerate(seeds):
            times, g_las, g_lux = gillespie_cond4(params, seed=int(rep_seed))
            idx = np.searchsorted(times, t_grid, side="right") - 1
            idx[idx < 0] = 0
            las_curves[i, :] = g_las[idx]
            lux_curves[i, :] = g_lux[idx]
        df = pd.DataFrame(las_curves, columns=[f"GLas_t{t}" for t in t_grid])
        df = df.join(pd.DataFrame(lux_curves, columns=[f"GLux_t{t}" for t in t_grid]))
        df["replicate_id"] = np.arange(n)
        df.to_csv(raw_path, index=False)

    las_cols = [f"GLas_t{t}" for t in range(w0, w1 + 1)]
    lux_cols = [f"GLux_t{t}" for t in range(w0, w1 + 1)]
    df = pd.read_csv(raw_path, usecols=las_cols + lux_cols)
    las_block = df[las_cols].to_numpy(dtype=float).mean(axis=1)
    lux_block = df[lux_cols].to_numpy(dtype=float).mean(axis=1)
    total_block = las_block + lux_block
    mean, sd, cv, cv2 = _stats(total_block)
    mean_las, sd_las, cv_las, cv2_las = _stats(las_block)
    mean_lux, sd_lux, cv_lux, cv2_lux = _stats(lux_block)
    return {
        "condition": "Autocrine (nc-6)",
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
        "mean_pLasGFP": mean_las,
        "sd_pLasGFP": sd_las,
        "CV_pLasGFP": cv_las,
        "CV2_pLasGFP": cv2_las,
        "mean_pLuxGFP": mean_lux,
        "sd_pLuxGFP": sd_lux,
        "CV_pLuxGFP": cv_lux,
        "CV2_pLuxGFP": cv2_lux,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--T", type=int, default=4000)
    parser.add_argument("--w0", type=int, default=3000)
    parser.add_argument("--w1", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "autocrine_nc6")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (rnap, ribo, args.N, args.T, args.w0, args.w1, 404000 + i * 1009, str(args.out))
        for i, (rnap, ribo) in enumerate((r, b) for r in RNAP_VALUES for b in RIBO_VALUES)
    ]
    summaries = []
    summary_path = args.out / "summary_autocrine_nc6.csv"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_point, job): (job[0], job[1]) for job in jobs}
        for future in as_completed(futures):
            rnap, ribo = futures[future]
            summaries.append(future.result())
            pd.DataFrame(summaries).sort_values(["RNAP", "Ribo"]).to_csv(summary_path, index=False)
            print(f"Autocrine (nc-6): RNAP={rnap}, Ribo={ribo}")


if __name__ == "__main__":
    main()
