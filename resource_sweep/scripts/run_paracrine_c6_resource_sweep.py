from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import os
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

RNAP_VALUES = [20, 35, 50, 75, 100, 150, 200]
RIBO_VALUES = [50, 100, 200, 300, 400, 500]


def _run_point(job: tuple[int, int, int, int, int, int, int, str]) -> dict:
    rnap, ribo, n, t_end, w0, w1, seed, outdir_s = job
    from paracrine_c6_model import run

    point_dir = Path(outdir_s) / f"RNAP{rnap}_Ribo{ribo}"
    summary_path = point_dir / "paracrine_c6_summary.csv"
    if not summary_path.exists():
        run(
            N=n,
            T_end=t_end,
            window=(w0, w1),
            outdir=str(point_dir),
            seed=seed,
            distance_cm=1.0,
            transfer_time_ahl=5.0,
            transfer_time_s=5.0,
            diff_coef_ahl=None,
            diff_coef_s=None,
            copies=10,
            RNAP_total=rnap,
            Ribo_total=ribo,
            make_plot=False,
        )
    row = pd.read_csv(summary_path).iloc[0].to_dict()
    row["condition"] = "Paracrine (c-6)"
    row["RNAP"] = rnap
    row["Ribo"] = ribo
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=500)
    parser.add_argument("--T", type=int, default=4000)
    parser.add_argument("--w0", type=int, default=3000)
    parser.add_argument("--w1", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "paracrine_c6")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (rnap, ribo, args.N, args.T, args.w0, args.w1, 606000 + i * 1009, str(args.out))
        for i, (rnap, ribo) in enumerate((r, b) for r in RNAP_VALUES for b in RIBO_VALUES)
    ]
    summaries = []
    summary_path = args.out / "summary_paracrine_c6.csv"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_point, job): (job[0], job[1]) for job in jobs}
        for future in as_completed(futures):
            rnap, ribo = futures[future]
            summaries.append(future.result())
            pd.DataFrame(summaries).sort_values(["RNAP", "Ribo"]).to_csv(summary_path, index=False)
            print(f"Paracrine (c-6): RNAP={rnap}, Ribo={ribo}")


if __name__ == "__main__":
    main()
