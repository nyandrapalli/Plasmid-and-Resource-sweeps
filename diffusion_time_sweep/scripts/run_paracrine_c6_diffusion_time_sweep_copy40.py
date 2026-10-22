import sys
from pathlib import Path
import argparse
import importlib.util


def script_path() -> Path:
    path = Path(__file__).resolve()
    if sys.platform == "cygwin":
        path_text = path.as_posix()
        if len(path_text) >= 3 and path_text[1:3] == ":/":
            return Path(f"/cygdrive/{path_text[0].lower()}{path_text[2:]}")
    return path


ROOT = script_path().parents[1]
MODEL_PATH = ROOT / "models" / "paracrine_c6_model.py"

spec = importlib.util.spec_from_file_location("paracrine_c6_model", str(MODEL_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load model from {MODEL_PATH}")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
run = model.run


DIFFUSION_TIMES = [30, 60, 120, 240]
SEEDS_BY_TIME = {
    30.0: 660041,
    60.0: 660042,
    120.0: 660043,
    240.0: 660044,
}
COPIES = 40
N = 500
T_END = 3500
WINDOW = (3000, 3500)
RNAP_TOTAL = 50
RIBO_TOTAL = 200
DISTANCE_CM = 1.0
OUTPUT_ROOT = ROOT / "outputs" / "diffusion_time_sweep_copy40" / "paracrine_c6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Paracrine (c-6) plasmid-sweep model at copy number 40 while varying "
            "AHL and S transfer times."
        )
    )
    parser.add_argument("-N", type=int, default=N)
    parser.add_argument("--T", type=int, default=T_END)
    parser.add_argument("--w0", type=int, default=WINDOW[0])
    parser.add_argument("--w1", type=int, default=WINDOW[1])
    parser.add_argument("--copies", type=int, default=COPIES)
    parser.add_argument("--RNAP", type=int, default=RNAP_TOTAL)
    parser.add_argument("--Ribo", type=int, default=RIBO_TOTAL)
    parser.add_argument("--distance", type=float, default=DISTANCE_CM)
    parser.add_argument("--times", nargs="+", type=float, default=DIFFUSION_TIMES)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def format_time_label(seconds: float) -> str:
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:g}s"


def seed_for_time(seconds: float) -> int:
    try:
        return SEEDS_BY_TIME[float(seconds)]
    except KeyError as exc:
        supported = ", ".join(f"{value:g}" for value in SEEDS_BY_TIME)
        raise ValueError(f"Unsupported transfer time {seconds:g} s; choose one of: {supported} s") from exc


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for transfer_time in args.times:
        label = format_time_label(transfer_time)
        outdir = args.out / f"tau_{label}"
        print(
            "Paracrine (c-6): "
            f"copies={args.copies}, AHL transfer time={transfer_time:g} s, "
            f"S transfer time={transfer_time:g} s"
        )
        run(
            N=args.N,
            T_end=args.T,
            window=(args.w0, args.w1),
            outdir=str(outdir),
            seed=seed_for_time(transfer_time),
            distance_cm=args.distance,
            transfer_time_ahl=transfer_time,
            transfer_time_s=transfer_time,
            diff_coef_ahl=None,
            diff_coef_s=None,
            copies=args.copies,
            RNAP_total=args.RNAP,
            Ribo_total=args.Ribo,
            make_plot=False,
        )


if __name__ == "__main__":
    main()
