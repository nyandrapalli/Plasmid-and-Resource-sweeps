from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from paracrine_c6_model import run  # noqa: E402


COPIES = [5, 10, 20, 40]
N = 500
T_END = 2000
WINDOW = (1500, 2000)
OUTPUT_ROOT = ROOT / "outputs" / "paracrine_c6"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for copies in COPIES:
        outdir = OUTPUT_ROOT / f"copies_{copies}"
        print(f"Paracrine (c-6): copies={copies}")
        run(
            N=N,
            T_end=T_END,
            window=WINDOW,
            outdir=str(outdir),
            seed=660 + copies,
            distance_cm=1.0,
            transfer_time_ahl=5.0,
            transfer_time_s=5.0,
            diff_coef_ahl=None,
            diff_coef_s=None,
            copies=copies,
            RNAP_total=50,
            Ribo_total=200,
            make_plot=False,
        )


if __name__ == "__main__":
    main()
