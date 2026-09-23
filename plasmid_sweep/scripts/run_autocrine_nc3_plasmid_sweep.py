from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from autocrine_nc3_model import run  # noqa: E402


COPIES = [5, 10, 20, 40]
N = 500
T_END = 2000
WINDOW = (1500, 2000)
OUTPUT_ROOT = ROOT / "outputs" / "autocrine_nc3"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for copies in COPIES:
        outdir = OUTPUT_ROOT / f"copies_{copies}"
        print(f"Autocrine (nc-3): copies={copies}")
        run(N=N, T_end=T_END, window=WINDOW, outdir=str(outdir), seed=111, copies=copies, make_plot=False)


if __name__ == "__main__":
    main()
