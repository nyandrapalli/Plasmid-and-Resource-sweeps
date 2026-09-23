import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "run_autocrine_nc3_plasmid_sweep.py",
    "run_paracrine_c3_plasmid_sweep.py",
    "run_autocrine_c6_plasmid_sweep.py",
    "run_paracrine_c6_plasmid_sweep.py",
]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script_dir / script)], check=True)


if __name__ == "__main__":
    main()
