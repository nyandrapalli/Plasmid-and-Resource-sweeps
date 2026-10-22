from run_paracrine_c3_diffusion_time_sweep_copy40 import main as run_c3_short
from run_paracrine_c3_long_diffusion_time_sweep_copy40 import main as run_c3_long
from run_paracrine_c6_diffusion_time_sweep_copy40 import main as run_c6_short
from run_paracrine_c6_long_diffusion_time_sweep_copy40 import main as run_c6_long


def main() -> None:
    run_c3_short()
    run_c6_short()
    run_c3_long()
    run_c6_long()


if __name__ == "__main__":
    main()
