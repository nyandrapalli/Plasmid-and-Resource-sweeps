# SSA event-count analysis

This section contains event-resolved Gillespie stochastic simulation algorithm (SSA) drivers for comparing the reaction structure of the four circuit architectures.

## Comparisons

| Script | Architectures compared |
|---|---|
| `scripts/run_event_counter_nc3_vs_c3.py` | Autocrine (nc-3) and Paracrine (c-3) |
| `scripts/run_event_counter_nc6_vs_c6.py` | Autocrine (nc-6) and Paracrine (c-6) |

The c-3 driver is standalone. The c-6 driver instruments the corresponding models in `plasmid_sweep/models/`, preserving their reaction networks while recording the selected SSA event in each time bin.

## Default settings

- trajectories per architecture and copy number: `N = 500`
- plasmid copy numbers: `5` and `40`
- total RNA polymerase: `50`
- total ribosomes: `200`
- simulation duration: `3500 s`
- analysis window: `3000-3500 s`
- event-count bin width: `10 s`
- signal-transfer time in paracrine systems: `240 s` per signal

## Usage

From the repository root, install the requirements and run either comparison:

```bash
python -m pip install -r requirements.txt
python event_counting/scripts/run_event_counter_nc3_vs_c3.py --out event_counting/outputs/nc3_vs_c3
python event_counting/scripts/run_event_counter_nc6_vs_c6.py --out event_counting/outputs/nc6_vs_c6
```

Use `-h` to list configurable simulation parameters. A short verification run can be performed with `-N 2 --T 20 --w0 10 --w1 20 --copies 5`.

## Outputs

Each driver writes:

- `reporter_summary.csv`: reporter mean, standard deviation, coefficient of variation (CV), and squared coefficient of variation (CV²).
- `event_bins_by_replicate.csv`: reaction counts for every trajectory and time bin.
- `event_bins_summary.csv`: mean, standard deviation, and total count for each event and time bin.
- `event_totals_by_replicate.csv`: full-run and analysis-window event totals for each trajectory.
- `run_qc.csv`: final simulated time, total event count, and event-cap status.

Generated `outputs/` directories are excluded from version control because event-level tables can be large.
