# Plasmid copy-number sweep simulations

This repository contains the stochastic simulation source files used for the plasmid copy-number sweep. The simulations use exact Gillespie stochastic simulation with finite RNAP and ribosome pools.

## Systems

| Label | Model description |
|---|---|
| Autocrine (nc-3) | Lux-only circuit in one cell: pCon-luxI, pCon-luxR, and pLux-eGFP. |
| Paracrine (c-3) | Lux sender and receiver split across two cells, with 1 cm separation and 5 s AHL transfer time. |
| Autocrine (nc-6) | Lux-Las circuit in one cell. |
| Paracrine (c-6) | Lux-Las circuit split across two cells, with 1 cm separation and 5 s transfer time for AHL and S. |

## Sweep design

- Plasmid copies per plasmid type: `5, 10, 20, 40`
- Replicates: `N = 500`
- Simulation time: `2000 s`
- Analysis window: `1500-2000 s`
- Resource setting: `RNAP_total = 50`, `Ribo_total = 200`

## Files

- `models/`: model definitions and Gillespie simulation functions.
- `scripts/`: runners for each plasmid copy-number sweep.
- `data/plasmid_sweep_summary.csv`: compact summary table from the completed simulations.
- `data/plasmid_sweep_cv2_matrix.csv`: CV2 table with copy number as rows and circuit system as columns.
- `data/plasmid_sweep_mean_matrix.csv`: mean reporter table with copy number as rows and circuit system as columns.

## Run

Install the Python dependencies, then run:

```bash
python scripts/run_all_plasmid_sweeps.py
```

Individual systems can be run with the corresponding script in `scripts/`.

Large raw timelapse files are written to `outputs/` and are excluded from version control by default.
