# Resource sweep simulations

This repository contains the stochastic simulation source files used for the RNAP/ribosome resource sweep. The simulations use exact Gillespie stochastic simulation with finite RNAP and ribosome pools.

## Systems

| Label | Model description |
|---|---|
| Autocrine (nc-3) | Lux-only circuit in one cell: pCon-luxI, pCon-luxR, and pLux-eGFP. |
| Paracrine (c-3) | Lux sender and receiver split across two cells, with 1 cm separation and 5 s AHL transfer time. |
| Autocrine (c-6) | Lux-Las circuit in one cell. |
| Paracrine (c-6) | Lux-Las circuit split across two cells, with 1 cm separation and 5 s transfer time for AHL and S. |

## Sweep design

- RNAP totals: `20, 35, 50, 75, 100, 150, 200`
- Ribosome totals: `50, 100, 200, 300, 400, 500`
- Plasmid copies per plasmid type: `10`
- Replicates per resource point: `N = 500`
- Simulation time: `4000 s`
- Analysis window: `3000-4000 s`

## Files

- `models/`: model definitions and Gillespie simulation functions.
- `scripts/`: runners for each resource sweep.
- `data/resource_sweep_summary.csv`: compact summary table from the completed simulations.
- `data/*_cv2_heatmap_matrix.csv`: heatmap-ready CV2 matrices for each system.
- `data/*_cv2_lines.csv`: line-plot source tables for each system.

## Run

Install the Python dependencies, then run:

```bash
python scripts/run_all_resource_sweeps.py
```

Individual systems can be run with the corresponding script in `scripts/`.

Large raw timelapse files are written to `outputs/` and are excluded from version control by default.
