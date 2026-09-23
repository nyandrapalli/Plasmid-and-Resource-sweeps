# Quorum-sensing stochastic simulation sweeps

This repository contains the source code and compact result tables for the stochastic simulation sweeps used as publication source material.

The simulations use the Gillespie stochastic simulation algorithm (SSA) with varying plasmid concentrations and resource pools.

## Repository structure

| Folder | Contents |
|---|---|
| `plasmid_sweep/` | Plasmid copy-number sweeps for the four circuit architectures. |
| `resource_sweep/` | RNAP/ribosome resource sweeps for the four circuit architectures. |

## Circuit labels

| Label | Model description |
|---|---|
| Autocrine (nc-3) | Lux-only circuit in one cell: pCon-luxI, pCon-luxR, and pLux-eGFP. |
| Paracrine (c-3) | Lux sender and receiver split across two cells. |
| Autocrine (c-6) | Lux-Las circuit in one cell. |
| Paracrine (c-6) | Lux-Las circuit split across two cells. |

## Plasmid sweep

See `plasmid_sweep/README.md`.

Summary data:

- `plasmid_sweep/data/plasmid_sweep_summary.csv`
- `plasmid_sweep/data/plasmid_sweep_cv2_matrix.csv`
- `plasmid_sweep/data/plasmid_sweep_mean_matrix.csv`

## Resource sweep

See `resource_sweep/README.md`.

Summary data:

- `resource_sweep/data/resource_sweep_summary.csv`
- `resource_sweep/data/*_cv2_heatmap_matrix.csv`
- `resource_sweep/data/*_cv2_lines.csv`

## Installation

```bash
python -m pip install -r requirements.txt
```

## Reproduction

Run plasmid copy-number sweeps:

```bash
python plasmid_sweep/scripts/run_all_plasmid_sweeps.py
```

Run resource sweeps:

```bash
python resource_sweep/scripts/run_all_resource_sweeps.py
```

Large raw timelapse outputs are written to `outputs/` folders and are ignored by default.
