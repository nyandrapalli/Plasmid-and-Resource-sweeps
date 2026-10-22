# Quorum-sensing stochastic simulation sweeps

This repository contains the source code and compact result tables for the stochastic simulation sweeps used as publication source material.

The simulations use the Gillespie stochastic simulation algorithm (SSA) with varying plasmid concentrations and resource pools.

## Repository structure

| Folder | Contents |
|---|---|
| `plasmid_sweep/` | Plasmid copy-number sweeps for the four circuit architectures. |
| `resource_sweep/` | RNAP/ribosome resource sweeps for the four circuit architectures. |
| `event_counting/` | Event-resolved SSA comparisons of autocrine and paracrine architectures. |
| `multicell_population_sweep/` | Explicit 10-cell population simulations coupled through shared extracellular signal pools. |
| `diffusion_time_sweep/` | Copy-40 signal-transfer-time sweeps for Paracrine (c-3) and Paracrine (c-6). |

## Circuit labels

| Label | Model description |
|---|---|
| Autocrine (nc-3) | Lux-only circuit in one cell: pCon-luxI, pCon-luxR, and pLux-eGFP. |
| Paracrine (c-3) | Lux sender and receiver split across two cells. |
| Autocrine (nc-6) | Lux-Las circuit colocated within the same noncommunicating population. |
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

## Event-count analysis

See `event_counting/README.md` for the event-resolved comparison scripts, default parameters, and output definitions.

Run the three-plasmid and six-plasmid comparisons:

```bash
python event_counting/scripts/run_event_counter_nc3_vs_c3.py --out event_counting/outputs/nc3_vs_c3
python event_counting/scripts/run_event_counter_nc6_vs_c6.py --out event_counting/outputs/nc6_vs_c6
```

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

Run the explicit multicell population simulations:

```bash
python multicell_population_sweep/scripts/run_multicell_population_sweeps.py --workers 4
```

Run the signal-transfer-time sweeps at plasmid copy number 40:

```bash
python diffusion_time_sweep/scripts/run_all_diffusion_time_sweeps_copy40.py
```

Large raw timelapse outputs are written to `outputs/` folders and are ignored by default.
