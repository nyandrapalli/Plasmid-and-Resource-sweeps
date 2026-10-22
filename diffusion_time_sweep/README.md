# Signal-transfer-time sweep at plasmid copy number 40

This folder contains the Gillespie stochastic simulation algorithm models and driver scripts used to test how signal-transfer time affects reporter variability in the two paracrine circuit architectures.

## Systems

| Label | Model description |
|---|---|
| Paracrine (c-3) | LuxI sender and constitutive LuxR/pLux-eGFP receiver in separate compartments. |
| Paracrine (c-6) | Lux/Las circuit split across two compartments, with pLas-eGFP in compartment A and pLux-eGFP in compartment B. |

Signal movement is represented as a first-order exchange reaction. For a specified mean transfer time `tau`, the exchange rate is `k_diff = 1/tau`. In Paracrine (c-6), AHL and S each use the stated transfer time independently; the two values are not added to define a cumulative delay.

## Sweep settings

- Plasmid copy number: 40 copies per plasmid type.
- Transfer times: 30, 60, 120, 240, 480, 960, 1920, and 3840 s.
- Separation parameter: 1 cm.
- Replicates: 500 trajectories per system and transfer time.
- Simulation duration: 3500 s.
- Steady-state analysis window: 3000–3500 s.
- Resource pools: 50 RNA polymerases and 200 ribosomes per compartment.
- The 10 s exploratory run is not included in this package or summary table.

## Reproduction

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run all transfer-time sweeps from the repository root:

```bash
python diffusion_time_sweep/scripts/run_all_diffusion_time_sweeps_copy40.py
```

The individual short- and long-time scripts can also be run separately. Raw trajectories and generated run summaries are written under `diffusion_time_sweep/outputs/`, which is excluded from version control because of file size.

## Published data

- `data/diffusion_time_copy40_summary.csv` contains the compact statistics for both systems at all eight transfer times.
- `data/seed_manifest.csv` records the random seed used for each simulation set.

For Paracrine (c-3), the reported reporter is pLux-eGFP. For Paracrine (c-6), the compact summary reports the total reporter signal used by the model summary routine; the raw simulation output contains the compartment-specific reporter traces.
