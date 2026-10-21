# Multicell population Gillespie simulations

This folder contains exact Gillespie stochastic simulation algorithm (SSA) models for testing reporter-expression noise in four quorum-sensing circuit arrangements. Each experimental population contains 10 explicit cells. Co-located cells share extracellular signal pools, while signal transfer between separated populations is represented by first-order exchange with a mean transfer time of 240 s.

## Systems

| Model name | Population structure |
|---|---|
| `autocrine_nc3` | 10 cells containing pCon-luxI, pCon-luxR, and pLux-eGFP |
| `paracrine_c3` | 10 pCon-luxI producers and 10 pCon-luxR/pLux-eGFP responders |
| `autocrine_nc6` | Autocrine (nc-6): 10 cells containing all six plasmids; pLux-eGFP and pLas-mScarlet are recorded separately |
| `paracrine_c6` | Two complementary 10-cell populations; pLux-eGFP and pLas-eGFP are recorded separately |

Every cell has an independent finite pool of 50 RNA polymerases and 200 ribosomes. Extracellular receptor-binding propensities are divided by the local pool volume scale, which is 10 for a 10-cell population. This prevents cell-number increases from artificially increasing signal concentration.

## Default scientific run

- Plasmid copies per plasmid type per carrying cell: 5 and 40
- Cells per experimental population: 10
- Independent population replicates: 500 per condition and copy number
- End time: 3500 s
- Steady-state analysis window: 3000-3500 s
- Sampling interval: 1 s
- AHL transfer time in paracrine systems: 240 s
- Las-signal transfer time in `paracrine_c6`: 240 s
- RNAP per cell: 50
- Ribosomes per cell: 200

## Requirements

- Python 3.10 or newer
- NumPy
- pandas

Install missing dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Run all simulations

Open PowerShell in this folder:

```powershell
cd "C:\path\to\multicell_shared_pool_sweeps"
```

Run all four systems at copy numbers 5 and 40:

```powershell
python .\scripts\run_multicell_population_sweeps.py --workers 4
```

If Python is not on `PATH`, replace `python` with the complete path to the intended Python executable:

```powershell
& "C:\path\to\python.exe" ".\scripts\run_multicell_population_sweeps.py" --workers 4
```

The terminal prints the condition, copy number, replicate number, event count, and elapsed time after every replicate. Results are written under `outputs/` after each batch. Rerunning the same command resumes from completed batches.

## Run individual systems

Three-plasmid systems:

```powershell
python .\scripts\run_multicell_population_sweeps.py --condition autocrine_nc3 paracrine_c3 --copies 5 40 --workers 4
```

Six-plasmid systems:

```powershell
python .\scripts\run_multicell_population_sweeps.py --condition autocrine_nc6 paracrine_c6 --copies 5 40 --workers 4
```

## Smoke test

This short run checks every condition and copy number without producing scientific results:

```powershell
python .\scripts\run_multicell_population_sweeps.py --smoke --out smoke_outputs
```

## Output files

Each `outputs/<condition>/copies_<copy>/` directory contains:

- `summary.csv`: reporter-specific single-cell, population-mean, within-population, Fano-factor, and correlation statistics
- `population_mean_timelapse.csv`: population-average reporter trajectories for every replicate
- `steady_cell_time_means.csv`: steady-window mean for every reporter cell
- `within_population_noise.csv`: replicate-level cell-to-cell noise and correlation values
- `run_qc.csv`: simulation time, event count, event-cap state, validity, seed, and runtime
- `metadata.json`: complete scientific and software settings
- `steady_cell_trajectories.npz`: compressed reporter trajectories for all cells in the steady-state window
- `batches/`: atomic restart files used for safe resume

The root output directory also contains `all_conditions_summary.csv` after all requested systems finish.

## Included summary data

- `data/multicell_population_reporter_summary.csv`: reporter-specific single-cell and population-level statistics from the completed N=500 simulations.
- `data/compartment_vs_multicell_cv2.csv`: like-for-like CV² comparison with the earlier single-cell/two-compartment plasmid sweep. For the six-plasmid systems, pLux and pLas reporter outputs are summed before calculating CV².
- `data/paracrine_effect_comparison.csv`: paracrine CV² relative to the corresponding autocrine architecture in the earlier and multicell simulations.

The large generated trajectory files are not included in the repository. They are reproducibly regenerated under `outputs/` by the command above.

For exact reproducibility of the completed runs, the corrected `autocrine_nc6` identifier retains the deterministic seed identity used before the nc-6 label correction. This changes nomenclature only; reaction networks, parameters, and stochastic trajectories are unchanged.

Replicates that reach the event cap before 3500 s remain in `run_qc.csv` but are excluded from all summary statistics.

## Tests

Run the complete dependency-free test suite with:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONWARNINGS = "error"
python -m unittest discover -s tests -v
```

The tests cover parameter validation, exact SSA behavior, resource and promoter conservation, concentration scaling, signal-transfer direction, reporter assignment, analytical calculations, resumable batches, QC exclusion, and all eight condition/copy-number combinations.
