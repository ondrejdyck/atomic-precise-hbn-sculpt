# atomic-precise-hbn-sculpt

Reproducible code + inputs to regenerate simulation figures for the accompanying manuscript draft (link when published).

This repository is organized **by figure**, with each figure folder containing:
- `inputs/`: required input files (structures/data)
- `outputs/`: generated outputs (ignored by git; directory kept via `.gitkeep`)
- notebooks/scripts used to generate outputs

## Repository layout

```text
figures/
  figure2/
    inputs/
    outputs/
    Multislice_simulations_h-BN.ipynb
    twisted_h-BN_simulation.ipynb
    stacking_illustration.py

  figure6_7/
    outputs/
    Atomistic Simulation.ipynb

  (Supplementary figures derived from multislice simulations are written into
  `figures/figure2/outputs/`.)
```

## Environment (uv)

This project uses a `uv`-managed environment for reproducibility.

Clone the repository:

```bash
git clone https://github.com/ondrejdyck/atomic-precise-hbn-sculpt.git
cd atomic-precise-hbn-sculpt
```

Create/sync the environment:

```bash
uv sync
```

Run commands inside the environment with `uv run ...`.

## Reproducing outputs

### Figure 2: stacking schematic (script)

Generates:
- `figures/figure2/outputs/hBN_AA_prime_stacking.svg`
- `figures/figure2/outputs/hBN_AA_stacking.svg`

```bash
uv run python figures/figure2/stacking_illustration.py --no-show
```

### Figure 2: multislice STEM simulations (notebooks)

Open the notebooks:

```bash
uv run jupyter lab
```

Then run:
- `figures/figure2/Multislice_simulations_h-BN.ipynb`
- `figures/figure2/twisted_h-BN_simulation.ipynb`

Notes:
- These simulations can be computationally intensive.
- The notebooks are written to save figures into `figures/figure2/outputs/`.
- This includes additional figures used in the manuscript Supporting Information.

### Figures 6/7: milling model (notebook)

Open and run:
- `figures/figure6_7/Atomistic Simulation.ipynb`

Outputs are written to `figures/figure6_7/outputs/`.

## Versioning / generated files

- Generated files under `figures/**/outputs/` are ignored by git.
- The output directories are tracked via `.gitkeep` so paths exist by default.

## Citation

If you use this repository, please cite the associated manuscript (details TBD).
