# Reproducibility artifact for nonnormal measurement blocks

This repository contains the code and numerical artifacts for the manuscript:

**Unitary Dilation Based Coherent Realization of Nonnormal Measurement Blocks in Quantum Weighted State Algorithms**

## Repository contents

- `scripts/generate_figures.py` regenerates the operator-level computational figures used in the manuscript: `Figure_2_Variance_Bound` and `Figure_3_Breakeven`.
- `scripts/gate_level_benchmark.py` regenerates the gate-level benchmark figures, `Figure_4_GateLevel_EndToEnd` through `Figure_8_Search_Workflow`, together with the associated tables in `data/`.
- `scripts/validate_corrected_results.py` checks the consistency of the released numerical tables and figure files.
- `data/` contains the resource summary, finite-budget curves, sensitivity and ensemble records, search records, run metadata, and validation report.
- `figures/` contains the final PDF figures used in the manuscript. `Figure_1_Overview.pdf` is the conceptual overview schematic, and `Figure_1_Overview.vsdx` is its editable Visio source.
- `requirements.txt` lists the Python dependencies used for the artifact.

## Reproducing the artifacts

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Regenerate the operator-level figures:

```bash
python scripts/generate_figures.py
```

Run the gate-level benchmark pipeline:

```bash
python scripts/gate_level_benchmark.py --stage all
```

Check the released numerical artifacts:

```bash
python scripts/validate_corrected_results.py
```

The gate-level pipeline performs Qiskit transpilation and TensorCircuit/JAX simulations and is therefore slower than the operator-level figure script. Running either generation script overwrites its corresponding files in `figures/` and `data/`.

## Reproducibility parameters

The manuscript-relevant deterministic settings are encoded in the scripts:

- Random survey for `Figure_2_Variance_Bound`: 5000 complex 2 by 2 matrices, `seed=7`.
- Compiled ensemble: 24 complex 2 by 2 matrices, spectral norm 2, `ENSEMBLE_SEED=17`.
- Sensitivity grid: workload depths `1, 2, 3, 4` and two-qubit noise values from `2.5e-4` to `5.0e-3`.
- Budgeted search: seven CNOT budgets, 96 repeated trials per budget, `SEARCH_SEED=23`.
- Transpilation target: Qiskit optimization level 3 to the native `{u, cx}` basis.
