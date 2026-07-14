# Corrected finite-budget update

This is the minimal GitHub update for the revised manuscript, *Unitary Dilation Based Coherent Realization of Nonnormal Measurement Blocks in Quantum Weighted State Algorithms*. It replaces the affected compiled results without duplicating unchanged repository files.

For a route \(m\), the corrected finite-budget analysis uses

```text
s_m(B_CX)   = max(1, floor(B_CX / n_CX,m))
MSE_m(B_CX) = v_m / s_m(B_CX) + b_m^2
RMSE_m      = sqrt(MSE_m)
```

The noise-induced squared-bias term is not divided by the number of shots.

## Included files

- `scripts/gate_level_benchmark.py` regenerates the corrected finite-budget analyses.
- `scripts/validate_corrected_results.py` independently recomputes and checks the reported finite-budget metrics from the raw tables.
- `data/` contains the route-specific variance, squared bias, CNOT cost, budget-specific metrics, ensemble records, search records, run metadata, and the validation report.
- `figures/` contains the eight manuscript PDFs required by the revised paper: Figure 1 overview, the restored Figures 1 and 3, and corrected Figures 4--8.

The existing repository files `requirements.txt`, `scripts/generate_figures.py`, and `figures/Figure_0.vsdx` are unchanged and are intentionally not duplicated in this update. Figures 1 and 3 are included solely because they are missing from the current public repository and must be restored.

## Verification

After uploading this update to the existing repository, install the repository's pinned dependencies and run:

```powershell
python scripts/validate_corrected_results.py
```

The expected result is `"status": "passed"` with no errors. The validation script checks the raw-to-budget aggregation for the named-family, sensitivity, ensemble, and search analyses.

## Scope

Figures 4--7 are basis-transpiled simulations under the stated simulated depolarizing-noise model. They are not hardware-backend measurements.
