# Repository manifest

This file lists the contents of the public reproducibility artifact.

## Scripts

- `scripts/generate_figures.py` - regenerates `Figure_1_Variance_Bound` and `Figure_3_Breakeven`.
- `scripts/gate_level_benchmark.py` - regenerates `Figure_4_GateLevel_EndToEnd`, `Figure_5_GateLevel_Sensitivity`, `Figure_6_Ensemble_PhaseDiagram`, `Figure_7_Compiled_Predictors`, `Figure_8_Search_Workflow`, and the CSV tables in `data/`.
- `requirements.txt` - pinned Python package versions.

## Data

- `data/gate_level_resource_summary.csv` - 6 rows.
- `data/gate_level_sensitivity_summary.csv` - 64 rows.
- `data/operator_ensemble_summary.csv` - 768 rows.
- `data/budgeted_search_summary.csv` - 21 rows.

## Figures

- `figures/Figure_0_Overview.pdf` - conceptual overview schematic.
- `figures/Figure_1_Variance_Bound.pdf`
- `figures/Figure_3_Breakeven.pdf`
- `figures/Figure_4_GateLevel_EndToEnd.pdf`
- `figures/Figure_5_GateLevel_Sensitivity.pdf`
- `figures/Figure_6_Ensemble_PhaseDiagram.pdf`
- `figures/Figure_7_Compiled_Predictors.pdf`
- `figures/Figure_8_Search_Workflow.pdf`

## Other files

- `README.md` - repository overview and reproduction instructions.
- `.gitignore` - ignores local cache/build artifacts.
- `checksums_sha256.txt` - SHA256 checksums for all tracked release files.