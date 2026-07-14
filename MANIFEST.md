# Minimal corrected release manifest

## Scripts

- `scripts/gate_level_benchmark.py`
- `scripts/validate_corrected_results.py`

## Validation data

- `data/gate_level_resource_summary.csv`
- `data/gate_level_budget_curves.csv`
- `data/gate_level_sensitivity_raw.csv`
- `data/gate_level_sensitivity_budget_metrics.csv`
- `data/gate_level_sensitivity_budget_summary.csv`
- `data/operator_ensemble_raw.csv`
- `data/operator_ensemble_budget_metrics.csv`
- `data/operator_ensemble_budget_summary.csv`
- `data/budgeted_search_candidate_stats.csv`
- `data/budgeted_search_trials.csv`
- `data/budgeted_search_summary.csv`
- `data/run_metadata_*.json`
- `data/validation_report.json`

## Manuscript figures updated by this release

- `figures/Figure_0_Overview.pdf`
- `figures/Figure_1_Variance_Bound.pdf`
- `figures/Figure_3_Breakeven.pdf`
- `figures/Figure_4_GateLevel_EndToEnd.pdf`
- `figures/Figure_5_GateLevel_Sensitivity.pdf`
- `figures/Figure_6_Ensemble_PhaseDiagram.pdf`
- `figures/Figure_7_Compiled_Predictors.pdf`
- `figures/Figure_8_Search_Workflow.pdf`

## Intentionally omitted unchanged files

- `requirements.txt`
- `scripts/generate_figures.py`
- `figures/Figure_0.vsdx`

This minimal release excludes PNG, SVG, and non-manuscript budget-variant figures.
