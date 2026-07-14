from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
REPORT_BUDGETS = (1.0e3, 1.0e4, 1.0e5)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, atol: float = 1.0e-11) -> bool:
    return math.isclose(a, b, rel_tol=1.0e-10, abs_tol=atol)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    checks: dict[str, object] = {}

    resource = read_csv("gate_level_resource_summary.csv")
    checks["resource_rows"] = len(resource)
    if len(resource) != 6:
        errors.append(f"Expected 6 resource rows, found {len(resource)}")

    required_resource = {
        "target_real",
        "target_imag",
        "noisy_mean_real",
        "noisy_mean_imag",
        "noisy_variance",
        "noisy_bias_squared",
        "cx_per_shot",
    }
    missing_resource = required_resource - set(resource[0])
    if missing_resource:
        errors.append(f"Resource CSV is missing {sorted(missing_resource)}")

    for row in resource:
        target = complex(float(row["target_real"]), float(row["target_imag"]))
        ideal_mean = complex(
            float(row["ideal_mean_real"]), float(row["ideal_mean_imag"])
        )
        if abs(ideal_mean - target) > 1.0e-5:
            errors.append(
                f"Ideal-route mismatch for {row['family']}/{row['method']}: "
                f"|ideal-target|={abs(ideal_mean - target)}"
            )
        noisy_mean = complex(
            float(row["noisy_mean_real"]), float(row["noisy_mean_imag"])
        )
        stored_bias = float(row["noisy_bias_squared"])
        recomputed_bias = abs(noisy_mean - target) ** 2
        if not close(stored_bias, recomputed_bias):
            errors.append(
                f"Bias mismatch for {row['family']}/{row['method']}: "
                f"stored={stored_bias}, recomputed={recomputed_bias}"
            )
        for budget in REPORT_BUDGETS:
            shots = max(1, int(math.floor(budget / float(row["cx_per_shot"]))))
            mse = float(row["noisy_variance"]) / shots + stored_bias
            if mse + 1.0e-14 < stored_bias:
                errors.append(f"MSE fell below bias floor for {row['family']}/{row['method']}")

    resource_lookup = {(row["family"], row["method"]): row for row in resource}
    jordan_rankings: dict[str, float] = {}
    for budget in REPORT_BUDGETS:
        method_mse: dict[str, float] = {}
        for method in ("split", "dilation"):
            row = resource_lookup[("jordan_gamma_1", method)]
            shots = max(1, int(math.floor(budget / float(row["cx_per_shot"]))))
            method_mse[method] = (
                float(row["noisy_variance"]) / shots
                + float(row["noisy_bias_squared"])
            )
        ratio = method_mse["dilation"] / method_mse["split"]
        jordan_rankings[str(int(budget))] = ratio
        if ratio <= 1.0:
            errors.append(f"Unexpected Jordan ranking at B={budget}: ratio={ratio}")
    checks["jordan_dilation_vs_split_mse_ratio"] = jordan_rankings

    sensitivity_raw = read_csv("gate_level_sensitivity_raw.csv")
    sensitivity_metrics = read_csv("gate_level_sensitivity_budget_metrics.csv")
    checks["sensitivity_raw_rows"] = len(sensitivity_raw)
    checks["sensitivity_budget_rows"] = len(sensitivity_metrics)
    if len(sensitivity_raw) != 64:
        errors.append(f"Expected 64 sensitivity raw rows, found {len(sensitivity_raw)}")
    if len(sensitivity_metrics) != 64 * len(REPORT_BUDGETS):
        errors.append(
            f"Expected {64 * len(REPORT_BUDGETS)} sensitivity metric rows, "
            f"found {len(sensitivity_metrics)}"
        )
    if any("factor" in column for column in sensitivity_raw[0]):
        errors.append("Legacy noisy factor remains in sensitivity raw CSV")

    sensitivity_lookup = {
        (
            row["family"],
            row["workload_layers"],
            row["one_qubit_noise"],
            row["two_qubit_noise"],
        ): row
        for row in sensitivity_raw
    }
    for metric in sensitivity_metrics:
        key = (
            metric["family"],
            metric["workload_layers"],
            metric["one_qubit_noise"],
            metric["two_qubit_noise"],
        )
        raw = sensitivity_lookup[key]
        budget = float(metric["total_cnot_budget"])
        recomputed: dict[str, float] = {}
        for method in ("split", "dilation", "genexp"):
            shots = max(1, int(math.floor(budget / float(raw[f"{method}_cx_per_shot"]))))
            mse = (
                float(raw[f"{method}_noisy_variance"]) / shots
                + float(raw[f"{method}_noisy_bias_squared"])
            )
            recomputed[method] = mse
            if int(metric[f"{method}_shots"]) != shots:
                errors.append(f"Sensitivity shot mismatch for {key}, B={budget}, {method}")
            if not close(float(metric[f"{method}_mse"]), mse):
                errors.append(f"Sensitivity MSE mismatch for {key}, B={budget}, {method}")
        if not close(
            float(metric["dil_vs_split_mse_ratio"]),
            recomputed["dilation"] / recomputed["split"],
        ):
            errors.append(f"Sensitivity split ratio mismatch for {key}, B={budget}")
        if not close(
            float(metric["dil_vs_genexp_mse_ratio"]),
            recomputed["dilation"] / recomputed["genexp"],
        ):
            errors.append(f"Sensitivity genexp ratio mismatch for {key}, B={budget}")

    ensemble_raw = read_csv("operator_ensemble_raw.csv")
    ensemble_metrics = read_csv("operator_ensemble_budget_metrics.csv")
    checks["ensemble_raw_rows"] = len(ensemble_raw)
    checks["ensemble_budget_rows"] = len(ensemble_metrics)
    if len(ensemble_raw) != 768:
        errors.append(f"Expected 768 ensemble raw rows, found {len(ensemble_raw)}")
    if len(ensemble_metrics) != 768 * len(REPORT_BUDGETS):
        errors.append(
            f"Expected {768 * len(REPORT_BUDGETS)} ensemble metric rows, "
            f"found {len(ensemble_metrics)}"
        )
    if any("noisy_factor" in column for column in ensemble_raw[0]):
        errors.append("Legacy noisy factor remains in ensemble raw CSV")

    ensemble_lookup = {
        (
            row["sample_id"],
            row["workload_layers"],
            row["one_qubit_noise"],
            row["two_qubit_noise"],
        ): row
        for row in ensemble_raw
    }
    for metric in ensemble_metrics:
        key = (
            metric["sample_id"],
            metric["workload_layers"],
            metric["one_qubit_noise"],
            metric["two_qubit_noise"],
        )
        raw = ensemble_lookup[key]
        budget = float(metric["total_cnot_budget"])
        recomputed = {}
        for method in ("split", "dilation", "genexp"):
            shots = max(1, int(math.floor(budget / float(raw[f"{method}_cx_per_shot"]))))
            mse = (
                float(raw[f"{method}_noisy_variance"]) / shots
                + float(raw[f"{method}_noisy_bias_squared"])
            )
            recomputed[method] = mse
            if int(metric[f"{method}_shots"]) != shots:
                errors.append(f"Ensemble shot mismatch for {key}, B={budget}, {method}")
            if not close(float(metric[f"{method}_mse"]), mse):
                errors.append(f"Ensemble MSE mismatch for {key}, B={budget}, {method}")
        expected_ratios = {
            "dil_vs_split_mse_ratio": recomputed["dilation"] / recomputed["split"],
            "dil_vs_genexp_mse_ratio": recomputed["dilation"] / recomputed["genexp"],
            "genexp_vs_split_mse_ratio": recomputed["genexp"] / recomputed["split"],
        }
        for column, expected in expected_ratios.items():
            if not close(float(metric[column]), expected):
                errors.append(f"Ensemble ratio mismatch for {key}, B={budget}, {column}")

    sensitivity_summary = read_csv("gate_level_sensitivity_budget_summary.csv")
    for summary in sensitivity_summary:
        values = [
            float(row["dil_vs_split_mse_ratio"])
            for row in sensitivity_metrics
            if row["family"] == summary["family"]
            and row["total_cnot_budget"] == summary["total_cnot_budget"]
        ]
        favorable = sum(value < 1.0 for value in values) / len(values)
        if not close(float(summary["dil_vs_split_beneficial_fraction"]), favorable):
            errors.append(f"Sensitivity summary fraction mismatch for {summary['family']}")
        if not close(float(summary["dil_vs_split_median_mse_ratio"]), statistics.median(values)):
            errors.append(f"Sensitivity summary median mismatch for {summary['family']}")

    ensemble_summary = read_csv("operator_ensemble_budget_summary.csv")
    for summary in ensemble_summary:
        values = [
            float(row["dil_vs_split_mse_ratio"])
            for row in ensemble_metrics
            if row["total_cnot_budget"] == summary["total_cnot_budget"]
        ]
        favorable = sum(value < 1.0 for value in values) / len(values)
        if not close(float(summary["dil_vs_split_beneficial_fraction"]), favorable):
            errors.append(f"Ensemble summary fraction mismatch at B={summary['total_cnot_budget']}")
        if not close(float(summary["dil_vs_split_median_mse_ratio"]), statistics.median(values)):
            errors.append(f"Ensemble summary median mismatch at B={summary['total_cnot_budget']}")

    search_summary = read_csv("budgeted_search_summary.csv")
    search_candidates = read_csv("budgeted_search_candidate_stats.csv")
    search_trials = read_csv("budgeted_search_trials.csv")
    checks["search_summary_rows"] = len(search_summary)
    checks["search_candidate_rows"] = len(search_candidates)
    checks["search_trial_rows"] = len(search_trials)
    if len(search_summary) != 3 * 7:
        errors.append(f"Expected 21 search summary rows, found {len(search_summary)}")
    if len(search_candidates) != 41 * 3:
        errors.append(f"Expected 123 search candidate rows, found {len(search_candidates)}")
    if len(search_trials) != 3 * 7 * 96:
        errors.append(f"Expected 2016 search trial rows, found {len(search_trials)}")
    if any(int(row["shots_per_evaluation"]) < 1 for row in search_summary):
        errors.append("Search summary contains a nonpositive shot allocation")
    if any(float(row["noisy_per_shot_variance"]) < -1.0e-12 for row in search_candidates):
        errors.append("Search candidate table contains a negative variance")

    expected_figures = [
        "Figure_4_GateLevel_EndToEnd.pdf",
        "Figure_5_GateLevel_Sensitivity.pdf",
        "Figure_6_Ensemble_PhaseDiagram.pdf",
        "Figure_7_Compiled_Predictors.pdf",
        "Figure_8_Search_Workflow.pdf",
    ]
    missing_figures = [name for name in expected_figures if not (FIGURE_DIR / name).exists()]
    if missing_figures:
        errors.append(f"Missing corrected figures: {missing_figures}")

    artifact_hashes: dict[str, str] = {}
    for folder in (DATA_DIR, FIGURE_DIR):
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.name != "validation_report.json":
                artifact_hashes[str(path.relative_to(ROOT))] = sha256(path)

    report = {
        "status": "passed" if not errors else "failed",
        "checks": checks,
        "errors": errors,
        "sha256": artifact_hashes,
    }
    report_path = DATA_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
