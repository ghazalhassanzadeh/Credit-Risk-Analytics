"""Create the compact model summaries included in the public repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_TABLES = ROOT / "reports/tables/modeling"

ABLATION_COMPARISONS = {
    "FY2010+": {
        "add_state": ("E05", "E06"),
        "add_processing": ("E05", "E07"),
        "add_processing_when_state_present": ("E06", "E08"),
    },
    "FY2001+": {
        "add_state": ("E13", "E14"),
        "add_processing": ("E13", "E15"),
        "add_processing_when_state_present": ("E14", "E16"),
    },
}


def summarize_validation_results(source: pd.DataFrame) -> pd.DataFrame:
    """Average each candidate's metrics across the two validation folds."""
    return source.groupby(
        ["experiment_id", "design", "family", "feature_variant"],
        as_index=False,
    ).agg(
        pr_auc=("pr_auc", "mean"),
        roc_auc=("roc_auc", "mean"),
        brier=("brier", "mean"),
        calibration_intercept=("calibration_intercept", "mean"),
        calibration_slope=("calibration_slope", "mean"),
        mean_predicted_risk=("mean_predicted_risk", "mean"),
        prevalence=("prevalence", "mean"),
    )


def summarize_ablation_results(source: pd.DataFrame) -> pd.DataFrame:
    """Measure the locked-test change from adding state or processing method."""
    candidate_results = source[
        source.experiment_id.str.fullmatch(r"E\d+")
    ].set_index("experiment_id")
    rows = []

    for design, comparisons in ABLATION_COMPARISONS.items():
        for comparison, (base_id, comparison_id) in comparisons.items():
            base = candidate_results.loc[base_id]
            candidate = candidate_results.loc[comparison_id]
            rows.append(
                {
                    "design": design,
                    "comparison": comparison,
                    "from_experiment": base_id,
                    "to_experiment": comparison_id,
                    "pr_auc_change": candidate.pr_auc - base.pr_auc,
                    "roc_auc_change": candidate.roc_auc - base.roc_auc,
                    "brier_change": candidate.brier - base.brier,
                    "mean_predicted_risk_change": (
                        candidate.mean_predicted_risk - base.mean_predicted_risk
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Read detailed model results and write two recruiter-facing summaries."""
    validation = pd.read_csv(MODEL_TABLES / "validation_candidate_metrics.csv")
    validation_summary = summarize_validation_results(validation)
    validation_summary.to_csv(
        MODEL_TABLES / "validation_candidate_summary.csv",
        index=False,
    )

    locked_test = pd.read_csv(MODEL_TABLES / "locked_test_candidate_metrics.csv")
    ablation_summary = summarize_ablation_results(locked_test)
    ablation_summary.to_csv(
        MODEL_TABLES / "locked_test_ablation_summary.csv",
        index=False,
    )

    print("Model summary tables created.")


if __name__ == "__main__":
    main()
