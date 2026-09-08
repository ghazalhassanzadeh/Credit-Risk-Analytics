"""Build safe aggregate data sources for the Tableau dashboards.

The script reads only approved portfolio, statistical and model-evaluation aggregates. It does
not read or write borrower-level records, predictions, names, addresses or IDs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "reports" / "tables"
OUT = ROOT / "tableau" / "data"
VALIDATION = ROOT / "reports" / "tables" / "tableau"


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / relative)


def write_csv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT / filename, index=False, float_format="%.10g")


def build_portfolio_vintage() -> pd.DataFrame:
    history = read_csv("portfolio/historical_approval_trend.csv")
    risk = read_csv("portfolio/vintage_risk.csv")
    risk = risk[(risk["candidate_start_fy"] == 2001) & (risk["horizon_months"] == 120)]
    keep = [
        "approval_fy",
        "eligible_loans",
        "chargeoffs",
        "chargeoff_rate_percent",
        "ci95_low_percent",
        "ci95_high_percent",
        "passes_rate_reporting_rule",
        "approval_weighted_chargeoff_incidence_percent",
        "gross_chargeoff_rate_percent",
        "sensitivity_gross_chargeoff_rate_percent",
    ]
    out = history.merge(risk[keep], on="approval_fy", how="left")
    out["complete_120_month_window"] = out["eligible_loans"].notna()
    out["partial_fiscal_year"] = out["approval_fy"].eq(2026)
    out["risk_display_allowed"] = out["passes_rate_reporting_rule"].eq(True)
    return out


def build_segment_risk_exposure() -> pd.DataFrame:
    source = read_csv("portfolio/financial_concentration.csv")
    # These are the four dimensions approved for the executive selector.
    allowed = ["loan_size_band", "term_band", "naics2", "business_age"]
    out = source[
        (source["candidate_start_fy"] == 2010)
        & (source["horizon_months"] == 120)
        & source["dimension"].isin(allowed)
    ].copy()
    out["risk_display_allowed"] = out["passes_rate_reporting_rule"].fillna(False)
    for column in ["chargeoff_rate_percent", "ci95_low_percent", "ci95_high_percent"]:
        out[f"display_{column}"] = out[column].where(out["risk_display_allowed"])
    columns = [
        "dimension",
        "segment",
        "eligible_loans",
        "chargeoffs",
        "gross_approval",
        "gross_approval_share_percent",
        "chargeoff_loan_approval",
        "chargeoff_loan_approval_share_percent",
        "gross_chargeoff_amount",
        "gross_chargeoff_amount_share_percent",
        "sensitivity_gross_chargeoff_amount",
        "sensitivity_gross_chargeoff_amount_share_percent",
        "financial_anomaly_share_shift_pp",
        "display_chargeoff_rate_percent",
        "display_ci95_low_percent",
        "display_ci95_high_percent",
        "risk_display_allowed",
        "approval_weighted_chargeoff_incidence_percent",
        "gross_chargeoff_rate_percent",
        "sensitivity_gross_chargeoff_rate_percent",
        "flagged_financial_chargeoffs",
    ]
    return out[columns]


def build_term_horizon() -> pd.DataFrame:
    source = read_csv("portfolio/common_cohort_horizon_sensitivity.csv")
    out = source[
        (source["candidate_start_fy"] == 2010)
        & source["dimension"].isin(["overall", "term_band"])
    ].copy()
    out["valid_term_for_primary_model"] = ~out["segment"].eq("Invalid or missing")
    return out.drop(columns=["candidate_start_fy"])


def build_observed_adjusted() -> pd.DataFrame:
    categorical = read_csv("statistics/categorical_adjusted_effects.csv")
    categorical = categorical[~categorical["factor"].eq("processing_method")].copy()
    categorical["factor"] = categorical["factor"].replace(
        {
            "naics2_model": "Two-digit NAICS",
            "project_state_model": "ProjectState",
            "business_age": "Business age",
            "business_type": "Business type",
        }
    )
    categorical["comparison_type"] = "Observed versus adjusted"
    categorical = categorical.rename(columns={"category": "comparison_value"})
    categorical["scenario_value"] = np.nan

    continuous = read_csv("statistics/continuous_adjusted_effects.csv")
    continuous["factor"] = continuous["factor"].replace(
        {
            "gross_approval": "Gross Approval",
            "term_months": "Original term",
            "guarantee_percent": "Guarantee percentage",
            "initial_interest_rate": "Initial interest rate",
        }
    )
    continuous["comparison_type"] = "Adjusted curve point"
    continuous["comparison_value"] = continuous["scenario_value"].map(
        lambda x: f"{x:g}"
    )
    continuous["category"] = continuous["comparison_value"]
    continuous["reference_category"] = continuous["reference_value"].map(
        lambda x: f"{x:g}"
    )
    continuous["pooled_risk_percent"] = np.nan
    continuous["pooled_ci95_low_percent"] = np.nan
    continuous["pooled_ci95_high_percent"] = np.nan
    continuous["vintage_standardized_risk_percent"] = np.nan
    continuous["eligible_loans"] = np.nan
    continuous["chargeoffs"] = np.nan
    continuous["vintages_passing_500_25"] = np.nan
    continuous["fdr_q_value"] = np.nan

    columns = [
        "comparison_type",
        "factor",
        "comparison_value",
        "reference_category",
        "scenario_value",
        "eligible_loans",
        "chargeoffs",
        "pooled_risk_percent",
        "pooled_ci95_low_percent",
        "pooled_ci95_high_percent",
        "vintage_standardized_risk_percent",
        "mix_adjusted_risk_percent",
        "adjusted_risk_difference_vs_reference_pp",
        "adjusted_risk_difference_ci95_low_pp",
        "adjusted_risk_difference_ci95_high_pp",
        "vintages_passing_500_25",
        "fdr_q_value",
    ]
    return pd.concat([categorical[columns], continuous[columns]], ignore_index=True)


def build_geography_detail() -> pd.DataFrame:
    segments = read_csv("portfolio/financial_concentration.csv")
    segments = segments[
        (segments["candidate_start_fy"] == 2010)
        & (segments["horizon_months"] == 120)
        & (segments["dimension"] == "project_state")
    ].copy()
    adjusted = read_csv("statistics/categorical_adjusted_effects.csv")
    adjusted = adjusted[adjusted["factor"] == "project_state_model"].copy()
    model = read_csv("modeling/selected_model_supported_segment_metrics.csv")
    model = model[
        (model["experiment_id"] == "E06") & (model["segment"] == "project_state")
    ].copy()

    out = segments.merge(
        adjusted[
            [
                "category",
                "mix_adjusted_risk_percent",
                "adjusted_risk_difference_vs_reference_pp",
                "adjusted_risk_difference_ci95_low_pp",
                "adjusted_risk_difference_ci95_high_pp",
                "vintages_passing_500_25",
                "fdr_q_value",
            ]
        ],
        left_on="segment",
        right_on="category",
        how="left",
    ).merge(
        model[
            [
                "value",
                "loans",
                "chargeoffs",
                "prevalence",
                "brier",
                "calibration_intercept",
                "calibration_slope",
                "mean_predicted_risk",
            ]
        ],
        left_on="segment",
        right_on="value",
        how="left",
        suffixes=("", "_model_test"),
    )
    out["risk_display_allowed"] = out["passes_rate_reporting_rule"].fillna(False)
    for column in ["chargeoff_rate_percent", "ci95_low_percent", "ci95_high_percent"]:
        out[f"display_{column}"] = out[column].where(out["risk_display_allowed"])
    out["model_calibration_available"] = out["loans"].notna()
    columns = [
        "segment",
        "eligible_loans",
        "chargeoffs",
        "gross_approval",
        "gross_approval_share_percent",
        "display_chargeoff_rate_percent",
        "display_ci95_low_percent",
        "display_ci95_high_percent",
        "risk_display_allowed",
        "mix_adjusted_risk_percent",
        "adjusted_risk_difference_vs_reference_pp",
        "adjusted_risk_difference_ci95_low_pp",
        "adjusted_risk_difference_ci95_high_pp",
        "vintages_passing_500_25",
        "loans",
        "chargeoffs_model_test",
        "prevalence",
        "mean_predicted_risk",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "model_calibration_available",
    ]
    return out[columns].rename(
        columns={"segment": "project_state", "loans": "model_test_loans"}
    )


def build_model_performance() -> pd.DataFrame:
    source = read_csv("modeling/locked_test_candidate_metrics.csv")
    roles = {
        "E06": "Primary model",
        "E02": "Logistic baseline",
        "E08": "ProcessingMethod sensitivity",
        "E14": "FY2001+ stress comparison",
    }
    out = source[source["experiment_id"].isin(roles)].copy()
    out["model_role"] = out["experiment_id"].map(roles)
    out["display_order"] = out["experiment_id"].map(
        {"E06": 1, "E02": 2, "E08": 3, "E14": 4}
    )
    return out.sort_values("display_order")


def build_model_capacity() -> pd.DataFrame:
    source = read_csv("modeling/locked_test_capacity_metrics.csv")
    out = source[source["experiment_id"].eq("E06")].copy()
    out["main_illustration"] = out["capacity_percent"].eq(10)
    out["scenario_label"] = out["capacity_percent"].map(lambda x: f"Top {x:g}%")
    return out


def build_model_calibration() -> pd.DataFrame:
    source = read_csv("modeling/locked_test_calibration_bins.csv")
    return source[source["experiment_id"].eq("E06")].copy()


def build_model_vintage() -> pd.DataFrame:
    source = read_csv("modeling/locked_test_vintage_metrics.csv")
    out = source[source["experiment_id"].eq("E06")].copy()
    out["vintage_label"] = out["approval_fy"].map(
        {2015: "FY2015", 2016: "FY2016 eligible approvals"}
    )
    return out


def build_feature_contribution() -> pd.DataFrame:
    out = read_csv("modeling/selected_model_importance_summary.csv")
    labels = {
        "term_months": "Original term",
        "initial_interest_rate": "Initial interest rate",
        "guarantee_percent": "Guarantee percentage",
        "project_state": "ProjectState",
        "gross_approval": "Gross Approval",
        "naics2": "Two-digit NAICS",
        "business_type": "Business type",
        "business_age": "Business age",
    }
    out["feature_label"] = out["feature"].map(labels)
    out["rank"] = (
        out["mean_pr_auc_decrease"].rank(method="first", ascending=False).astype(int)
    )
    return out.sort_values("rank")


def build_financial_sensitivity() -> pd.DataFrame:
    source = read_csv("portfolio/candidate_population_summary.csv")
    out = source[
        (source["candidate_start_fy"] == 2010) & (source["horizon_months"] == 120)
    ].copy()
    out["measure_label"] = "Recorded GrossChargeOffAmount as a share of Gross Approval"
    return out[
        [
            "candidate_start_fy",
            "horizon_months",
            "eligible_loans",
            "gross_approval",
            "gross_chargeoff_amount",
            "flagged_financial_chargeoffs",
            "gross_chargeoff_rate_percent",
            "sensitivity_gross_chargeoff_rate_percent",
            "measure_label",
        ]
    ]


def validate(outputs: dict[str, pd.DataFrame]) -> dict:
    checks = []

    def check(
        name: str, actual: float, expected: float, tolerance: float = 1e-8
    ) -> None:
        passed = bool(abs(actual - expected) <= tolerance)
        checks.append(
            {
                "check": name,
                "actual": actual,
                "expected": expected,
                "tolerance": tolerance,
                "passed": passed,
            }
        )

    perf = outputs["tableau_model_performance.csv"].set_index("experiment_id")
    e06 = perf.loc["E06"]
    check("E06 PR-AUC", e06["pr_auc"], 0.6563126262700382)
    check("E06 ROC-AUC", e06["roc_auc"], 0.9638243859594409)
    check("E06 Brier", e06["brier"], 0.03334720446914445)
    check("E06 event rate", e06["prevalence"], 0.06711891542400887)
    check("E06 mean prediction", e06["mean_predicted_risk"], 0.0585214209551366)

    capacity = outputs["tableau_model_capacity.csv"].set_index("capacity_percent")
    check("Top 5% capture", capacity.loc[5, "capture_percent"], 52.55351681957187)
    check("Top 10% capture", capacity.loc[10, "capture_percent"], 82.30886850152905)
    check("Top 20% capture", capacity.loc[20, "capture_percent"], 95.73394495412843)

    fin = outputs["tableau_financial_sensitivity.csv"].iloc[0]
    check(
        "Reported financial ratio", fin["gross_chargeoff_rate_percent"], 2.816299, 1e-6
    )
    check(
        "Sensitivity financial ratio",
        fin["sensitivity_gross_chargeoff_rate_percent"],
        2.68848,
        1e-6,
    )

    vintage = outputs["tableau_portfolio_vintage.csv"].set_index("approval_fy")
    check(
        "FY2007 ten-year rate",
        vintage.loc[2007, "chargeoff_rate_percent"],
        35.777977,
        1e-6,
    )

    geography = outputs["tableau_geography_detail.csv"]
    suppressed_have_rates = (
        geography.loc[
            ~geography["risk_display_allowed"], "display_chargeoff_rate_percent"
        ]
        .notna()
        .sum()
    )
    check("Suppressed state rates exposed", float(suppressed_have_rates), 0.0)

    forbidden = {
        "borrowername",
        "borrowerstreet",
        "borrowerzip",
        "lendername",
        "loan_id",
        "borrower_id",
    }
    columns = {c.lower() for frame in outputs.values() for c in frame.columns}
    check("Forbidden identifier columns", float(len(columns & forbidden)), 0.0)

    return {"all_passed": all(row["passed"] for row in checks), "checks": checks}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VALIDATION.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tableau_portfolio_vintage.csv": build_portfolio_vintage(),
        "tableau_segment_risk_exposure.csv": build_segment_risk_exposure(),
        "tableau_term_horizon_sensitivity.csv": build_term_horizon(),
        "tableau_observed_adjusted_risk.csv": build_observed_adjusted(),
        "tableau_geography_detail.csv": build_geography_detail(),
        "tableau_model_performance.csv": build_model_performance(),
        "tableau_model_capacity.csv": build_model_capacity(),
        "tableau_model_calibration.csv": build_model_calibration(),
        "tableau_model_vintage.csv": build_model_vintage(),
        "tableau_feature_contribution.csv": build_feature_contribution(),
        "tableau_financial_sensitivity.csv": build_financial_sensitivity(),
    }
    for filename, frame in outputs.items():
        write_csv(frame, filename)

    result = validate(outputs)
    pd.DataFrame(result["checks"]).to_csv(
        VALIDATION / "tableau_reconciliation_checks.csv", index=False
    )
    if not result["all_passed"]:
        raise RuntimeError("One or more Tableau reconciliation checks failed")


if __name__ == "__main__":
    main()
