"""Run the approved E06 sensitivity checks and create aggregate model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.model_data import SPLITS, features, load_cohort, population, years
from src.modeling.model_pipeline import boosting, capacities, metrics

RANDOM_SEED = 42
MIN_SEGMENT_LOANS = 500
MIN_SEGMENT_CHARGEOFFS = 25
PERMUTATION_SAMPLE_SIZE = 50_000
PARTIAL_DEPENDENCE_SAMPLE_SIZE = 20_000

SEGMENT_FIELDS = [
    "approval_fy",
    "naics2",
    "project_state",
    "business_age",
    "business_type",
    "processing_method",
]
CONTINUOUS_FEATURES = [
    "gross_approval",
    "term_months",
    "guarantee_percent",
    "initial_interest_rate",
]


def load_e06_parameters() -> dict:
    """Read the frozen E06 boosting settings from the development lock."""
    lock_path = ROOT / "config/model_development_lock.json"
    with lock_path.open(encoding="utf-8") as handle:
        development_lock = json.load(handle)

    if not development_lock.get("locked_before_test"):
        raise RuntimeError("Model development configuration is not marked as locked")
    return development_lock["FY2010+"]["boost"]


def supported_segment_metrics(
    test_data: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Calculate metrics only for segments with adequate outcome support."""
    scored = test_data.copy()
    scored["predicted_probability"] = probabilities
    rows = []

    for field in SEGMENT_FIELDS:
        for value, segment in scored.groupby(field, dropna=False):
            chargeoffs = int(segment.target_120.sum())
            if (
                len(segment) < MIN_SEGMENT_LOANS
                or chargeoffs < MIN_SEGMENT_CHARGEOFFS
            ):
                continue
            rows.append(
                {
                    "segment": field,
                    "value": str(value),
                    **metrics(
                        segment.target_120.to_numpy(),
                        segment.predicted_probability.to_numpy(),
                    ),
                }
            )

    return pd.DataFrame(rows)


def permutation_importance_summary(
    model,
    test_features: pd.DataFrame,
    outcomes: np.ndarray,
) -> pd.DataFrame:
    """Summarize the loss in performance after permuting each feature."""
    random = np.random.default_rng(RANDOM_SEED)

    if len(test_features) > PERMUTATION_SAMPLE_SIZE:
        sample_index = random.choice(
            len(test_features),
            PERMUTATION_SAMPLE_SIZE,
            replace=False,
        )
        test_features = test_features.iloc[sample_index].copy()
        outcomes = outcomes[sample_index]

    baseline = model.predict_proba(test_features)[:, 1]
    baseline_pr_auc = average_precision_score(outcomes, baseline)
    baseline_brier = brier_score_loss(outcomes, baseline)
    rows = []

    for feature in test_features.columns:
        for _ in range(5):
            permuted = test_features.copy()
            permuted[feature] = random.permutation(permuted[feature].to_numpy())
            probabilities = model.predict_proba(permuted)[:, 1]
            rows.append(
                {
                    "feature": feature,
                    "pr_auc_decrease": baseline_pr_auc
                    - average_precision_score(outcomes, probabilities),
                    "brier_increase": brier_score_loss(outcomes, probabilities)
                    - baseline_brier,
                }
            )

    return (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)
        .agg(
            mean_pr_auc_decrease=("pr_auc_decrease", "mean"),
            sd_pr_auc_decrease=("pr_auc_decrease", "std"),
            mean_brier_increase=("brier_increase", "mean"),
        )
        .sort_values("mean_pr_auc_decrease", ascending=False)
    )


def partial_dependence_summary(
    model,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate bounded partial-dependence values for continuous features."""
    sample = test_features.sample(
        min(PARTIAL_DEPENDENCE_SAMPLE_SIZE, len(test_features)),
        random_state=RANDOM_SEED,
    )
    rows = []

    for feature in CONTINUOUS_FEATURES:
        grid = np.unique(
            np.quantile(sample[feature], [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        )
        for value in grid:
            scenario = sample.copy()
            scenario[feature] = value
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "mean_predicted_risk": model.predict_proba(scenario)[:, 1].mean(),
                }
            )

    return pd.DataFrame(rows)


def sensitivity_result(name: str, outcomes, probabilities) -> dict:
    """Combine model metrics and capacity results in one compact record."""
    result = {"analysis": name, **metrics(outcomes, probabilities)}
    for row in capacities(outcomes, probabilities):
        capacity = int(row["capacity_percent"])
        result[f"capture_at_{capacity}_percent"] = row["capture_percent"]
    return result


def save_model_figures(
    output_dir: Path,
    importance: pd.DataFrame,
    partial_dependence: pd.DataFrame,
) -> None:
    """Create the four model figures used in the public report."""
    figure_dir = ROOT / "reports/figures/modeling"
    figure_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics = pd.read_csv(output_dir / "locked_test_candidate_metrics.csv")
    candidate_metrics = candidate_metrics[
        candidate_metrics.experiment_id.str.fullmatch(r"E\d+")
    ]
    colors = [
        "#2b6cb0" if design == "FY2010+" else "#8a5a44"
        for design in candidate_metrics.design
    ]
    label_offsets = {
        "E01": (3, 5), "E02": (3, 5), "E03": (3, -10), "E04": (3, -10),
        "E05": (3, 6), "E06": (3, -11), "E07": (-19, 6), "E08": (3, -11),
        "E09": (3, 5), "E10": (3, 5), "E11": (3, 5), "E12": (3, 5),
        "E13": (3, 5), "E14": (3, 5), "E15": (-20, -11), "E16": (3, -11),
    }
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(candidate_metrics.pr_auc, candidate_metrics.brier, c=colors, s=55)
    for row in candidate_metrics.itertuples():
        axis.annotate(
            row.experiment_id,
            (row.pr_auc, row.brier),
            xytext=label_offsets[row.experiment_id],
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        title="Locked-test ranking and probability error",
        xlabel="PR-AUC (higher is better)",
        ylabel="Brier score (lower is better)",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_locked_test_candidates.png", dpi=180)
    plt.close(fig)

    capacity = pd.read_csv(output_dir / "locked_test_capacity_metrics.csv")
    capacity = capacity[capacity.experiment_id.eq("E06")]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        capacity.capacity_percent,
        capacity.capture_percent,
        marker="o",
        color="#2b6cb0",
    )
    axis.set(
        title="Charge-offs captured by review capacity",
        xlabel="Portfolio reviewed (%)",
        ylabel="Charge-offs captured (%)",
        xticks=[5, 10, 20],
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "02_monitoring_capacity.png", dpi=180)
    plt.close(fig)

    plot_data = importance.sort_values("mean_pr_auc_decrease")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(plot_data.feature, plot_data.mean_pr_auc_decrease, color="#2b6cb0")
    axis.set(
        title="Loss of ranking performance after feature permutation",
        xlabel="Mean decrease in PR-AUC",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "03_grouped_permutation_importance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (feature, group) in zip(
        axes.flat,
        partial_dependence.groupby("feature"),
    ):
        axis.plot(group.value, group.mean_predicted_risk, marker="o")
        axis.set_title(feature.replace("_", " ").title())
        axis.set_ylabel("Mean predicted risk")
        axis.grid(alpha=0.2)
    fig.suptitle("Bounded partial-dependence checks")
    fig.tight_layout()
    fig.savefig(figure_dir / "04_partial_dependence.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Fit E06 checks and write only aggregate outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()

    output_dir = ROOT / "reports/tables/modeling"
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort, _, _ = load_cohort(args.raw_dir)
    design_population = population(cohort, "FY2010+")
    split_definition = SPLITS["FY2010+"]
    training = years(design_population, split_definition["final_refit"])
    locked_test = years(design_population, split_definition["locked_test"])
    parameters = load_e06_parameters()

    feature_names = features("FY2010+", state=True, processing=False)
    model = boosting(feature_names, parameters).fit(
        training[feature_names],
        training.target_120,
    )
    probabilities = model.predict_proba(locked_test[feature_names])[:, 1]

    without_rate = features(
        "FY2010+",
        state=True,
        processing=False,
        interest=False,
    )
    no_rate_model = boosting(without_rate, parameters).fit(
        training[without_rate],
        training.target_120,
    )
    no_rate_probabilities = no_rate_model.predict_proba(locked_test[without_rate])[:, 1]

    model_96 = boosting(feature_names, parameters).fit(
        training[feature_names],
        training.target_96,
    )
    probabilities_96 = model_96.predict_proba(locked_test[feature_names])[:, 1]

    sensitivity_rows = [
        sensitivity_result(
            "primary_E06_120_month",
            locked_test.target_120.to_numpy(),
            probabilities,
        ),
        sensitivity_result(
            "interest_rate_removed_120_month",
            locked_test.target_120.to_numpy(),
            no_rate_probabilities,
        ),
        sensitivity_result(
            "same_cohort_96_month_sensitivity",
            locked_test.target_96.to_numpy(),
            probabilities_96,
        ),
    ]
    pd.DataFrame(sensitivity_rows).to_csv(
        output_dir / "model_sensitivity_summary.csv",
        index=False,
    )

    e06_segments = supported_segment_metrics(locked_test, probabilities)
    e06_segments.insert(0, "experiment_id", "E06")
    core_features = features("FY2010+", state=False, processing=False)
    core_model = boosting(core_features, parameters).fit(
        training[core_features],
        training.target_120,
    )
    core_probabilities = core_model.predict_proba(locked_test[core_features])[:, 1]
    e05_segments = supported_segment_metrics(locked_test, core_probabilities)
    e05_segments.insert(0, "experiment_id", "E05")
    pd.concat([e05_segments, e06_segments], ignore_index=True).to_csv(
        output_dir / "selected_model_supported_segment_metrics.csv",
        index=False,
    )

    importance = permutation_importance_summary(
        model,
        locked_test[feature_names],
        locked_test.target_120.to_numpy(),
    )
    importance.to_csv(
        output_dir / "selected_model_importance_summary.csv",
        index=False,
    )
    partial_dependence = partial_dependence_summary(
        model,
        locked_test[feature_names],
    )
    partial_dependence.to_csv(
        output_dir / "selected_model_partial_dependence.csv",
        index=False,
    )
    save_model_figures(output_dir, importance, partial_dependence)

    print("Sensitivity checks and aggregate model outputs completed.")


if __name__ == "__main__":
    main()
