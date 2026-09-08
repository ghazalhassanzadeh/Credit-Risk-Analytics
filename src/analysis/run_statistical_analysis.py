"""Estimate adjusted relationships for the ten-year charge-off outcome.

The script reads the cleaned local cohort and writes aggregate statistical
results and figures. It does not export loan-level records.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_cohort import wilson_interval  # noqa: E402

DEFAULT_COHORT = PROJECT_ROOT / "data/processed/sba_7a_analysis_cohort.parquet"
MIN_LOANS = 500
MIN_EVENTS = 25
MIN_OVERLAP_VINTAGES = 4


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    values = pvalues.to_numpy(float)
    valid = np.isfinite(values)
    result = np.full(len(values), np.nan)
    if not valid.any():
        return pd.Series(result, index=pvalues.index)
    selected = values[valid]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1)
    result[np.flatnonzero(valid)] = restored
    return pd.Series(result, index=pvalues.index)


def load_analysis_cohort(path: Path) -> pd.DataFrame:
    """Load and validate the cleaned FY2010+ analysis cohort."""
    data = pd.read_parquet(path)
    required = {
        "approval_fy",
        "target_120",
        "target_96",
        "gross_approval",
        "term_months",
        "guarantee_percent",
        "initial_interest_rate",
        "naics2",
        "project_state",
        "business_age",
        "business_type",
        "processing_method",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Analysis cohort is missing columns: {sorted(missing)}")
    if data[list(required)].isna().any().any():
        raise ValueError("Analysis cohort contains unexpected missing values.")
    if not set(data["target_120"].unique()).issubset({0, 1}):
        raise ValueError("target_120 must contain only 0 and 1.")

    data = data.copy()
    data["log_gross_approval"] = np.log1p(data["gross_approval"])
    data["processing_method_raw"] = data["processing_method"].astype(str)
    return data


def reporting_categories(data: pd.DataFrame, field: str) -> set[str]:
    counts = data.groupby(field)["target_120"].agg(["size", "sum"])
    return set(
        counts[
            (counts["size"] >= MIN_LOANS) & (counts["sum"] >= MIN_EVENTS)
        ].index.astype(str)
    )


def processing_overlap(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in data.groupby("processing_method_raw"):
        by_vintage = group.groupby("approval_fy")["target_120"].agg(["size", "sum"])
        passing = (
            by_vintage["size"].ge(MIN_LOANS) & by_vintage["sum"].ge(MIN_EVENTS)
        ).sum()
        rows.append(
            {
                "processing_method": method,
                "eligible_loans": len(group),
                "chargeoffs": int(group["target_120"].sum()),
                "vintages_present": int(by_vintage.shape[0]),
                "vintages_passing_500_25": int(passing),
                "adequate_overlap_for_named_comparison": bool(
                    len(group) >= MIN_LOANS
                    and group["target_120"].sum() >= MIN_EVENTS
                    and passing >= MIN_OVERLAP_VINTAGES
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["adequate_overlap_for_named_comparison", "eligible_loans"],
        ascending=[False, False],
    )


def make_categories(
    data: pd.DataFrame, overlap: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    data = data.copy()
    naics_keep = reporting_categories(data, "naics2")
    state_keep = reporting_categories(data, "project_state")
    process_keep = set(
        overlap.loc[
            overlap["adequate_overlap_for_named_comparison"], "processing_method"
        ]
    )
    data["naics2_model"] = data["naics2"].where(
        data["naics2"].isin(naics_keep), "Other or thin"
    )
    data["project_state_model"] = data["project_state"].where(
        data["project_state"].isin(state_keep), "Other or thin"
    )
    data["processing_method"] = data["processing_method_raw"].where(
        data["processing_method_raw"].isin(process_keep), "Other or non-comparable"
    )

    references = {
        "approval_fy": "2010",
        "naics2_model": data["naics2_model"].value_counts().index[0],
        "project_state_model": (
            "CA"
            if "CA" in set(data["project_state_model"])
            else data["project_state_model"].value_counts().index[0]
        ),
        "business_age": "Existing",
        "business_type": "CORPORATION",
        "processing_method": (
            "SBA Express Program"
            if "SBA Express Program" in process_keep
            else data["processing_method"].value_counts().index[0]
        ),
    }
    data["approval_fy"] = data["approval_fy"].astype(str)
    categorical_fields = list(references)
    category_orders = {}
    for field in categorical_fields:
        levels = sorted(data[field].astype(str).unique())
        ref = str(references[field])
        category_orders[field] = [ref] + [level for level in levels if level != ref]
        data[field] = data[field].astype(str)
    return data, category_orders, references


def build_design(data: pd.DataFrame, category_orders: dict[str, list[str]]):
    categorical_fields = list(category_orders)
    encoder = OneHotEncoder(
        categories=[category_orders[field] for field in categorical_fields],
        drop="first",
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )
    x_cat = encoder.fit_transform(data[categorical_fields])
    cat_names = list(encoder.get_feature_names_out(categorical_fields))
    blocks = [sparse.csr_matrix(np.ones((len(data), 1))), x_cat]
    names = ["intercept"] + cat_names
    groups = {field: [] for field in categorical_fields}
    for index, name in enumerate(cat_names, start=1):
        for field in categorical_fields:
            if name.startswith(field + "_"):
                groups[field].append(index)
                break

    spline_fields = [
        "log_gross_approval",
        "term_months",
        "guarantee_percent",
        "initial_interest_rate",
    ]
    spline_objects = {}
    for field in spline_fields:
        transformer = SplineTransformer(
            n_knots=5, degree=3, include_bias=False, knots="quantile"
        )
        block = transformer.fit_transform(data[[field]])
        start = len(names)
        blocks.append(sparse.csr_matrix(block))
        field_names = [f"{field}_spline_{i + 1}" for i in range(block.shape[1])]
        names.extend(field_names)
        groups[field] = list(range(start, start + block.shape[1]))
        spline_objects[field] = transformer
    return sparse.hstack(blocks, format="csr"), names, groups, encoder, spline_objects


def fit_logistic(
    x: sparse.csr_matrix, y: np.ndarray
) -> tuple[LogisticRegression, np.ndarray]:
    model = LogisticRegression(
        C=np.inf,
        fit_intercept=False,
        solver="newton-cholesky",
        max_iter=100,
        tol=1e-8,
    )
    model.fit(x, y)
    if int(model.n_iter_[0]) >= model.max_iter:
        raise RuntimeError("Logistic model did not converge. Results were not written.")
    predictions = model.predict_proba(x)[:, 1]
    return model, predictions


def robust_covariance(x: sparse.csr_matrix, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    w = np.clip(p * (1 - p), 1e-9, None)
    bread_matrix = (x.T @ x.multiply(w[:, None])).toarray()
    bread = np.linalg.pinv(bread_matrix, rcond=1e-10)
    residual_sq = np.square(y - p)
    meat = (x.T @ x.multiply(residual_sq[:, None])).toarray()
    return bread @ meat @ bread


def scenario_gradient(
    x: sparse.csr_matrix,
    eta_without_group: np.ndarray,
    beta_add: float,
    group_indices: list[int],
    target_index: int | None,
) -> tuple[float, np.ndarray]:
    p = expit(eta_without_group + beta_add)
    derivative = p * (1 - p)
    gradient = np.asarray(x.T @ derivative).reshape(-1) / len(p)
    gradient[group_indices] = 0
    if target_index is not None:
        gradient[target_index] = derivative.mean()
    return float(p.mean()), gradient


def categorical_effects(
    data: pd.DataFrame,
    x: sparse.csr_matrix,
    beta: np.ndarray,
    cov: np.ndarray,
    names: list[str],
    groups: dict[str, list[int]],
    references: dict[str, str],
    category_orders: dict[str, list[str]],
) -> pd.DataFrame:
    eta = np.asarray(x @ beta).reshape(-1)
    rows = []
    for field in [
        "naics2_model",
        "project_state_model",
        "business_age",
        "business_type",
        "processing_method",
    ]:
        indices = groups[field]
        contribution = (
            np.asarray(x[:, indices] @ beta[indices]).reshape(-1)
            if indices
            else np.zeros(len(data))
        )
        base_eta = eta - contribution
        reference = str(references[field])
        ref_risk, ref_gradient = scenario_gradient(x, base_eta, 0, indices, None)
        for category in category_orders[field]:
            if category == reference:
                risk, gradient, coefficient, odds_ratio = (
                    ref_risk,
                    ref_gradient,
                    0.0,
                    1.0,
                )
            else:
                name = f"{field}_{category}"
                column = names.index(name)
                coefficient = beta[column]
                odds_ratio = float(np.exp(coefficient))
                risk, gradient = scenario_gradient(
                    x, base_eta, coefficient, indices, column
                )
            difference = risk - ref_risk
            diff_gradient = gradient - ref_gradient
            se = float(np.sqrt(max(diff_gradient @ cov @ diff_gradient, 0)))
            z = difference / se if se else np.nan
            pooled = (
                data.groupby(field)["target_120"].agg(["size", "sum"]).loc[category]
            )
            by_vintage = (
                data[data[field].eq(category)]
                .groupby("approval_fy")["target_120"]
                .agg(["size", "sum"])
            )
            portfolio_by_vintage = data.groupby("approval_fy")["target_120"].mean()
            common = by_vintage.index.intersection(portfolio_by_vintage.index)
            weights = data["approval_fy"].value_counts(normalize=True).reindex(common)
            weights = weights / weights.sum()
            vintage_adjusted = float(
                (
                    weights
                    * (by_vintage.loc[common, "sum"] / by_vintage.loc[common, "size"])
                ).sum()
            )
            passing = by_vintage["size"].ge(MIN_LOANS) & by_vintage["sum"].ge(
                MIN_EVENTS
            )
            gaps = by_vintage.loc[passing, "sum"] / by_vintage.loc[
                passing, "size"
            ] - portfolio_by_vintage.reindex(by_vintage.index[passing])
            low, high = wilson_interval(int(pooled["sum"]), int(pooled["size"]))
            rows.append(
                {
                    "factor": field,
                    "category": category,
                    "reference_category": reference,
                    "eligible_loans": int(pooled["size"]),
                    "chargeoffs": int(pooled["sum"]),
                    "pooled_risk_percent": 100 * pooled["sum"] / pooled["size"],
                    "pooled_ci95_low_percent": 100 * low,
                    "pooled_ci95_high_percent": 100 * high,
                    "vintage_standardized_risk_percent": 100 * vintage_adjusted,
                    "mix_adjusted_risk_percent": 100 * risk,
                    "adjusted_risk_difference_vs_reference_pp": 100 * difference,
                    "adjusted_risk_difference_ci95_low_pp": 100
                    * (difference - 1.9599639845 * se),
                    "adjusted_risk_difference_ci95_high_pp": 100
                    * (difference + 1.9599639845 * se),
                    "adjusted_odds_ratio_vs_reference": odds_ratio,
                    "p_value_adjusted_contrast": (
                        2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
                    ),
                    "vintages_passing_500_25": int(passing.sum()),
                    "vintages_above_portfolio_rate": int((gaps > 0).sum()),
                    "vintages_below_portfolio_rate": int((gaps < 0).sum()),
                }
            )
    result = pd.DataFrame(rows)
    result["fdr_q_value"] = np.nan
    for factor in ["naics2_model", "project_state_model"]:
        mask = result["factor"].eq(factor) & result["category"].ne(
            result["reference_category"]
        )
        result.loc[mask, "fdr_q_value"] = bh_adjust(
            result.loc[mask, "p_value_adjusted_contrast"]
        )
    return result.round(8)


def continuous_effects(
    data: pd.DataFrame,
    x: sparse.csr_matrix,
    beta: np.ndarray,
    cov: np.ndarray,
    groups: dict[str, list[int]],
    transformers: dict[str, SplineTransformer],
    target: str = "target_120",
) -> pd.DataFrame:
    eta = np.asarray(x @ beta).reshape(-1)
    rows = []
    for field in [
        "log_gross_approval",
        "term_months",
        "guarantee_percent",
        "initial_interest_rate",
    ]:
        source = "gross_approval" if field == "log_gross_approval" else field
        values = data[source]
        if field == "term_months":
            scenarios = np.array([12, 60, 84, 120, 180, 240], dtype=float)
        else:
            scenarios = np.unique(
                values.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_numpy(float)
            )
        transformed_values = (
            np.log1p(scenarios) if field == "log_gross_approval" else scenarios
        )
        basis = transformers[field].transform(pd.DataFrame({field: transformed_values}))
        indices = groups[field]
        current = np.asarray(x[:, indices] @ beta[indices]).reshape(-1)
        base_eta = eta - current
        reference_position = int(np.argmin(np.abs(scenarios - np.median(values))))
        scenario_results = []
        for value, b in zip(scenarios, basis):
            p = expit(base_eta + float(b @ beta[indices]))
            derivative = p * (1 - p)
            gradient = np.asarray(x.T @ derivative).reshape(-1) / len(p)
            gradient[indices] = derivative.mean() * b
            scenario_results.append((value, float(p.mean()), gradient))
        ref_value, ref_risk, ref_gradient = scenario_results[reference_position]
        for value, risk, gradient in scenario_results:
            difference = risk - ref_risk
            diff_gradient = gradient - ref_gradient
            se = float(np.sqrt(max(diff_gradient @ cov @ diff_gradient, 0)))
            z = difference / se if se else np.nan
            rows.append(
                {
                    "factor": source,
                    "scenario_value": value,
                    "reference_value": ref_value,
                    "mix_adjusted_risk_percent": 100 * risk,
                    "adjusted_risk_difference_vs_reference_pp": 100 * difference,
                    "adjusted_risk_difference_ci95_low_pp": 100
                    * (difference - 1.9599639845 * se),
                    "adjusted_risk_difference_ci95_high_pp": 100
                    * (difference + 1.9599639845 * se),
                    "p_value_adjusted_contrast": (
                        2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
                    ),
                    "target": target,
                }
            )
    return pd.DataFrame(rows).round(8)


def term_horizon_sensitivity(
    data: pd.DataFrame,
    x: sparse.csr_matrix,
    groups: dict[str, list[int]],
    transformers: dict[str, SplineTransformer],
) -> pd.DataFrame:
    results = []
    for target in ["target_96", "target_120"]:
        model, p = fit_logistic(x, data[target].to_numpy())
        cov = robust_covariance(x, data[target].to_numpy(), p)
        table = continuous_effects(
            data, x, model.coef_.reshape(-1), cov, groups, transformers, target
        )
        results.append(table[table["factor"].eq("term_months")])
    combined = pd.concat(results, ignore_index=True)
    wide = combined.pivot(
        index=["factor", "scenario_value", "reference_value"],
        columns="target",
        values="mix_adjusted_risk_percent",
    ).reset_index()
    wide["risk_difference_120_minus_96_pp"] = wide["target_120"] - wide["target_96"]
    return wide.round(8)


def save_figures(
    out: Path,
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
    term_sensitivity: pd.DataFrame,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, factor, title in [
        (
            axes[0],
            "gross_approval",
            "Loan-size pattern reverses after portfolio-mix adjustment",
        ),
        (axes[1], "term_months", "Adjusted term relationship is non-linear"),
    ]:
        data = continuous[continuous["factor"].eq(factor)].sort_values("scenario_value")
        xvalue = (
            data["scenario_value"] / 1000
            if factor == "gross_approval"
            else data["scenario_value"]
        )
        ax.plot(
            xvalue,
            data["mix_adjusted_risk_percent"],
            marker="o",
            color="#24557a",
            linewidth=2,
        )
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(
            "Gross approval ($000s)"
            if factor == "gross_approval"
            else "Original term (months)"
        )
        ax.set_ylabel("Adjusted ten-year risk (%)")
    fig.tight_layout()
    fig.savefig(out / "01_adjusted_size_and_term.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    industry = categorical[
        (categorical["factor"].eq("naics2_model"))
        & categorical["category"].ne("Other or thin")
    ]
    industry = industry.sort_values("mix_adjusted_risk_percent")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        industry["category"], industry["mix_adjusted_risk_percent"], color="#487a9b"
    )
    ax.set_title(
        "Industry differences narrow after portfolio-mix adjustment",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("Adjusted ten-year risk (%)")
    ax.set_ylabel("Two-digit NAICS")
    fig.tight_layout()
    fig.savefig(out / "02_adjusted_industry_risk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    state = categorical[
        (categorical["factor"].eq("project_state_model"))
        & categorical["category"].ne("Other or thin")
    ].copy()
    state = state[
        state["fdr_q_value"].lt(0.05)
        & state["adjusted_risk_difference_vs_reference_pp"].abs().ge(1)
        & state["vintages_passing_500_25"].ge(MIN_OVERLAP_VINTAGES)
    ]
    state = state.sort_values("adjusted_risk_difference_vs_reference_pp")
    if not state.empty:
        fig, ax = plt.subplots(figsize=(9, max(5, len(state) * 0.32)))
        ax.barh(
            state["category"],
            state["adjusted_risk_difference_vs_reference_pp"],
            color="#b45f35",
        )
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(
            "Stable state contrasts retained after multiplicity and mix controls",
            loc="left",
            weight="bold",
        )
        ax.set_xlabel("Adjusted risk difference from California (percentage points)")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(
            out / "03_adjusted_state_contrasts.png", dpi=180, bbox_inches="tight"
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    t = term_sensitivity.sort_values("scenario_value")
    ax.plot(
        t["scenario_value"],
        t["target_120"],
        marker="o",
        label="120-month outcome",
        linewidth=2,
    )
    ax.plot(
        t["scenario_value"],
        t["target_96"],
        marker="o",
        label="96-month outcome",
        linewidth=2,
    )
    ax.set_title(
        "The shorter horizon changes the term profile unevenly",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("Original term (months)")
    ax.set_ylabel("Adjusted risk (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "04_term_horizon_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-file", type=Path, default=DEFAULT_COHORT)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/tables/statistics")
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("reports/figures/statistics")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_analysis_cohort(args.cohort_file)
    overlap = processing_overlap(cohort)
    cohort, category_orders, references = make_categories(cohort, overlap)
    x, names, groups, _, transformers = build_design(cohort, category_orders)
    y = cohort["target_120"].to_numpy()
    model, predictions = fit_logistic(x, y)
    beta = model.coef_.reshape(-1)
    covariance = robust_covariance(x, y, predictions)

    categorical = categorical_effects(
        cohort, x, beta, covariance, names, groups, references, category_orders
    )
    continuous = continuous_effects(cohort, x, beta, covariance, groups, transformers)
    term_sensitivity = term_horizon_sensitivity(cohort, x, groups, transformers)

    categorical.to_csv(
        args.output_dir / "categorical_adjusted_effects.csv", index=False
    )
    continuous.to_csv(args.output_dir / "continuous_adjusted_effects.csv", index=False)
    term_sensitivity.to_csv(
        args.output_dir / "term_96_120_adjusted_sensitivity.csv", index=False
    )
    save_figures(args.figure_dir, categorical, continuous, term_sensitivity)
    print(
        f"Statistical population: {len(cohort):,}; "
        f"charge-offs: {int(y.sum()):,}; design columns: {x.shape[1]}"
    )


if __name__ == "__main__":
    main()
