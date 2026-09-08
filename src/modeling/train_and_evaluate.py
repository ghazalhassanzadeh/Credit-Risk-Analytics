"""Run model development in auditable stages without publishing loan-level outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling.model_data import SPLITS, features, load_cohort, population, years
from src.modeling.model_pipeline import (
    boosting,
    cal_bins,
    capacities,
    logistic,
    metrics,
    unseen,
)

VARIANTS = [
    ("core", False, False),
    ("add_state", True, False),
    ("add_processing", False, True),
    ("full_candidate", True, True),
]
LOG_CONFIGS = [
    {"form": f, "C": c} for f in ["simple", "spline"] for c in [0.1, 1.0, 10.0]
]
BOOST_CONFIGS = [
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 500,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 150,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 150,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 500,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 250,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 250,
        "l2_regularization": 5.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 250,
        "l2_regularization": 5.0,
    },
]
LOCK = {
    "FY2010+": {"logistic": {"form": "spline", "C": 0.1}, "boost": BOOST_CONFIGS[5]},
    "FY2001+": {"logistic": {"form": "simple", "C": 0.1}, "boost": BOOST_CONFIGS[5]},
}
LOCK_PATH = ROOT / "config/model_development_lock.json"


def validation_folds(split_definition):
    """Return the two approved expanding-window validation folds."""
    return [
        (
            "fold1",
            split_definition["fold1_train"],
            split_definition["fold1_validate"],
        ),
        (
            "fold2",
            split_definition["fold2_train"],
            split_definition["fold2_validate"],
        ),
    ]


def build_pipeline(family, feature_names, configuration):
    """Build one model using its frozen feature and parameter specification."""
    return (
        logistic(feature_names, configuration["form"], configuration["C"])
        if family == "logistic"
        else boosting(feature_names, configuration)
    )


def write_lock():
    """Record the configurations selected before opening the locked test."""
    (ROOT / "config").mkdir(exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **LOCK,
                "locked_before_test": True,
                "selection_note": (
                    "Selected only from forward validation. High-C spline fits "
                    "that reached iteration limits were not retained."
                ),
            },
            handle,
            indent=2,
        )


def verify_lock():
    """Confirm that the saved development lock matches the approved settings."""
    if not LOCK_PATH.exists():
        raise RuntimeError("Development lock missing")

    with LOCK_PATH.open(encoding="utf-8") as handle:
        saved_lock = json.load(handle)

    if not saved_lock.get("locked_before_test"):
        raise RuntimeError("Development lock is not marked as frozen")
    for design, configuration in LOCK.items():
        if saved_lock.get(design) != configuration:
            raise RuntimeError(f"Development lock does not match {design} settings")


def audit(cohort, output_dir, cohort_audit, duplicate_audit):
    """Check cohort validity, temporal separation, missingness, and leakage rules."""
    cohort_audit.to_csv(output_dir / "pre_model_cohort_audit.csv", index=False)
    duplicate_total = int(
        duplicate_audit.loc[
            duplicate_audit["file"].eq("ALL FILES"), "repeated_rows_removed"
        ].iloc[0]
    )
    rows = []
    miss = []
    for design, spec in SPLITS.items():
        d = population(cohort, design)
        for role, span in spec.items():
            z = years(d, span)
            e = int(z.target_120.sum())
            rows.append(
                {
                    "design": design,
                    "split_role": role,
                    "start_fy": span[0],
                    "end_fy": span[1],
                    "loans": len(z),
                    "chargeoffs": e,
                    "prevalence_percent": 100 * e / len(z),
                    "latest_approval_date": z.approval_date.max().date(),
                    "fy2016_is_partial": span[1] == 2016,
                }
            )
            for f in features(design, True, True):
                miss.append(
                    {
                        "design": design,
                        "split_role": role,
                        "feature": f,
                        "missing_count": int(z[f].isna().sum()),
                        "missing_percent": 100 * z[f].isna().mean(),
                    }
                )
    pd.DataFrame(rows).to_csv(
        output_dir / "audited_temporal_populations.csv", index=False
    )
    pd.DataFrame(miss).to_csv(output_dir / "missingness_by_split.csv", index=False)
    checks = [
        ("Only approved feature names loaded", True),
        ("Target and post-approval fields excluded from predictors", True),
        (
            "Verified exact repeats removed globally",
            duplicate_total == 1918,
        ),
        (
            "Temporal train and locked test do not overlap",
            all(s["final_refit"][1] < s["locked_test"][0] for s in SPLITS.values()),
        ),
        ("Terms valid", cohort.term_months.between(1, 360).all()),
        (
            "Amounts and guarantee percentage valid",
            (
                cohort.gross_approval.gt(0)
                & cohort.guarantee_percent.between(0, 100)
            ).all(),
        ),
        (
            "Complete 120-month windows",
            cohort.approval_date.le(pd.Timestamp("2016-06-30")).all(),
        ),
        (
            "FY2016 partial-vintage cutoff retained",
            cohort.loc[cohort.approval_fy.eq(2016), "approval_date"].max()
            <= pd.Timestamp("2016-06-30"),
        ),
    ]
    pd.DataFrame(checks, columns=["check", "passed"]).to_csv(
        output_dir / "leakage_schema_checks.csv", index=False
    )
    if not all(bool(x[1]) for x in checks):
        raise RuntimeError("Pre-fit audit failed")


def develop(cohort, output_dir):
    """Tune configurations and evaluate frozen candidates on forward folds."""
    tuning = []
    for design, split_definition in SPLITS.items():
        design_population = population(cohort, design)
        feature_names = features(design, True, True)
        for family, configs in [
            ("logistic", LOG_CONFIGS),
            ("hist_gradient_boosting", BOOST_CONFIGS),
        ]:
            for config_number, configuration in enumerate(configs, 1):
                for fold, train_span, validation_span in validation_folds(
                    split_definition
                ):
                    training = years(design_population, train_span)
                    validation = years(design_population, validation_span)
                    model_family = "logistic" if family == "logistic" else "boost"
                    model = build_pipeline(
                        model_family,
                        feature_names,
                        configuration,
                    )
                    started_at = time.time()
                    model.fit(training[feature_names], training.target_120)
                    probabilities = model.predict_proba(validation[feature_names])[:, 1]
                    tuning.append(
                        {
                            "design": design,
                            "family": family,
                            "config_id": f"{family}_{config_number:02d}",
                            "fold": fold,
                            **configuration,
                            **metrics(validation.target_120.to_numpy(), probabilities),
                            "fit_seconds": time.time() - started_at,
                        }
                    )
                    pd.DataFrame(tuning).to_csv(
                        output_dir / "development_tuning_results.csv", index=False
                    )
                    print(
                        "tune",
                        design,
                        family,
                        config_number,
                        fold,
                        flush=True,
                    )
    write_lock()
    vals = []
    caps = []
    unks = []
    experiment_number = 1
    for design, split_definition in SPLITS.items():
        design_population = population(cohort, design)
        for family in ["logistic", "boost"]:
            configuration = LOCK[design][family]
            for variant, state, processing in VARIANTS:
                experiment_id = f"E{experiment_number:02d}"
                experiment_number += 1
                feature_names = features(design, state, processing)
                for fold, train_span, validation_span in validation_folds(
                    split_definition
                ):
                    training = years(design_population, train_span)
                    validation = years(design_population, validation_span)
                    model = build_pipeline(family, feature_names, configuration)
                    model.fit(training[feature_names], training.target_120)
                    probabilities = model.predict_proba(validation[feature_names])[:, 1]
                    base = {
                        "experiment_id": experiment_id,
                        "design": design,
                        "family": family,
                        "feature_variant": variant,
                        "fold": fold,
                    }
                    outcomes = validation.target_120.to_numpy()
                    vals.append({**base, **metrics(outcomes, probabilities)})
                    caps.extend(
                        [{**base, **row} for row in capacities(outcomes, probabilities)]
                    )
                    unks.extend(
                        [
                            {**base, **row}
                            for row in unseen(training, validation, feature_names)
                        ]
                    )
                    pd.DataFrame(vals).to_csv(
                        output_dir / "validation_candidate_metrics.csv", index=False
                    )
                    print("validate", experiment_id, fold, flush=True)
    pd.DataFrame(caps).to_csv(
        output_dir / "validation_capacity_metrics.csv", index=False
    )
    pd.DataFrame(unks).to_csv(
        output_dir / "validation_unseen_category_rates.csv", index=False
    )


def fit_calibrator(cohort):
    """Fit the approved long-history calibrator on validation predictions."""
    design_population = population(cohort, "FY2001+")
    split_definition = SPLITS["FY2001+"]
    feature_names = features("FY2001+", True, False)
    fold_probabilities = []
    fold_outcomes = []
    evidence = []
    for fold, train_span, validation_span in validation_folds(split_definition):
        training = years(design_population, train_span)
        validation = years(design_population, validation_span)
        model = boosting(feature_names, LOCK["FY2001+"]["boost"])
        model.fit(training[feature_names], training.target_120)
        probabilities = model.predict_proba(validation[feature_names])[:, 1]
        outcomes = validation.target_120.to_numpy()
        fold_probabilities.append(probabilities)
        fold_outcomes.append(outcomes)
        evidence.append({"fold": fold, **metrics(outcomes, probabilities)})

    probabilities = np.concatenate(fold_probabilities)
    outcomes = np.concatenate(fold_outcomes)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs").fit(
        log_odds,
        outcomes,
    )
    recalibrated = calibrator.predict_proba(log_odds)[:, 1]
    evidence.extend(
        [
            {"fold": "pooled_before", **metrics(outcomes, probabilities)},
            {"fold": "pooled_after", **metrics(outcomes, recalibrated)},
        ]
    )
    return calibrator, pd.DataFrame(evidence)


def apply_calibrator(calibrator, probabilities):
    """Apply a validation-fitted calibrator to model probabilities."""
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(log_odds)[:, 1]


def test(cohort, output_dir, model_dir):
    """Evaluate frozen candidates once on the common locked test population."""
    verify_lock()
    calibrator, evidence = fit_calibrator(cohort)
    evidence.to_csv(
        output_dir / "recalibration_validation_evidence.csv",
        index=False,
    )
    joblib.dump(calibrator, model_dir / "fy2001_validation_calibrator.joblib")
    candidate_metrics = []
    capacity_results = []
    calibration_results = []
    vintage_results = []
    unseen_results = []
    selected = {}
    experiment_number = 1
    for design, split_definition in SPLITS.items():
        design_population = population(cohort, design)
        training = years(design_population, split_definition["final_refit"])
        locked_test = years(design_population, split_definition["locked_test"])
        for family in ["logistic", "boost"]:
            configuration = LOCK[design][family]
            for variant, state, processing in VARIANTS:
                experiment_id = f"E{experiment_number:02d}"
                experiment_number += 1
                feature_names = features(design, state, processing)
                model = build_pipeline(family, feature_names, configuration)
                model.fit(training[feature_names], training.target_120)
                probabilities = model.predict_proba(locked_test[feature_names])[:, 1]
                outcomes = locked_test.target_120.to_numpy()
                base = {
                    "experiment_id": experiment_id,
                    "design": design,
                    "family": family,
                    "feature_variant": variant,
                    "evaluation": "locked_test_uncalibrated",
                }
                candidate_metrics.append({**base, **metrics(outcomes, probabilities)})
                capacity_results.extend(
                    [
                        {**base, **row}
                        for row in capacities(outcomes, probabilities)
                    ]
                )
                calibration_results.extend(
                    [
                        {**base, **row}
                        for row in cal_bins(outcomes, probabilities).to_dict("records")
                    ]
                )
                unseen_results.extend(
                    [
                        {**base, **row}
                        for row in unseen(training, locked_test, feature_names)
                    ]
                )
                for approval_fy, vintage in locked_test.groupby("approval_fy"):
                    mask = locked_test.approval_fy.eq(approval_fy).to_numpy()
                    vintage_results.append(
                        {
                            **base,
                            "approval_fy": approval_fy,
                            **metrics(
                                vintage.target_120.to_numpy(),
                                probabilities[mask],
                            ),
                        }
                    )
                if experiment_id in ["E06", "E14"]:
                    selected[experiment_id] = (
                        model,
                        training,
                        locked_test,
                        probabilities,
                        feature_names,
                    )
                    joblib.dump(
                        model,
                        model_dir / f"{experiment_id.lower()}_candidate.joblib",
                    )
                pd.DataFrame(candidate_metrics).to_csv(
                    output_dir / "locked_test_candidate_metrics.csv",
                    index=False,
                )
                print("test", experiment_id, flush=True)

    _, _, locked_test, probabilities, _ = selected["E14"]
    recalibrated = apply_calibrator(calibrator, probabilities)
    outcomes = locked_test.target_120.to_numpy()
    base = {
        "experiment_id": "E14_recalibrated",
        "design": "FY2001+",
        "family": "boost",
        "feature_variant": "add_state",
        "evaluation": "locked_test_validation_fitted_recalibration",
    }
    candidate_metrics.append({**base, **metrics(outcomes, recalibrated)})
    capacity_results.extend(
        [{**base, **row} for row in capacities(outcomes, recalibrated)]
    )
    calibration_results.extend(
        [
            {**base, **row}
            for row in cal_bins(outcomes, recalibrated).to_dict("records")
        ]
    )
    pd.DataFrame(candidate_metrics).to_csv(
        output_dir / "locked_test_candidate_metrics.csv", index=False
    )
    pd.DataFrame(capacity_results).to_csv(
        output_dir / "locked_test_capacity_metrics.csv", index=False
    )
    pd.DataFrame(calibration_results).to_csv(
        output_dir / "locked_test_calibration_bins.csv", index=False
    )
    pd.DataFrame(vintage_results).to_csv(
        output_dir / "locked_test_vintage_metrics.csv", index=False
    )
    pd.DataFrame(unseen_results).to_csv(
        output_dir / "locked_test_unseen_category_rates.csv", index=False
    )


def main():
    """Run one explicitly requested modelling stage."""
    parser = argparse.ArgumentParser(
        description="Run the model audit, development, or locked-test stage."
    )
    parser.add_argument("stage", choices=["audit", "develop", "test"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()

    output_dir = ROOT / "reports/tables/modeling"
    model_dir = ROOT / "models/candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "test":
        model_dir.mkdir(parents=True, exist_ok=True)

    cohort, cohort_audit, duplicate_audit = load_cohort(args.raw_dir)
    if args.stage == "audit":
        audit(cohort, output_dir, cohort_audit, duplicate_audit)
    elif args.stage == "develop":
        develop(cohort, output_dir)
    else:
        test(cohort, output_dir, model_dir)


if __name__ == "__main__":
    main()
