"""Build aggregate portfolio-risk evidence from the official SBA 7(a) files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_cohort import (
    EXPECTED_FILES,
    SNAPSHOT,
    business_age_group,
    clean_text,
    loan_size_band,
    status_value,
    term_band,
    verify_duplicate_rows,
    wilson_interval,
)

USECOLS = (
    "AsOfDate",
    "GrossApproval",
    "SBAGuaranteedApproval",
    "ApprovalDate",
    "ApprovalFY",
    "ProcessingMethod",
    "InitialInterestRate",
    "TermInMonths",
    "NaicsCode",
    "ProjectState",
    "BusinessType",
    "BusinessAge",
    "LoanStatus",
    "ChargeOffDate",
    "GrossChargeOffAmount",
)

CANDIDATE_STARTS = (2001, 2010)
HORIZONS = (96, 120)
MIN_LOANS = 500
MIN_EVENTS = 25

def guarantee_band(gross: pd.Series, guaranteed: pd.Series) -> pd.Series:
    pct = (
        100
        * pd.to_numeric(guaranteed, errors="coerce")
        / pd.to_numeric(gross, errors="coerce")
    )
    pct = pct.where(pd.to_numeric(gross, errors="coerce").gt(0) & pct.between(0, 100))
    result = pd.cut(
        pct,
        bins=[-0.001, 0, 50, 75, 85, 90, 100],
        labels=[
            "0%",
            ">0% to 50%",
            ">50% to 75%",
            ">75% to 85%",
            ">85% to 90%",
            ">90% to 100%",
        ],
        include_lowest=True,
    ).astype("string")
    return result.fillna("Invalid or missing")


def interest_band(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.cut(
        values,
        bins=[0, 5, 7.5, 10, 12.5, 15, np.inf],
        labels=[
            "Up to 5%",
            ">5% to 7.5%",
            ">7.5% to 10%",
            ">10% to 12.5%",
            ">12.5% to 15%",
            "More than 15%",
        ],
        include_lowest=True,
    ).astype("string")
    return result.mask(values.lt(0) | values.isna(), "Invalid or missing").fillna(
        "Invalid or missing"
    )


def add_aggregate(
    store: defaultdict,
    key: tuple,
    mask: pd.Series,
    event: pd.Series,
    gross: pd.Series,
    guaranteed: pd.Series,
    chargeoff_amount: pd.Series,
    flagged_financial: pd.Series,
) -> None:
    values = store[key]
    values["eligible_loans"] += int(mask.sum())
    values["chargeoffs"] += int((mask & event).sum())
    valid_gross = mask & gross.gt(0)
    valid_guarantee = mask & guaranteed.ge(0) & guaranteed.le(gross)
    event_mask = mask & event
    values["valid_gross_amount_loans"] += int(valid_gross.sum())
    values["gross_approval"] += float(gross[valid_gross].sum())
    values["sba_guaranteed_approval"] += float(guaranteed[valid_guarantee].sum())
    values["chargeoff_loan_approval"] += float(gross[event_mask & gross.gt(0)].sum())
    values["gross_chargeoff_amount"] += float(chargeoff_amount[event_mask].sum())
    values["flagged_financial_chargeoffs"] += int(
        (event_mask & flagged_financial).sum()
    )
    sensitivity_mask = mask & ~(event & flagged_financial)
    sensitivity_event = event_mask & ~flagged_financial
    values["sensitivity_gross_approval"] += float(
        gross[sensitivity_mask & gross.gt(0)].sum()
    )
    values["sensitivity_chargeoff_loan_approval"] += float(
        gross[sensitivity_event & gross.gt(0)].sum()
    )
    values["sensitivity_gross_chargeoff_amount"] += float(
        chargeoff_amount[sensitivity_event].sum()
    )


def aggregate_frame(store: dict, key_names: list[str]) -> pd.DataFrame:
    rows = []
    for key, values in store.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(key_names, key_tuple))
        row.update(values)
        total = int(values["eligible_loans"])
        events = int(values["chargeoffs"])
        low, high = wilson_interval(events, total)
        row["chargeoff_rate_percent"] = 100 * events / total if total else np.nan
        row["ci95_low_percent"] = 100 * low if low is not None else np.nan
        row["ci95_high_percent"] = 100 * high if high is not None else np.nan
        row["passes_rate_reporting_rule"] = bool(
            total >= MIN_LOANS and events >= MIN_EVENTS
        )
        row["approval_weighted_chargeoff_incidence_percent"] = (
            100 * values["chargeoff_loan_approval"] / values["gross_approval"]
            if values["gross_approval"]
            else np.nan
        )
        row["gross_chargeoff_rate_percent"] = (
            100 * values["gross_chargeoff_amount"] / values["gross_approval"]
            if values["gross_approval"]
            else np.nan
        )
        row["chargeoff_severity_percent"] = (
            100 * values["gross_chargeoff_amount"] / values["chargeoff_loan_approval"]
            if values["chargeoff_loan_approval"]
            else np.nan
        )
        row["sensitivity_gross_chargeoff_rate_percent"] = (
            100
            * values["sensitivity_gross_chargeoff_amount"]
            / values["sensitivity_gross_approval"]
            if values["sensitivity_gross_approval"]
            else np.nan
        )
        row["sensitivity_chargeoff_severity_percent"] = (
            100
            * values["sensitivity_gross_chargeoff_amount"]
            / values["sensitivity_chargeoff_loan_approval"]
            if values["sensitivity_chargeoff_loan_approval"]
            else np.nan
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include="number").columns
    frame[numeric] = frame[numeric].round(6)
    return frame


def save_figures(
    out: Path,
    historical: pd.DataFrame,
    vintage: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    history = historical.sort_values("approval_fy")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(
        history["approval_fy"], history["loan_count"], color="#24557a", linewidth=2
    )
    axes[0].set_ylabel("Approved loans")
    axes[0].set_title(
        "7(a) approval volume changed substantially across fiscal years",
        loc="left",
        weight="bold",
    )
    axes[1].plot(
        history["approval_fy"],
        history["gross_approval"] / 1e9,
        color="#b45f35",
        linewidth=2,
    )
    axes[1].set_ylabel("Gross approval ($bn)")
    axes[1].set_xlabel("Approval fiscal year")
    axes[1].text(
        0,
        -0.25,
        "Dollar amounts are nominal and are not adjusted for inflation.",
        transform=axes[1].transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "01_historical_approval_volume.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    primary = vintage[
        (vintage["candidate_start_fy"] == 2001) & (vintage["horizon_months"] == 120)
    ].sort_values("approval_fy")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        primary["approval_fy"],
        primary["chargeoff_rate_percent"],
        marker="o",
        color="#24557a",
        linewidth=2,
    )
    ax.fill_between(
        primary["approval_fy"],
        primary["ci95_low_percent"],
        primary["ci95_high_percent"],
        color="#24557a",
        alpha=0.15,
    )
    ax.set_title(
        "Ten-year charge-off risk varies sharply by approval vintage",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("Approval fiscal year")
    ax.set_ylabel("Charge-off within 120 months (%)")
    ax.text(
        0,
        -0.18,
        "Only vintages with a complete 120-month observation window are included.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out / "02_vintage_chargeoff_risk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for number, dimension, filename, title in [
        (
            3,
            "loan_size_band",
            "03_loan_size_risk.png",
            "Loan size separates portfolio exposure from observed risk",
        ),
        (
            4,
            "term_band",
            "04_term_risk.png",
            "Term bands show different ten-year risk patterns",
        ),
        (
            5,
            "business_age",
            "05_business_age_risk.png",
            "Business age groups differ in observed ten-year risk",
        ),
    ]:
        data = segments[
            (segments["candidate_start_fy"] == 2010)
            & (segments["horizon_months"] == 120)
            & (segments["dimension"] == dimension)
            & segments["passes_rate_reporting_rule"]
        ].copy()
        data = data.sort_values("chargeoff_rate_percent")
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(data) * 0.48)))
        ax.barh(data["segment"], data["chargeoff_rate_percent"], color="#487a9b")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Charge-off within 120 months (%)")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    naics = segments[
        (segments["candidate_start_fy"] == 2010)
        & (segments["horizon_months"] == 120)
        & (segments["dimension"] == "naics2")
        & segments["passes_rate_reporting_rule"]
    ].copy()
    total_approval = naics["gross_approval"].sum()
    naics["approval_share_percent"] = 100 * naics["gross_approval"] / total_approval
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(
        naics["approval_share_percent"],
        naics["chargeoff_rate_percent"],
        s=np.sqrt(naics["eligible_loans"]) * 5,
        alpha=0.7,
        color="#24557a",
    )
    for _, row in naics.nlargest(8, "gross_approval").iterrows():
        ax.annotate(
            str(row["segment"]),
            (row["approval_share_percent"], row["chargeoff_rate_percent"]),
            fontsize=8,
        )
    ax.set_title(
        "Industry monitoring must consider both exposure and risk",
        loc="left",
        weight="bold",
    )
    ax.set_xlabel("Share of gross approvals in candidate cohort (%)")
    ax.set_ylabel("Charge-off within 120 months (%)")
    fig.tight_layout()
    fig.savefig(out / "06_industry_exposure_and_risk.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/tables/portfolio")
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("reports/figures/portfolio")
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repeated_positions, _ = verify_duplicate_rows(args.raw_dir, args.chunk_size)
    historical = defaultdict(Counter)
    overall = defaultdict(Counter)
    vintage = defaultdict(Counter)
    segment = defaultdict(Counter)
    common_horizon = defaultdict(Counter)

    for filename in EXPECTED_FILES:
        offset = 0
        for frame in pd.read_csv(
            args.raw_dir / filename,
            usecols=list(USECOLS),
            chunksize=args.chunk_size,
            low_memory=False,
        ):
            row_numbers = np.arange(offset, offset + len(frame))
            keep = ~np.isin(row_numbers, list(repeated_positions[filename]))
            offset += len(frame)
            frame = frame.loc[keep].copy()

            approval_fy = pd.to_numeric(frame["ApprovalFY"], errors="coerce").astype(
                "Int64"
            )
            approval_date = pd.to_datetime(frame["ApprovalDate"], errors="coerce")
            chargeoff_date = pd.to_datetime(frame["ChargeOffDate"], errors="coerce")
            asof_date = pd.to_datetime(frame["AsOfDate"], errors="coerce")
            status = status_value(frame["LoanStatus"])
            gross = pd.to_numeric(frame["GrossApproval"], errors="coerce")
            guaranteed = pd.to_numeric(frame["SBAGuaranteedApproval"], errors="coerce")
            chargeoff_amount = pd.to_numeric(
                frame["GrossChargeOffAmount"], errors="coerce"
            )
            term = pd.to_numeric(frame["TermInMonths"], errors="coerce")
            interest = pd.to_numeric(frame["InitialInterestRate"], errors="coerce")

            valid_history = approval_fy.between(1991, 2026)
            for fy in approval_fy[valid_history].dropna().unique():
                mask = valid_history & approval_fy.eq(fy)
                h = historical[int(fy)]
                h["loan_count"] += int(mask.sum())
                h["valid_gross_amount_loans"] += int((mask & gross.gt(0)).sum())
                h["gross_approval"] += float(gross[mask & gross.gt(0)].sum())

            valid_chargeoff_date = (
                status.eq("CHGOFF")
                & chargeoff_date.notna()
                & approval_date.notna()
                & chargeoff_date.ge(approval_date)
                & chargeoff_date.le(asof_date)
            )
            post_snapshot = status.eq("CHGOFF") & chargeoff_date.gt(asof_date)
            flagged_financial = (
                chargeoff_amount.gt(gross)
                | chargeoff_amount.fillna(0).le(0)
                | gross.le(0)
                | gross.isna()
            )

            naics_numeric = pd.to_numeric(frame["NaicsCode"], errors="coerce").astype(
                "Int64"
            )
            naics2 = naics_numeric.astype("string").str.zfill(6).str[:2]
            naics2 = naics2.where(naics_numeric.notna(), "Missing")
            state = (
                clean_text(frame["ProjectState"])
                .replace("", "Missing")
                .fillna("Missing")
            )
            processing = (
                clean_text(frame["ProcessingMethod"])
                .replace("", "Missing")
                .fillna("Missing")
            )
            business_type = (
                clean_text(frame["BusinessType"])
                .replace("", "Missing")
                .fillna("Missing")
            )
            age = business_age_group(frame["BusinessAge"]).replace(
                "Unknown or unmapped", "Unknown"
            )

            dimensions = {
                "loan_size_band": loan_size_band(gross),
                "term_band": term_band(term),
                "guarantee_percent_band": guarantee_band(gross, guaranteed),
                "naics2": naics2,
                "project_state": state,
                "business_age": age,
                "business_type": business_type,
                "processing_method": processing,
                "interest_rate_band": interest_band(interest),
            }

            for horizon in HORIZONS:
                cutoff = SNAPSHOT - pd.DateOffset(months=horizon)
                base = (
                    approval_date.notna()
                    & approval_date.le(cutoff)
                    & ~post_snapshot
                    & status.isin(["CHGOFF", "PIF", "EXEMPT"])
                )
                invalid_event_timing = (
                    base & status.eq("CHGOFF") & ~valid_chargeoff_date
                )
                eligible = base & ~invalid_event_timing
                event = (
                    eligible
                    & status.eq("CHGOFF")
                    & chargeoff_date.le(approval_date + pd.DateOffset(months=horizon))
                )

                for start in CANDIDATE_STARTS:
                    population = eligible & approval_fy.ge(start)
                    add_aggregate(
                        overall,
                        (start, horizon),
                        population,
                        event,
                        gross,
                        guaranteed,
                        chargeoff_amount,
                        flagged_financial,
                    )

                    for fy in approval_fy[population].dropna().unique():
                        mask = population & approval_fy.eq(fy)
                        add_aggregate(
                            vintage,
                            (start, horizon, int(fy)),
                            mask,
                            event,
                            gross,
                            guaranteed,
                            chargeoff_amount,
                            flagged_financial,
                        )

                    for dim, values in dimensions.items():
                        if dim == "interest_rate_band" and start != 2010:
                            continue
                        for value in values[population].dropna().unique():
                            mask = population & values.eq(value)
                            add_aggregate(
                                segment,
                                (start, horizon, dim, str(value)),
                                mask,
                                event,
                                gross,
                                guaranteed,
                                chargeoff_amount,
                                flagged_financial,
                            )

            # Compare 96 and 120 months on the same loans. This separates the
            # horizon effect from the extra FY2017 and FY2018 vintages admitted
            # by the ordinary 96-month design.
            cutoff_120 = SNAPSHOT - pd.DateOffset(months=120)
            base_common = (
                approval_date.notna()
                & approval_date.le(cutoff_120)
                & ~post_snapshot
                & status.isin(["CHGOFF", "PIF", "EXEMPT"])
            )
            invalid_common_timing = (
                base_common & status.eq("CHGOFF") & ~valid_chargeoff_date
            )
            eligible_common = base_common & ~invalid_common_timing
            event_96 = (
                eligible_common
                & status.eq("CHGOFF")
                & chargeoff_date.le(approval_date + pd.DateOffset(months=96))
            )
            event_120 = (
                eligible_common
                & status.eq("CHGOFF")
                & chargeoff_date.le(approval_date + pd.DateOffset(months=120))
            )
            common_dimensions = {
                "overall": pd.Series("All eligible loans", index=frame.index),
                "term_band": dimensions["term_band"],
            }
            for start in CANDIDATE_STARTS:
                population = eligible_common & approval_fy.ge(start)
                for dim, values in common_dimensions.items():
                    for value in values[population].dropna().unique():
                        mask = population & values.eq(value)
                        row = common_horizon[(start, dim, str(value))]
                        row["eligible_loans"] += int(mask.sum())
                        row["chargeoffs_within_96_months"] += int(
                            (mask & event_96).sum()
                        )
                        row["chargeoffs_within_120_months"] += int(
                            (mask & event_120).sum()
                        )

    historical_df = pd.DataFrame(
        [{"approval_fy": fy, **values} for fy, values in historical.items()]
    ).sort_values("approval_fy")
    overall_df = aggregate_frame(overall, ["candidate_start_fy", "horizon_months"])
    vintage_df = aggregate_frame(
        vintage, ["candidate_start_fy", "horizon_months", "approval_fy"]
    )
    segment_df = aggregate_frame(
        segment, ["candidate_start_fy", "horizon_months", "dimension", "segment"]
    )
    common_horizon_df = pd.DataFrame(
        [
            {
                "candidate_start_fy": key[0],
                "dimension": key[1],
                "segment": key[2],
                **values,
            }
            for key, values in common_horizon.items()
        ]
    )
    common_horizon_df["risk_96_month_percent"] = (
        100
        * common_horizon_df["chargeoffs_within_96_months"]
        / common_horizon_df["eligible_loans"]
    )
    common_horizon_df["risk_120_month_percent"] = (
        100
        * common_horizon_df["chargeoffs_within_120_months"]
        / common_horizon_df["eligible_loans"]
    )
    common_horizon_df["additional_risk_96_to_120_pp"] = (
        common_horizon_df["risk_120_month_percent"]
        - common_horizon_df["risk_96_month_percent"]
    )
    common_horizon_df["share_120_month_events_captured_by_96_percent"] = (
        100
        * common_horizon_df["chargeoffs_within_96_months"]
        / common_horizon_df["chargeoffs_within_120_months"]
    )
    common_horizon_df = common_horizon_df.round(6)

    primary_segments = segment_df[
        (segment_df["horizon_months"] == 120) & segment_df["passes_rate_reporting_rule"]
    ].copy()
    totals = overall_df[overall_df["horizon_months"] == 120][
        [
            "candidate_start_fy",
            "gross_approval",
            "chargeoff_loan_approval",
            "gross_chargeoff_amount",
            "sensitivity_gross_chargeoff_amount",
        ]
    ].rename(
        columns={
            c: f"total_{c}"
            for c in [
                "gross_approval",
                "chargeoff_loan_approval",
                "gross_chargeoff_amount",
                "sensitivity_gross_chargeoff_amount",
            ]
        }
    )
    concentration_df = primary_segments.merge(
        totals, on="candidate_start_fy", how="left"
    )
    concentration_df["gross_approval_share_percent"] = (
        100
        * concentration_df["gross_approval"]
        / concentration_df["total_gross_approval"]
    )
    concentration_df["chargeoff_loan_approval_share_percent"] = (
        100
        * concentration_df["chargeoff_loan_approval"]
        / concentration_df["total_chargeoff_loan_approval"]
    )
    concentration_df["gross_chargeoff_amount_share_percent"] = (
        100
        * concentration_df["gross_chargeoff_amount"]
        / concentration_df["total_gross_chargeoff_amount"]
    )
    concentration_df["sensitivity_gross_chargeoff_amount_share_percent"] = (
        100
        * concentration_df["sensitivity_gross_chargeoff_amount"]
        / concentration_df["total_sensitivity_gross_chargeoff_amount"]
    )
    concentration_df["financial_anomaly_share_shift_pp"] = (
        concentration_df["gross_chargeoff_amount_share_percent"]
        - concentration_df["sensitivity_gross_chargeoff_amount_share_percent"]
    )

    outputs = {
        "historical_approval_trend.csv": historical_df,
        "candidate_population_summary.csv": overall_df,
        "vintage_risk.csv": vintage_df,
        "segment_risk.csv": segment_df,
        "financial_concentration.csv": concentration_df,
        "common_cohort_horizon_sensitivity.csv": common_horizon_df,
    }
    for name, data in outputs.items():
        data.to_csv(args.output_dir / name, index=False)

    save_figures(args.figure_dir, historical_df, vintage_df, segment_df)
    print(f"Wrote {len(outputs)} aggregate tables and portfolio figures.")


if __name__ == "__main__":
    main()
