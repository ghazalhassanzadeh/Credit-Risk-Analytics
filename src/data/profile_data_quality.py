"""Create the data-quality checks retained in the public project."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.verify_schema import EXPECTED_FILES

PERIOD_LABELS = {
    "FOIA_7a_FY1991_FY1999_asof_260630.csv": "FY1991-FY1999",
    "FOIA_7a_FY2000_FY2009_asof_260630.csv": "FY2000-FY2009",
    "FOIA_7a_FY2010_FY2019_asof_260630.csv": "FY2010-FY2019",
    "FOIA_7a_FY2020_Present_asof_260630.csv": "FY2020-FY2026 YTD",
}

DATE_FIELDS = (
    "AsOfDate",
    "ApprovalDate",
    "FirstDisbursementDate",
    "PaidInFullDate",
    "ChargeOffDate",
)

MODEL_FIELDS = (
    "GrossApproval",
    "SBAGuaranteedApproval",
    "ApprovalDate",
    "ProcessingMethod",
    "InitialInterestRate",
    "FixedorVariableInterestInd",
    "TermInMonths",
    "NaicsCode",
    "ProjectState",
    "BusinessType",
    "BusinessAge",
    "RevolverStatus",
    "JobsSupported",
    "CollateralInd",
    "SoldSecMrktInd",
)

VALID_NAICS_SECTORS = {
    "11", "21", "22", "23", "31", "32", "33", "42", "44", "45",
    "48", "49", "51", "52", "53", "54", "55", "56", "61", "62",
    "71", "72", "81", "92",
}


def fiscal_year(dates: pd.Series) -> pd.Series:
    """Convert calendar dates to the federal fiscal year."""
    return dates.dt.year + (dates.dt.month >= 10).astype("Int64")


def count_issue(
    issue_counts: defaultdict,
    period: str,
    issue: str,
    condition: pd.Series,
) -> None:
    """Add the number of records meeting one quality condition."""
    issue_counts[(period, issue)] += int(condition.fillna(False).sum())


def update_completeness(
    store: defaultdict,
    period: str,
    frame: pd.DataFrame,
) -> None:
    """Accumulate field completeness for one publication period."""
    for field in frame.columns:
        store[(period, field)]["rows"] += len(frame)
        store[(period, field)]["missing"] += int(frame[field].isna().sum())


def update_model_availability(
    store: defaultdict,
    frame: pd.DataFrame,
) -> None:
    """Accumulate candidate-feature availability by approval vintage."""
    approval_fy = pd.to_numeric(frame["ApprovalFY"], errors="coerce").astype("Int64")
    for year, row_index in approval_fy.groupby(approval_fy, dropna=False).groups.items():
        year_label = "MISSING" if pd.isna(year) else str(int(year))
        group = frame.loc[row_index]
        for field in MODEL_FIELDS:
            store[(year_label, field)]["rows"] += len(group)
            store[(year_label, field)]["missing"] += int(group[field].isna().sum())


def evaluate_quality_rules(
    issue_counts: defaultdict,
    period: str,
    frame: pd.DataFrame,
) -> None:
    """Evaluate the source checks that influenced cleaning and feature selection."""
    dates = {
        field: pd.to_datetime(frame[field], errors="coerce")
        for field in DATE_FIELDS
    }
    for field in DATE_FIELDS:
        count_issue(
            issue_counts,
            period,
            f"{field}: non-null value did not parse",
            frame[field].notna() & dates[field].isna(),
        )

    approval_fy = pd.to_numeric(frame["ApprovalFY"], errors="coerce").astype("Int64")
    status = frame["LoanStatus"].astype("string").str.strip()
    gross = pd.to_numeric(frame["GrossApproval"], errors="coerce")
    guarantee = pd.to_numeric(frame["SBAGuaranteedApproval"], errors="coerce")
    chargeoff = pd.to_numeric(frame["GrossChargeOffAmount"], errors="coerce")
    term = pd.to_numeric(frame["TermInMonths"], errors="coerce")
    rate = pd.to_numeric(frame["InitialInterestRate"], errors="coerce")
    jobs = pd.to_numeric(frame["JobsSupported"], errors="coerce")
    naics = pd.to_numeric(frame["NaicsCode"], errors="coerce")
    naics2 = naics.astype("Int64").astype("string").str.zfill(6).str[:2]

    rules = [
        ("ApprovalDate after AsOfDate", dates["ApprovalDate"] > dates["AsOfDate"]),
        (
            "ApprovalFY inconsistent with ApprovalDate fiscal year",
            approval_fy.notna()
            & dates["ApprovalDate"].notna()
            & approval_fy.ne(fiscal_year(dates["ApprovalDate"])),
        ),
        (
            "FirstDisbursementDate before ApprovalDate",
            dates["FirstDisbursementDate"] < dates["ApprovalDate"],
        ),
        (
            "FirstDisbursementDate after AsOfDate",
            dates["FirstDisbursementDate"] > dates["AsOfDate"],
        ),
        (
            "PaidInFullDate before ApprovalDate",
            dates["PaidInFullDate"] < dates["ApprovalDate"],
        ),
        (
            "PaidInFullDate after AsOfDate",
            dates["PaidInFullDate"] > dates["AsOfDate"],
        ),
        (
            "ChargeOffDate before ApprovalDate",
            dates["ChargeOffDate"] < dates["ApprovalDate"],
        ),
        (
            "ChargeOffDate after AsOfDate",
            dates["ChargeOffDate"] > dates["AsOfDate"],
        ),
        (
            "Both PaidInFullDate and ChargeOffDate present",
            dates["PaidInFullDate"].notna() & dates["ChargeOffDate"].notna(),
        ),
        (
            "P I F status missing PaidInFullDate",
            status.eq("P I F") & dates["PaidInFullDate"].isna(),
        ),
        (
            "CHGOFF status missing ChargeOffDate",
            status.eq("CHGOFF") & dates["ChargeOffDate"].isna(),
        ),
        (
            "Non-CHGOFF status with ChargeOffDate",
            status.ne("CHGOFF") & dates["ChargeOffDate"].notna(),
        ),
        (
            "Non-P I F status with PaidInFullDate",
            status.ne("P I F") & dates["PaidInFullDate"].notna(),
        ),
        (
            "NAICS code has an unrecognised 2-digit sector",
            naics.notna() & ~naics2.isin(VALID_NAICS_SECTORS),
        ),
        ("GrossApproval less than or equal to zero", gross.le(0)),
        ("SBAGuaranteedApproval less than zero", guarantee.lt(0)),
        ("SBAGuaranteedApproval exceeds GrossApproval", guarantee.gt(gross)),
        ("GrossChargeOffAmount less than zero", chargeoff.lt(0)),
        ("GrossChargeOffAmount exceeds GrossApproval", chargeoff.gt(gross)),
        (
            "GrossChargeOffAmount exceeds 150 percent of GrossApproval",
            chargeoff.gt(1.5 * gross),
        ),
        (
            "Non-CHGOFF status with positive GrossChargeOffAmount",
            status.ne("CHGOFF") & chargeoff.gt(0),
        ),
        (
            "CHGOFF status with zero or missing GrossChargeOffAmount",
            status.eq("CHGOFF") & chargeoff.fillna(0).le(0),
        ),
        ("TermInMonths less than or equal to zero", term.le(0)),
        ("TermInMonths is fractional", term.notna() & term.mod(1).ne(0)),
        ("TermInMonths exceeds 360", term.gt(360)),
        ("InitialInterestRate below zero", rate.lt(0)),
        ("InitialInterestRate equals zero", rate.eq(0)),
        ("InitialInterestRate above 25", rate.gt(25)),
        ("InitialInterestRate above 100", rate.gt(100)),
        ("JobsSupported below zero", jobs.lt(0)),
        ("JobsSupported is fractional", jobs.notna() & jobs.mod(1).ne(0)),
        ("JobsSupported above 500", jobs.gt(500)),
        ("JobsSupported above 1000", jobs.gt(1000)),
    ]

    for issue, condition in rules:
        count_issue(issue_counts, period, issue, condition)


def completeness_table(store: defaultdict) -> pd.DataFrame:
    """Convert accumulated completeness counts to a report table."""
    rows = []
    for (period, field), values in sorted(store.items()):
        rows.append(
            {
                "period": period,
                "field": field,
                "row_count": values["rows"],
                "missing_count": values["missing"],
                "missing_percent": round(
                    100 * values["missing"] / values["rows"], 4
                ),
            }
        )
    return pd.DataFrame(rows)


def issue_table(issue_counts: defaultdict, period_rows: Counter) -> pd.DataFrame:
    """Convert accumulated issue counts to a report table."""
    rows = []
    for (period, issue), count in sorted(issue_counts.items()):
        rows.append(
            {
                "period": period,
                "issue": issue,
                "affected_count": count,
                "percent_of_period_rows": round(
                    100 * count / period_rows[period], 6
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_timing_table() -> pd.DataFrame:
    """Document whether published fields are suitable at the prediction point."""
    rows = [
        ("GrossApproval", "Candidate", "Approved amount known at approval"),
        ("TermInMonths", "Candidate", "Approved term known at approval"),
        ("NaicsCode", "Candidate", "Industry code available at approval"),
        ("ProjectState", "Candidate", "Project geography available at approval"),
        ("BusinessType", "Candidate", "Borrower legal form at approval"),
        ("BusinessAge", "Candidate", "Reported business-age category"),
        (
            "InitialInterestRate",
            "FY2010+ candidate",
            "Available at approval but incomplete in earlier periods",
        ),
        (
            "ProcessingMethod",
            "Sensitivity only",
            "Available at approval but not stable enough for the final model",
        ),
        (
            "SBAGuaranteedApproval",
            "Derived input only",
            "Used with GrossApproval to calculate guarantee percentage",
        ),
        ("ApprovalFY", "Split only", "Used for temporal validation, not prediction"),
        ("JobsSupported", "Exclude", "Application estimate, not a core risk field"),
        ("LocationID", "Exclude", "Lender identity is outside project scope"),
        ("FirstDisbursementDate", "Exclude", "Occurs after loan approval"),
        ("SoldSecMrktInd", "Exclude", "May be updated after approval"),
        ("LoanStatus", "Outcome", "Defines outcome or censoring state"),
        ("PaidInFullDate", "Outcome", "Observed after approval"),
        ("ChargeOffDate", "Outcome", "Observed after approval"),
        ("GrossChargeOffAmount", "Outcome", "Observed after approval"),
        ("AsOfDate", "Metadata", "Publication snapshot date"),
    ]
    return pd.DataFrame(rows, columns=["field", "assessment", "reason"])


def main() -> None:
    """Read the four source files in chunks and write four aggregate reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/tables/data_quality"),
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    field_completeness = defaultdict(lambda: {"rows": 0, "missing": 0})
    model_availability = defaultdict(lambda: {"rows": 0, "missing": 0})
    issue_counts = defaultdict(int)
    period_rows = Counter()

    for filename in EXPECTED_FILES:
        period = PERIOD_LABELS[filename]
        source_path = args.raw_dir / filename
        for frame in pd.read_csv(
            source_path,
            chunksize=args.chunk_size,
            low_memory=False,
        ):
            period_rows[period] += len(frame)
            update_completeness(field_completeness, period, frame)
            update_model_availability(model_availability, frame)
            evaluate_quality_rules(issue_counts, period, frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completeness_table(field_completeness).to_csv(
        args.output_dir / "field_completeness_by_period.csv",
        index=False,
    )
    availability = completeness_table(model_availability).rename(
        columns={"period": "approval_fy"}
    )
    availability.to_csv(
        args.output_dir / "model_field_availability_by_fy.csv",
        index=False,
    )
    issue_table(issue_counts, period_rows).to_csv(
        args.output_dir / "data_quality_issue_counts.csv",
        index=False,
    )
    feature_timing_table().to_csv(
        args.output_dir / "feature_timing_assessment.csv",
        index=False,
    )
    print("Data-quality profile completed for all four SBA 7(a) files.")


if __name__ == "__main__":
    main()
