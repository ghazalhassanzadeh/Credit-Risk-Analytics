"""Create the cleaned SBA 7(a) analysis cohort.

Raw SBA files remain unchanged. The script removes verified repeated rows,
applies the final cohort rules, and saves an analysis-ready Parquet file for
the statistical and modelling workflows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FILES = (
    "FOIA_7a_FY1991_FY1999_asof_260630.csv",
    "FOIA_7a_FY2000_FY2009_asof_260630.csv",
    "FOIA_7a_FY2010_FY2019_asof_260630.csv",
    "FOIA_7a_FY2020_Present_asof_260630.csv",
)

SNAPSHOT = pd.Timestamp("2026-06-30")

RAW_COLUMNS = (
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
)

OUTPUT_COLUMNS = (
    "approval_fy",
    "target_120",
    "target_96",
    "gross_approval",
    "term_months",
    "guarantee_percent",
    "naics2",
    "project_state",
    "business_age",
    "business_type",
    "processing_method",
    "initial_interest_rate",
)

BUSINESS_AGE_MAP = {
    "Startup, Loan Funds will Open Business": "Startup",
    "New, Less than 1 Year old": "New or early-stage",
    "New Business or 2 years or less": "New or early-stage",
    "Less than 3 years old but at least 2": "New or early-stage",
    "Less than 4 years old but at least 3": "Existing",
    "Less than 5 years old but at least 4": "Existing",
    "Existing, 5 or more years": "Existing",
    "Existing or more than 2 years old": "Existing",
    "Change of Ownership": "Change of ownership",
    "Unanswered": "Unknown",
}


def clean_text(series: pd.Series) -> pd.Series:
    """Trim text while preserving missing values."""
    return series.astype("string").str.strip()


def status_value(series: pd.Series) -> pd.Series:
    """Standardise the one known formatting variation in loan status."""
    return clean_text(series).replace({"P I F": "PIF"})


def business_age_group(series: pd.Series) -> pd.Series:
    """Map published business-age labels into stable reporting groups."""
    return clean_text(series).map(BUSINESS_AGE_MAP).fillna("Unknown or unmapped")


def loan_size_band(gross: pd.Series) -> pd.Series:
    """Create the loan-size groups used in the portfolio analysis."""
    values = pd.to_numeric(gross, errors="coerce")
    bands = pd.cut(
        values,
        bins=[0, 50_000, 150_000, 350_000, 1_000_000, np.inf],
        labels=[
            "Up to $50k",
            "$50k to $150k",
            "$150k to $350k",
            "$350k to $1m",
            "More than $1m",
        ],
        include_lowest=True,
    ).astype("string")
    return bands.mask(values.le(0) | values.isna(), "Invalid or missing").fillna(
        "Invalid or missing"
    )


def term_band(term: pd.Series) -> pd.Series:
    """Create the original-term groups used in reporting."""
    values = pd.to_numeric(term, errors="coerce")
    bands = pd.cut(
        values,
        bins=[0, 12, 60, 84, 120, 180, 240, 360],
        labels=[
            "1 to 12",
            "13 to 60",
            "61 to 84",
            "85 to 120",
            "121 to 180",
            "181 to 240",
            "241 to 360",
        ],
        include_lowest=True,
    ).astype("string")
    return bands.mask(
        values.le(0) | values.gt(360) | values.isna(), "Invalid or missing"
    ).fillna("Invalid or missing")


def wilson_interval(
    events: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    """Return a 95% Wilson interval for a proportion."""
    if total == 0:
        return None, None
    proportion = events / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return centre - half_width, centre + half_width


def verify_duplicate_rows(
    raw_dir: Path, chunk_size: int = 100_000
) -> tuple[dict[str, set[int]], list[dict]]:
    """Find exact repeated rows across all 42 published fields."""
    hashes_by_file: dict[str, np.ndarray] = {}
    all_hashes = []

    for filename in EXPECTED_FILES:
        file_hashes = []
        for chunk in pd.read_csv(
            raw_dir / filename,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
            low_memory=False,
        ):
            file_hashes.append(
                pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype="uint64")
            )
        combined_file_hashes = np.concatenate(file_hashes)
        hashes_by_file[filename] = combined_file_hashes
        all_hashes.append(combined_file_hashes)

    combined_hashes = np.concatenate(all_hashes)
    unique_hashes, counts = np.unique(combined_hashes, return_counts=True)
    duplicate_hashes = set(unique_hashes[counts > 1].tolist())

    repeated_positions = {filename: set() for filename in EXPECTED_FILES}
    seen_rows: set[tuple[str, ...]] = set()

    for filename in EXPECTED_FILES:
        offset = 0
        for chunk in pd.read_csv(
            raw_dir / filename,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
            low_memory=False,
        ):
            hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(
                dtype="uint64"
            )
            candidates = np.flatnonzero(np.isin(hashes, list(duplicate_hashes)))
            for position in candidates:
                row = tuple(chunk.iloc[position].tolist())
                row_number = offset + int(position)
                if row in seen_rows:
                    repeated_positions[filename].add(row_number)
                else:
                    seen_rows.add(row)
            offset += len(chunk)

    verification = []
    for filename in EXPECTED_FILES:
        verification.append(
            {
                "file": filename,
                "rows": int(hashes_by_file[filename].size),
                "repeated_rows_removed": len(repeated_positions[filename]),
            }
        )
    verification.append(
        {
            "file": "ALL FILES",
            "rows": int(combined_hashes.size),
            "repeated_rows_removed": sum(
                len(positions) for positions in repeated_positions.values()
            ),
        }
    )
    return repeated_positions, verification


def _category(series: pd.Series) -> pd.Series:
    """Prepare a categorical feature without using outcome information."""
    return clean_text(series).replace("", pd.NA).fillna("Unknown").astype(str)


def load_clean_cohort(
    raw_dir: Path, chunk_size: int = 100_000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the cleaned FY2001+ cohort used by the modelling workflow."""
    repeated_rows, duplicate_audit = verify_duplicate_rows(raw_dir, chunk_size)
    frames = []
    counts = defaultdict(int)

    for filename in EXPECTED_FILES:
        offset = 0
        for frame in pd.read_csv(
            raw_dir / filename,
            usecols=list(RAW_COLUMNS),
            chunksize=chunk_size,
            low_memory=False,
        ):
            row_numbers = np.arange(offset, offset + len(frame))
            repeated = np.isin(row_numbers, list(repeated_rows[filename]))
            offset += len(frame)

            counts["published_rows"] += len(frame)
            counts["verified_repeated_rows_removed"] += int(repeated.sum())
            frame = frame.loc[~repeated].copy()

            approval_fy = pd.to_numeric(frame["ApprovalFY"], errors="coerce").astype(
                "Int64"
            )
            approval_date = pd.to_datetime(frame["ApprovalDate"], errors="coerce")
            chargeoff_date = pd.to_datetime(frame["ChargeOffDate"], errors="coerce")
            as_of_date = pd.to_datetime(frame["AsOfDate"], errors="coerce")
            status = status_value(frame["LoanStatus"])
            gross = pd.to_numeric(frame["GrossApproval"], errors="coerce")
            guaranteed = pd.to_numeric(
                frame["SBAGuaranteedApproval"], errors="coerce"
            )
            term = pd.to_numeric(frame["TermInMonths"], errors="coerce")
            interest_rate = pd.to_numeric(
                frame["InitialInterestRate"], errors="coerce"
            )

            eligible = (
                approval_fy.between(2001, 2016)
                & approval_date.notna()
                & approval_date.le(SNAPSHOT - pd.DateOffset(months=120))
                & status.isin(["CHGOFF", "PIF", "EXEMPT"])
            )
            eligible &= ~(
                status.eq("CHGOFF") & chargeoff_date.gt(as_of_date)
            )
            eligible &= status.ne("CHGOFF") | (
                chargeoff_date.notna()
                & chargeoff_date.ge(approval_date)
                & chargeoff_date.le(as_of_date)
            )

            valid_term = term.between(1, 360)
            valid_amounts = (
                gross.gt(0) & guaranteed.ge(0) & guaranteed.le(gross)
            )

            counts["fixed_horizon_before_validity_checks"] += int(eligible.sum())
            counts["excluded_invalid_term"] += int((eligible & ~valid_term).sum())
            counts["excluded_invalid_amount"] += int(
                (eligible & valid_term & ~valid_amounts).sum()
            )

            keep = eligible & valid_term & valid_amounts
            if not keep.any():
                continue

            target_120 = status.eq("CHGOFF") & chargeoff_date.le(
                approval_date + pd.DateOffset(months=120)
            )
            target_96 = status.eq("CHGOFF") & chargeoff_date.le(
                approval_date + pd.DateOffset(months=96)
            )
            naics = pd.to_numeric(frame["NaicsCode"], errors="coerce").astype("Int64")
            naics2 = (
                naics.astype("string")
                .str.zfill(6)
                .str[:2]
                .where(naics.notna(), "Unknown")
            )

            frames.append(
                pd.DataFrame(
                    {
                        "approval_fy": approval_fy[keep].astype(int),
                        "approval_date": approval_date[keep],
                        "target_120": target_120[keep].astype("int8"),
                        "target_96": target_96[keep].astype("int8"),
                        "gross_approval": gross[keep].astype(float),
                        "term_months": term[keep].astype(float),
                        "guarantee_percent": (
                            100 * guaranteed[keep].astype(float) / gross[keep]
                        ),
                        "naics2": naics2[keep].fillna("Unknown").astype(str),
                        "project_state": _category(frame["ProjectState"])[keep],
                        "business_age": business_age_group(frame["BusinessAge"])[keep]
                        .replace("Unknown or unmapped", "Unknown")
                        .astype(str),
                        "business_type": _category(frame["BusinessType"])[keep],
                        "processing_method": _category(frame["ProcessingMethod"])[
                            keep
                        ],
                        "initial_interest_rate": interest_rate[keep].astype(float),
                    }
                )
            )

    cohort = pd.concat(frames, ignore_index=True)
    counts["clean_fy2001_plus_rows"] = len(cohort)

    summary = pd.DataFrame(
        [{"measure": measure, "count": count} for measure, count in counts.items()]
    )
    return cohort, summary, pd.DataFrame(duplicate_audit)


def primary_analysis_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return the final FY2010+ cohort used for statistics and primary modelling."""
    valid_interest = cohort["initial_interest_rate"].notna() & cohort[
        "initial_interest_rate"
    ].ge(0)
    primary = cohort[cohort["approval_fy"].ge(2010) & valid_interest].copy()
    return primary.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--processed-file",
        type=Path,
        default=Path("data/processed/sba_7a_analysis_cohort.parquet"),
    )
    parser.add_argument(
        "--audit-dir", type=Path, default=Path("reports/tables/cohort_design")
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    args.processed_file.parent.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)

    cohort, summary, duplicate_audit = load_clean_cohort(
        args.raw_dir, args.chunk_size
    )
    primary = primary_analysis_cohort(cohort)

    primary.to_parquet(args.processed_file, index=False)
    duplicate_audit.to_csv(
        args.audit_dir / "duplicate_verification.csv", index=False
    )

    primary_summary = pd.DataFrame(
        [
            {"measure": "analysis_rows", "count": len(primary)},
            {"measure": "analysis_columns", "count": len(primary.columns)},
            {"measure": "chargeoffs_within_120_months", "count": int(primary["target_120"].sum())},
            {"measure": "chargeoffs_within_96_months", "count": int(primary["target_96"].sum())},
            {"measure": "first_approval_fy", "count": int(primary["approval_fy"].min())},
            {"measure": "last_approval_fy", "count": int(primary["approval_fy"].max())},
        ]
    )
    pd.concat([summary, primary_summary], ignore_index=True).to_csv(
        args.audit_dir / "cohort_summary.csv", index=False
    )

    print(
        f"Saved {len(primary):,} rows and {len(primary.columns)} columns "
        f"to {args.processed_file}"
    )


if __name__ == "__main__":
    main()
