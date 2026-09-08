"""Verify the local SBA 7(a) source files before running the analysis."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

EXPECTED_FILES = {
    "FOIA_7a_FY1991_FY1999_asof_260630.csv": {
        "rows": 337_033,
        "sha256": "05040efc4a43224a02460d606ea744579a54b650bdb99663a833ed0b31936213",
    },
    "FOIA_7a_FY2000_FY2009_asof_260630.csv": {
        "rows": 690_333,
        "sha256": "66674e18a700fbba0378c25118c291ac2759784a550c643cab5747eced763d6a",
    },
    "FOIA_7a_FY2010_FY2019_asof_260630.csv": {
        "rows": 545_751,
        "sha256": "01a3e2c7988a6f4052e53f218a309feb2ec2fe42887bebdc0fa94ac8b1024ade",
    },
    "FOIA_7a_FY2020_Present_asof_260630.csv": {
        "rows": 388_338,
        "sha256": "6c1e9132b5141a19f82bdc8ccafb86c9a01662461cad41ddb36a3cf409d8a4fe",
    },
}
DICTIONARY_FILE = "7a_504_foia_data_dictionary.xlsx"
DICTIONARY_SHEET = "7(a) Data Dictionary"
DATE_COLUMNS = (
    "AsOfDate",
    "ApprovalDate",
    "FirstDisbursementDate",
    "PaidInFullDate",
    "ChargeOffDate",
)


def calculate_sha256(path: Path) -> str:
    """Calculate a file fingerprint without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialise(value):
    """Convert pandas and NumPy scalar values to ordinary Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def profile_file(path: Path) -> tuple[dict, pd.DataFrame, list[str]]:
    """Profile one source file and return its summary, fields, and column order."""
    frame = pd.read_csv(path, low_memory=False)
    row_count = len(frame)
    missing_counts = frame.isna().sum()
    blank_counts = {
        column: (
            int(frame[column].str.strip().eq("").sum())
            if pd.api.types.is_string_dtype(frame[column])
            else 0
        )
        for column in frame.columns
    }

    date_summary = {}
    for column in DATE_COLUMNS:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        date_summary[f"{column}_minimum"] = (
            parsed.min().date().isoformat() if parsed.notna().any() else None
        )
        date_summary[f"{column}_maximum"] = (
            parsed.max().date().isoformat() if parsed.notna().any() else None
        )
        date_summary[f"{column}_unparsed_non_null"] = int(
            (frame[column].notna() & parsed.isna()).sum()
        )

    summary = {
        "file": path.name,
        "size_mib": round(path.stat().st_size / 1024**2, 2),
        "sha256": calculate_sha256(path),
        "rows": row_count,
        "columns": len(frame.columns),
        "approval_fy_min": serialise(frame["ApprovalFY"].min()),
        "approval_fy_max": serialise(frame["ApprovalFY"].max()),
        "program_values": ", ".join(
            sorted(frame["Program"].dropna().astype(str).unique())
        ),
        **date_summary,
    }

    field_rows = [
        {
            "file": path.name,
            "field": column,
            "pandas_dtype": str(frame[column].dtype),
            "missing_count": int(missing_counts[column]),
            "missing_percent": round(100 * missing_counts[column] / row_count, 4),
            "blank_string_count": blank_counts[column],
            "distinct_count": int(frame[column].nunique(dropna=True)),
        }
        for column in frame.columns
    ]
    return summary, pd.DataFrame(field_rows), frame.columns.tolist()


def load_dictionary_fields(path: Path) -> list[str]:
    """Read the documented 7(a) field names in their official order."""
    dictionary = pd.read_excel(path, sheet_name=DICTIONARY_SHEET)
    return dictionary["Field Name"].dropna().astype(str).tolist()


def require_source_files(raw_dir: Path) -> None:
    """Raise a clear error when an expected local source file is missing."""
    required = [*EXPECTED_FILES, DICTIONARY_FILE]
    missing = [name for name in required if not (raw_dir / name).is_file()]
    if missing:
        formatted = "\n- ".join(missing)
        raise FileNotFoundError(f"Missing source files:\n- {formatted}")


def validate_source(
    summaries: list[dict],
    column_orders: list[list[str]],
    dictionary_fields: list[str],
) -> None:
    """Stop when the local snapshot does not match the approved source."""
    errors = []

    for summary in summaries:
        expected = EXPECTED_FILES[summary["file"]]
        if summary["rows"] != expected["rows"]:
            errors.append(
                f"{summary['file']}: expected {expected['rows']:,} rows, "
                f"found {summary['rows']:,}"
            )
        if summary["sha256"] != expected["sha256"]:
            errors.append(f"{summary['file']}: SHA-256 fingerprint does not match")

    if len(dictionary_fields) != 42:
        errors.append(
            f"The 7(a) dictionary should contain 42 fields, found {len(dictionary_fields)}"
        )
    if any(columns != column_orders[0] for columns in column_orders[1:]):
        errors.append("The four source files do not use the same column order")
    if column_orders[0] != dictionary_fields:
        errors.append("Source columns do not match the official 7(a) dictionary")

    if errors:
        raise ValueError("Source verification failed:\n- " + "\n- ".join(errors))


def main() -> None:
    """Verify the source snapshot and write two readable audit tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/tables/data_audit"),
    )
    args = parser.parse_args()

    require_source_files(args.raw_dir)
    dictionary_fields = load_dictionary_fields(args.raw_dir / DICTIONARY_FILE)
    summaries = []
    field_profiles = []
    column_orders = []

    for filename in EXPECTED_FILES:
        summary, fields, columns = profile_file(args.raw_dir / filename)
        summaries.append(summary)
        field_profiles.append(fields)
        column_orders.append(columns)

    validate_source(summaries, column_orders, dictionary_fields)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "source_file_summary.csv",
        index=False,
    )
    pd.concat(field_profiles, ignore_index=True).to_csv(
        args.output_dir / "field_profile.csv",
        index=False,
    )
    print("Source verification passed for all four SBA 7(a) files.")


if __name__ == "__main__":
    main()
