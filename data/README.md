# Data

## Source

This project uses four U.S. Small Business Administration 7(a) loan files covering FY1991 through June 2026, together with the official 7(a) data dictionary.

* [SBA 7(a) and 504 FOIA dataset](https://data.sba.gov/dataset/7a-504-foia)
* Snapshot date: June 30, 2026
* Dataset identifier: `SBA-OHA-2016-08-001`

The 504 loan files are outside the scope of this project.

## Local data

The raw files are stored locally in `data/raw/` and are not committed to GitHub. They contain borrower names and addresses that are not needed for this portfolio analysis.

After downloading the four 7(a) CSV files and the data dictionary, verify their structure with:

```bash
python src/data/verify_schema.py
```

The expected filenames and file hashes are documented in
[`docs/01_data_sources.md`](../docs/01_data_sources.md).

## Data preparation

Run the data-quality profile with:

```bash
python src/data/profile_data_quality.py
```

Build the cohort and supporting audit tables with:

```bash
python src/data/build_cohort.py
```

The scripts check field quality, repeated records, dates, loan status, observation periods and the ten-year charge-off outcome. Raw source values are not overwritten.

## Processed dataset

A local analysis-ready cohort is stored at:

```text
data/processed/sba_7a_analysis_cohort.parquet
```

It contains 307,957 FY2010+ loans and 12 analytical fields. Borrower names, addresses, ZIP codes, borrower identifiers and lender identifiers are excluded.

Both `data/raw/` and `data/processed/` are ignored by Git and remain on the local computer. Public project outputs contain only aggregate results.
