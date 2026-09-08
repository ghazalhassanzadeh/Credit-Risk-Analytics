# Cohort-design tables

These two aggregate tables are created by `src/data/build_cohort.py`:

- `cohort_summary.csv` records the final cleaning and cohort counts.
- `duplicate_verification.csv` records verified repeated-row removals by file.

They contain no borrower names, addresses or row-level loan records. The local
analysis cohort is written to `data/processed/` and excluded from Git.
