# Data sources

## Official publication

This project uses the U.S. Small Business Administration’s 7(a) and 504 FOIA publication.

* [SBA 7(a) and 504 FOIA dataset](https://data.sba.gov/dataset/7a-504-foia)
* [SBA 7(a) and 504 loan data dictionary](https://www.sba.gov/document/support--7a-504-loan-data-dictionary)
* Dataset identifier: `SBA-OHA-2016-08-001`
* Data snapshot: June 30, 2026
* Publication frequency: Quarterly

Only the four 7(a) files and the 7(a) worksheet of the shared data dictionary were used. The two 504 files cover a different loan program and are outside the scope of this project.

## Files used

| Approval period  | Official filename                        |           Size |       Records |
| ---------------- | ---------------------------------------- | -------------: | ------------: |
| FY1991–FY1999    | `FOIA_7a_FY1991_FY1999_asof_260630.csv`  |     139.63 MiB |       337,033 |
| FY2000–FY2009    | `FOIA_7a_FY2000_FY2009_asof_260630.csv`  |     303.67 MiB |       690,333 |
| FY2010–FY2019    | `FOIA_7a_FY2010_FY2019_asof_260630.csv`  |     243.28 MiB |       545,751 |
| FY2020–June 2026 | `FOIA_7a_FY2020_Present_asof_260630.csv` |     172.74 MiB |       388,338 |
| **Total**        |                                          | **859.33 MiB** | **1,961,455** |

The record counts exclude the header. Physical line counts can be slightly higher because quoted text fields may contain line breaks.

## Schema verification

The 7(a) worksheet in the official data dictionary contains 42 fields. All four CSV files contained the same 42 columns in the same order, with no undocumented or missing fields.

SHA-256 hashes were recorded during source validation so the downloaded files can be checked against the snapshot used for this project.

| Approval period  | SHA-256                                                            |
| ---------------- | ------------------------------------------------------------------ |
| FY1991–FY1999    | `05040efc4a43224a02460d606ea744579a54b650bdb99663a833ed0b31936213` |
| FY2000–FY2009    | `66674e18a700fbba0378c25118c291ac2759784a550c643cab5747eced763d6a` |
| FY2010–FY2019    | `01a3e2c7988a6f4052e53f218a309feb2ec2fe42887bebdc0fa94ac8b1024ade` |
| FY2020–June 2026 | `6c1e9132b5141a19f82bdc8ccafb86c9a01662461cad41ddb36a3cf409d8a4fe` |

## Local setup

Download the four 7(a) CSV files and `7a_504_foia_data_dictionary.xlsx`, then place them in `data/raw/`:

```text
data/raw/
├── FOIA_7a_FY1991_FY1999_asof_260630.csv
├── FOIA_7a_FY2000_FY2009_asof_260630.csv
├── FOIA_7a_FY2010_FY2019_asof_260630.csv
├── FOIA_7a_FY2020_Present_asof_260630.csv
└── 7a_504_foia_data_dictionary.xlsx
```

Validate the files before running the analysis:

```bash
python src/data/verify_schema.py
```

## Privacy

The raw files are not included in this repository because they contain borrower and lender information that is not needed in a public portfolio.

The public repository does not contain:

* Raw SBA files
* Borrower or lender names
* Street addresses
* Exact ZIP-level records
* Borrower or lender identifiers
* Individual predictions or risk scores
* Fitted model binaries

Published tables and Tableau sources contain aggregate results only.
