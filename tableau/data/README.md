# Tableau aggregate data

These files support the four approved Tableau views. They are built from the portfolio, statistical and model-evaluation outputs by `src/tableau/prepare_tableau_data.py`.

No file contains borrower-level records, individual predictions, names, addresses, ZIP codes, borrower identifiers or lender identifiers. ProcessingMethod is excluded from the executive Tableau sources. State rates are withheld where the approved 500-loan and 25-charge-off rule is not met.

Use the CSV files as separate Tableau data sources. Do not join files with
different grains. See
[`docs/06_tableau_guide.md`](../../docs/06_tableau_guide.md) for the dashboard
structure and worksheet use.
