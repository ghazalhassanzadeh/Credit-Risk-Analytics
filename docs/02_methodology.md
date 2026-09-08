# Methodology

## Objective

This project examines historical charge-off patterns in the SBA 7(a) portfolio and tests whether information available at loan approval can support portfolio monitoring.

The main outcome is **charge-off within 120 months of approval**, referred to as ten-year charge-off risk. It is not a measure of lifetime default or general business failure.

The analysis separates three types of evidence:

1. Observed portfolio patterns
2. Relationships after statistical adjustment
3. Predictive performance on later approval years

This distinction is important because an observed difference does not automatically represent an independent or causal effect.

## Data validation

The project uses four official SBA 7(a) files covering FY1991 through June 2026. Before analysis, the files were checked for:

* Expected filenames and file hashes
* Row and column counts
* Agreement with the 42-field SBA data dictionary
* Missing and unusual values
* Date ranges and invalid date sequences
* Historical field availability
* Potential duplicate records
* Variables that could create target leakage

All four files contained the same 42 columns in the same order.

## Duplicate and data-quality handling

The SBA files do not include a verified unique loan identifier. Duplicate checks therefore compared all 42 published fields. A total of 1,918 exact repeated copies were removed from the analytical data. The original source files were not changed.

Unusual values were reviewed according to how they affected each analysis. Invalid dates, loan terms and monetary relationships were not automatically corrected or replaced.

For example, a questionable recorded charge-off amount could be excluded from a financial calculation without removing the loan from a count-based charge-off analysis.

## Ten-year outcome

A loan was eligible for the ten-year outcome when:

* It had a complete 120-month observation window by June 30, 2026
* Its status and dates supported a clear outcome
* Any recorded charge-off occurred within a valid period after approval

A loan was assigned a positive outcome when it had a valid charge-off date within 120 months of approval. Loans with no charge-off during the complete observation window were assigned a negative outcome.

Cancelled commitments, unresolved records with insufficient follow-up, invalid event timing and post-snapshot charge-offs were excluded.

Recent loans are not included in the ten-year outcome because they have not been observed long enough. They remain available for portfolio-volume and composition reporting.

## Analytical populations

The descriptive FY2010+ fixed-horizon population contained:

* 308,134 eligible loans
* 20,299 charge-offs within 120 months
* A ten-year charge-off rate of 6.59%

For statistical and predictive modelling, 177 records with invalid original terms were excluded. The final FY2010+ modelling population contained:

* 307,957 loans
* 20,199 charge-offs
* A ten-year charge-off rate of 6.56%

A longer FY2001+ population was also evaluated to test whether older stress-period data improved predictions for later loans.

## Descriptive analysis

The portfolio analysis examined:

* Approval count and gross approval by fiscal year
* Ten-year charge-off risk by approval vintage
* Loan size and approved-dollar exposure
* Original term
* Industry and geography
* Business age and business type
* Guarantee percentage and initial interest rate

Segment risk comparisons required at least 500 eligible loans and 25 charge-offs. Smaller groups could remain in volume reporting but were not treated as reliable risk rankings.

Gross approval was used as the main exposure measure. Recorded gross charge-off amount was treated as a supporting field because some values exceeded gross approval. Financial results therefore include an anomaly sensitivity.

## Statistical analysis

A multivariable logistic regression was used to examine whether the main descriptive patterns remained after accounting for approval vintage and observable portfolio characteristics.

The model included:

* Gross approval
* Original term
* Guarantee percentage
* Initial interest rate
* Two-digit NAICS industry
* Project state
* Business age
* Business type
* Processing method

Continuous variables were allowed to have curved relationships. Adjusted risks and confidence intervals were calculated in Python before being prepared for Tableau.

This model was explanatory. It was separate from the predictive machine-learning experiment and was not used to score loans.

## Predictive modelling

The predictive experiment compared:

* Regularized logistic regression as an interpretable baseline
* `HistGradientBoostingClassifier` as a non-linear model
* FY2010+ and FY2001+ training histories
* Feature sets with and without project state and processing method

The final selected model used:

* Gross approval
* Original term
* Guarantee percentage
* Two-digit NAICS industry
* Business age
* Business type
* Project state
* Initial interest rate

Processing method was excluded from the final model because its small test improvement was not consistent across validation periods.

Loan status, charge-off information, servicing dates, recorded losses, borrower and lender identities, addresses and other post-approval fields were excluded from the predictors.

## Time-based validation

A random train-test split was not used because portfolio composition and economic conditions changed over time.

Models were trained on earlier approval years and validated on later years. After preprocessing and model settings were selected, they were frozen before evaluation on the locked FY2015–FY2016 test population.

The test data was not used to select features, tune model settings or change preprocessing.

## Model evaluation

Model performance was assessed using:

* PR-AUC
* ROC-AUC
* Brier score
* Calibration
* Performance by approval vintage
* Top 5%, 10% and 20% monitoring-capacity scenarios

PR-AUC was the primary ranking measure because charge-off was the minority outcome. ROC-AUC was reported as supporting evidence.

The 10% capacity scenario is an analytical illustration, not a recommended lending threshold.

## Reproducibility and privacy

The raw SBA files are stored locally in `data/raw/`. A cleaned FY2010+ analytical cohort is stored locally in:

```text
data/processed/sba_7a_analysis_cohort.parquet
```

Both folders are excluded from Git.

The public repository contains code, documentation, aggregate tables, figures and privacy-safe Tableau sources. It does not contain borrower-level predictions, names, addresses, identifiers or fitted model binaries.
