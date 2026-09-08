# Small Business Loan Portfolio Risk Analytics

An end-to-end analysis of U.S. Small Business Administration 7(a) loans, combining portfolio analysis, statistical adjustment, machine learning and Tableau-ready reporting to examine ten-year charge-off risk.

> **Project status:** The Python analysis and Tableau data preparation are complete. The interactive Tableau dashboards will be added after the manual Tableau Desktop build.

## Project overview

The SBA 7(a) programme supports lending to small businesses through participating lenders. This project examines how the portfolio changed over time, where observed charge-off risk and approved-dollar exposure were concentrated, which descriptive patterns remained after statistical adjustment, and whether an approval-time model could improve portfolio monitoring.

The analysis covers 1,961,455 published loan records from FY1991 through June 2026. The primary modelling outcome is charge-off within 120 months of approval. Only variables available at or near approval were considered as predictors.

The final model is intended as a research tool for portfolio monitoring. It is not designed for automated loan approval, denial, pricing or borrower-level intervention.

## Business questions

- How did SBA 7(a) lending volume and portfolio composition change over time?
- Which approval vintages and portfolio segments recorded higher ten-year charge-off rates?
- Where did charge-off frequency and approved-dollar exposure tell different stories?
- Which apparent risk patterns remained after controlling for vintage and observable portfolio mix?
- Can approval-time information help concentrate monitoring attention on a smaller share of the portfolio?
- How stable are model ranking and calibration across later approval periods?

## Key findings

- **Approval vintage mattered substantially.** FY2007 recorded a 35.78% ten-year charge-off rate, while eligible FY2011-FY2016 vintages ranged from 5.60% to 6.92%. This made temporal validation essential.
- **Frequency and exposure were not the same.** Smaller loans had higher unadjusted charge-off frequency, but loans above $1 million represented 51.36% of gross approvals in the FY2010+ analytical cohort.
- **The small-loan pattern changed after adjustment.** Once vintage and observable portfolio mix were controlled for, smaller loan amounts were not independently associated with higher estimated risk.
- **Original term produced the strongest separation.** It also requires careful interpretation because term reflects product structure and interacts with the fixed 120-month outcome horizon.
- **More history was not automatically better.** A model trained from FY2001 retained useful ranking information but produced probabilities that were less aligned with the later portfolio.
- **Processing method was excluded from the final model.** Its small locked-test improvement did not outweigh inconsistent behaviour across validation periods.

## Selected findings

### Ten-year risk changed sharply across approval vintages

![Ten-year charge-off risk by approval vintage](reports/figures/portfolio/02_vintage_chargeoff_risk.png)

Crisis-era vintages recorded substantially higher ten-year charge-off rates.
This is why the analysis uses time-based validation rather than a random
train-test split.

### Statistical adjustment changed the loan-size story

![Adjusted loan-size and term relationships](reports/figures/statistics/01_adjusted_size_and_term.png)

Smaller loans showed higher unadjusted charge-off rates, but that pattern
reversed after accounting for vintage and observable portfolio characteristics.
Original term remained strongly associated with the outcome.

## Final model

The selected specification, E06, uses scikit-learn's `HistGradientBoostingClassifier`. It was trained on FY2010-FY2014 approvals and evaluated on a locked FY2015-FY2016 test population.

Predictors:

- Gross approval amount
- Original term
- SBA guarantee percentage
- Two-digit NAICS sector
- Business age
- Business type
- Project state
- Initial interest rate

Approval vintage was used for temporal splitting and evaluation, not as a predictor. Borrower and lender identifiers, servicing fields, outcome fields and post-approval information were excluded.

### Locked-test performance

| Metric | Result |
|---|---:|
| Test loans | 97,439 |
| Charge-offs within 120 months | 6,540 |
| Observed event rate | 6.71% |
| PR-AUC | 0.656 |
| ROC-AUC | 0.964 |
| Brier score | 0.03335 |
| Mean predicted risk | 5.85% |

The model provided useful risk ranking, but its average predicted risk was below the observed event rate. It should therefore not be described as operationally calibrated or deployment-ready.

![Model feature contribution](reports/figures/modeling/03_grouped_permutation_importance.png)

Original term contributed most to model ranking, although this relationship
must be interpreted alongside product structure and the fixed 120-month outcome
window.

### Monitoring-capacity scenarios

| Share of portfolio reviewed | Charge-offs captured | Precision | Lift |
|---:|---:|---:|---:|
| 5% | 52.6% | 70.5% | 10.51 |
| 10% | 82.3% | 55.2% | 8.23 |
| 20% | 95.7% | 32.1% | 4.79 |

![Charge-offs captured by review capacity](reports/figures/modeling/02_monitoring_capacity.png)

The 10% scenario illustrates how a portfolio team could assess monitoring capacity. It is not a recommended operational threshold.

## Analytical approach

1. **Source validation:** Verified the four official SBA files against the 42-field data dictionary and recorded file hashes.
2. **Data-quality assessment:** Profiled missingness, dates, categories, numeric anomalies, potential duplicates and feature timing.
3. **Cohort design:** Constructed a fixed 120-month outcome while accounting for incomplete follow-up and invalid event timing.
4. **Portfolio analysis:** Compared approval volume, vintage risk, segment frequency and gross-approval exposure.
5. **Statistical analysis:** Estimated adjusted relationships using multivariable logistic regression and marginal standardisation.
6. **Machine learning:** Compared regularized logistic regression with histogram gradient boosting using expanding-window validation.
7. **Locked evaluation:** Assessed ranking, calibration, temporal stability and monitoring-capacity scenarios on later vintages.
8. **Tableau preparation:** Created privacy-safe aggregate sources for four connected dashboards.

## Tableau story

The planned Tableau portfolio contains four connected views:

1. **Portfolio & Vintage Context:** How did lending volume, exposure and observed risk change?
2. **Risk Frequency & Exposure Concentration:** Which segments mattered because of event frequency, dollar exposure or both?
3. **Observed vs Adjusted Risk:** Which descriptive patterns remained after statistical adjustment?
4. **Model Monitoring & Capacity:** What additional monitoring value did E06 provide, and where are its limits?

<!-- Add the final Tableau dashboard image and Tableau Public link here after the workbook is complete. -->

## Repository guide

```text
Credit-Risk-Analytics/
|-- README.md
|-- requirements.txt
|-- data/                  # Download instructions; raw data remains local
|-- docs/                  # Methodology, findings and model documentation
|-- src/
|   |-- data/              # Source validation and cohort preparation
|   |-- analysis/          # Portfolio and statistical analysis
|   |-- modeling/          # Model training and evaluation
|   `-- tableau/           # Tableau aggregate-data preparation
|-- reports/
|   |-- figures/           # Selected analytical charts
|   `-- tables/            # Selected aggregate results
|-- tableau/
|   `-- data/              # Privacy-safe aggregate CSV sources
`-- config/                # Frozen model configuration
```

## Data

Source: [U.S. Small Business Administration 7(a) & 504 FOIA dataset](https://data.sba.gov/dataset/7a-504-foia)

This project uses only the four 7(a) CSV files and the 7(a) worksheet from the official data dictionary. The two 504 files are outside scope.

Raw files are not stored in this repository. Although the source is publicly available, the files are large and contain borrower names and addresses that are unnecessary for a public portfolio. Reproduction instructions and expected filenames are provided in `data/README.md`.

## Reproducing the analysis

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

After downloading the four official 7(a) files and the data dictionary into
`data/raw/`, run the main analysis from the repository root:

```powershell
python src/data/verify_schema.py
python src/data/profile_data_quality.py
python src/data/build_cohort.py
python src/analysis/analyze_portfolio.py
python src/analysis/run_statistical_analysis.py
```

The modelling workflow is separated into auditable stages because it is more
computationally expensive:

```powershell
python src/modeling/train_and_evaluate.py audit
python src/modeling/train_and_evaluate.py develop
python src/modeling/train_and_evaluate.py test
python src/modeling/summarize_model_results.py
python src/modeling/run_sensitivity_checks.py
python src/tableau/prepare_tableau_data.py
```

The `develop` and `test` stages refit multiple models and can take considerable
time on a local computer. Existing aggregate results are included for review.

## Tools

- Python
- pandas and NumPy
- SciPy
- scikit-learn
- Matplotlib
- Excel
- Tableau
- Git and GitHub

## Limitations

- The published data does not include credit scores, borrower financial statements, verified collateral values or complete underwriting information.
- Complete ten-year follow-up limits the primary outcome population to older approvals.
- The primary FY2010+ analysis contains seven eligible approval vintages.
- The locked test covers FY2015 and only the eligible portion of FY2016.
- The selected model underpredicted the locked-test event rate.
- Programme rules, portfolio composition and category definitions changed over time.
- State improved aggregate model performance, but state-level calibration was uneven.
- Original term may capture product structure and the relationship between contractual term and the fixed outcome horizon.
- The findings describe historical and predictive associations, not causal effects.

## Responsible use

This work supports aggregate portfolio analysis and monitoring research. It does not evaluate individual borrower quality, approval suitability, discrimination or fair-lending compliance. No borrower-level predictions, names, addresses, exact ZIP-level records or fitted model binaries are included in the public repository.

The project is independent and is not endorsed by the U.S. Small Business Administration.
