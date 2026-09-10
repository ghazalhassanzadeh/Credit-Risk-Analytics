# Tableau story guide

## Story concept

The completed Tableau portfolio uses five connected story points:

**Portfolio Context → Risk and Exposure → Geography → Statistical Adjustment → Model Monitoring**

The interactive story was built in Tableau Desktop using privacy-safe aggregate CSV files from `tableau/data/`. These sources do not contain borrower-level records, identifying information, or individual model scores.

## 1. How the portfolio changed

**Question:** How did SBA 7(a) lending activity and historical charge-off risk change over time?

This story point combines:

- approval count by fiscal year
- gross approval amount by fiscal year
- ten-year charge-off rate by approval vintage

The results show that portfolio risk changed substantially across economic periods. The ten-year charge-off rate reached 35.78% for FY2007 approvals before declining considerably in later mature vintages.

Recent approval years remain visible in the volume charts but are not presented as complete ten-year outcomes.

## 2. Where risk and exposure concentrate

**Question:** Where were charge-offs more frequent, and where were approved dollars concentrated?

This story point compares:

- ten-year charge-off rates across loan-size segments
- each segment’s share of gross approved dollars
- charge-off patterns across 96-month and 120-month outcome windows

Smaller loans had higher observed charge-off frequency, while loans above $1 million represented more than half of gross approved-dollar exposure. This demonstrates why risk frequency and financial exposure should be monitored separately.

## 3. Geographic portfolio context

**Question:** How are loan activity, exposure, and observed outcomes distributed across states and territories?

The geographic view provides:

- state or territory name
- eligible loan count
- gross approval amount
- portfolio share
- ten-year charge-off rate
- mix-adjusted risk estimate

The map is intended to provide portfolio context rather than a state risk ranking. Differences may reflect approval vintage, loan mix, sample size, and uneven model calibration across states.

## 4. What changes after adjustment

**Question:** Which descriptive patterns remain after accounting for observed portfolio composition?

This story point compares observed and adjusted estimates for selected borrower and loan characteristics. It also shows adjusted risk across representative loan-size scenarios.

The adjustment controls for approval vintage and available portfolio characteristics. These estimates describe associations and should not be interpreted as causal effects.

## 5. How the model supports monitoring

**Question:** What monitoring value does the predictive model provide?

The final story point presents the primary locked-test results:

| Metric | Result |
|---|---:|
| Test loans | 97,439 |
| Observed charge-offs | 6,540 |
| Observed event rate | 6.71% |
| Mean predicted risk | 5.85% |
| PR AUC | 0.656 |
| ROC AUC | 0.964 |
| Brier score | 0.03335 |

It also compares monitoring-capacity scenarios:

| Portfolio reviewed | Observed charge-offs captured |
|---:|---:|
| 5% | 52.6% |
| 10% | 82.3% |
| 20% | 95.7% |

These are monitoring scenarios, not recommended decision thresholds. The calibration chart shows that the model ranked risk effectively but understated overall locked-test risk.

The model is presented as a research tool for retrospective portfolio monitoring and review prioritisation—not automated lending decisions.

## Visual approach

The story uses:

- consistent typography and spacing
- a restrained professional colour palette
- short explanatory subtitles
- formatted tooltips
- percentage and currency formatting appropriate to each measure
- state abbreviations on the map and full state names in tooltips
- consistent story-point navigation

The dashboards intentionally exclude borrower-level records, lender or borrower identifiers, individual model scores, state risk league tables, and detailed model-tuning results.

## Live story

[Open the interactive Tableau Public story](https://public.tableau.com/views/SBA7aLoanPortfolioRiskAnalytics/SBA7aLoanPortfolioRiskStory?:language=en-GB&:showVizHome=no)