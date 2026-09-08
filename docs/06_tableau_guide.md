# Tableau dashboard guide

## Dashboard concept

The planned Tableau portfolio uses four connected dashboards to tell the project story:

**Context → Concentration → Adjustment → Model**

The workbook will be built manually in Tableau Desktop using the safe aggregate CSV files in `tableau/data/`. These files do not contain borrower-level records, identifying information or individual model scores.

## 1. Portfolio context

**Question:** How did lending activity and historical charge-off risk change over time?

The first dashboard will combine:

* Approval count by fiscal year
* Gross approval by fiscal year
* Ten-year charge-off rate by approval vintage

The main message is that portfolio risk changed considerably across economic periods. For example, the ten-year charge-off rate reached 35.78% for FY2007 approvals before falling below 9% in FY2010.

Recent approvals will remain visible in the volume charts, but they will not be presented as complete ten-year outcomes.

## 2. Risk and exposure concentration

**Question:** Where were charge-offs frequent, and where were approved dollars concentrated?

A scatterplot will compare charge-off rates with gross approval share across:

* Loan size
* Original term
* Industry
* Business age

This view will show why charge-off frequency and financial exposure should be monitored separately. Smaller loans had higher observed charge-off rates, while loans above $1 million represented more than half of approved-dollar exposure.

A supporting state view may show geographic exposure, but it will not rank states by risk.

## 3. Observed and adjusted risk

**Question:** Which descriptive patterns remained after accounting for portfolio composition?

An observed-versus-adjusted comparison will open with business age. Supporting charts will show adjusted relationships for loan amount, original term, guarantee percentage and initial interest rate.

The main finding is that some descriptive patterns changed after adjustment. In particular, smaller loans had higher observed charge-off rates, but this relationship reversed after controlling for the available portfolio characteristics.

The dashboard will clearly describe adjusted estimates as associations, not causal effects.

## 4. Model monitoring

**Question:** What monitoring value does the predictive model provide?

The model dashboard will present the main locked-test results:

| Metric              |  Result |
| ------------------- | ------: |
| PR-AUC              |   0.656 |
| ROC-AUC             |   0.964 |
| Brier score         | 0.03335 |
| Observed event rate |   6.71% |
| Mean predicted risk |   5.85% |

The main chart will compare three review-capacity scenarios:

| Portfolio reviewed | Charge-offs captured |
| -----------------: | -------------------: |
|                 5% |                52.6% |
|                10% |                82.3% |
|                20% |                95.7% |

The 10% scenario will be highlighted as an example of portfolio prioritisation, not as a recommended decision threshold.

A calibration chart will show that the model ranked risk effectively but slightly underpredicted the overall event rate. The model will be presented as a research tool for portfolio monitoring, not as a system for automated lending decisions.

## Visual approach

The dashboards will use a consistent professional style:

* Navy for exposure
* Amber for observed risk
* Teal for adjusted estimates
* Indigo for model results
* Simple navigation in the same location on every dashboard
* Clear annotations and short tooltips
* No gauges, rainbow palettes or red-versus-green risk labels

Borrower-level records, state risk rankings, processing-method rankings and detailed model-tuning results will remain outside the executive dashboards.
