# Model evaluation

## Objective

The predictive analysis tests whether information available when a loan is approved can help identify concentrations of ten-year charge-off risk.

The model is intended for portfolio monitoring. It does not predict business failure or overall borrower quality, and it should not be used for loan approval, denial, pricing or borrower-level intervention.

## Validation design

The primary experiment used loans approved from FY2010 through FY2014 for model development. Loans from FY2015 and the eligible portion of FY2016 formed a locked test set.

The test set contained 97,439 loans and 6,540 charge-offs, giving an observed event rate of 6.71%. All preprocessing and model choices were completed before evaluating this test population.

Time-based validation was used instead of random train-test splitting so that the experiment better reflected how a model would perform on later approval periods.

## Models compared

Two model families were evaluated:

* Regularised logistic regression as an interpretable baseline
* `HistGradientBoostingClassifier` as a non-linear alternative

Each model was tested with controlled feature combinations, including versions with and without project state and processing method.

A separate FY2001+ experiment examined whether adding older historical records improved performance on the same locked test.

Preprocessing was fitted separately within each training period. Rare and previously unseen categories were grouped using training data only. No SMOTE, resampling or automatic class weighting was used.

## Selected model

The selected model, E06, is a histogram gradient-boosting classifier using:

* Gross approval
* Original term
* Guarantee percentage
* Two-digit NAICS industry
* Business age
* Business type
* Project state
* Initial interest rate

Approval vintage was used to create the time-based splits but was not included as a predictor. Processing method was excluded from the final specification.

## Locked-test performance

| Metric              |  Result |
| ------------------- | ------: |
| PR-AUC              |   0.656 |
| ROC-AUC             |   0.964 |
| Brier score         | 0.03335 |
| Observed event rate |   6.71% |
| Mean predicted risk |   5.85% |

The ROC-AUC indicates strong separation between higher-risk and lower-risk loans, but it should not be interpreted alone because charge-offs are relatively uncommon.

PR-AUC provides a more informative view of performance among higher-risk predictions. The Brier score measures the quality of the predicted probabilities.

The model’s average predicted risk was 5.85%, compared with an observed test rate of 6.71%. This means the model ranked risk effectively but slightly underpredicted the overall event rate. Its probabilities should not be treated as fully calibrated.

## Monitoring-capacity scenarios

The predicted scores were also evaluated as a portfolio-ranking tool.

| Share of portfolio reviewed | Loans reviewed | Charge-offs captured | Precision |  Lift |
| --------------------------: | -------------: | -------------------: | --------: | ----: |
|                          5% |          4,872 |                52.6% |     70.5% | 10.51 |
|                         10% |          9,744 |                82.3% |     55.2% |  8.23 |
|                         20% |         19,488 |                95.7% |     32.1% |  4.79 |

The highest-scored 10% of the test portfolio contained 82.3% of observed charge-offs. This illustrates how the model could help analysts focus limited review capacity on a smaller part of the portfolio.

The 10% scenario is an analytical example, not a recommended operational threshold. It does not justify an automatic decision or borrower-level action.

## Why E06 was selected

A second model, E08, included processing method and achieved a slightly higher locked-test PR-AUC of 0.659. The precise improvement over E06 was only 0.0023.

Processing method behaved inconsistently across the forward-validation periods. It was also less useful when the longer historical period was included. The small improvement on the locked test was therefore not strong enough to justify adding an unstable feature.

E06 was selected because it provided nearly identical performance with a simpler and more stable feature set. This choice reflects model governance as well as predictive performance.

Adding project state to the core boosting model improved PR-AUC from 0.641 to 0.656 and reduced the Brier score from 0.03419 to 0.03335. However, calibration was not equally stable across all states. Geography is therefore useful for portfolio-level prediction, but state-level probabilities require caution.

## What the longer history showed

The FY2001+ boosting model achieved a PR-AUC of 0.622 on the same test set. It retained useful ranking ability, but its probability estimates were less aligned with the later portfolio.

The older data included the financial-crisis period and other historical conditions that differed from the FY2015 to FY2016 test population. Although those records provided useful stress-period information, they did not improve the primary model.

This comparison shows why more historical data is not automatically better for estimating current portfolio risk.

## Feature interpretation

Original term contributed most to the model’s ranking performance. Initial interest rate and guarantee percentage were the next most influential feature groups. Project state, loan amount and industry provided smaller improvements, while business age and business type added less after the other variables were included.

Removing initial interest rate reduced PR-AUC from 0.656 to 0.633 and lowered top-10% charge-off capture from 82.3% to 79.9%.

A sensitivity test using a 96-month outcome produced a similar concentration pattern, with the highest-scored 10% capturing 82.1% of charge-offs.

These results describe predictive contribution, not causation. Original term may reflect loan structure and its relationship with the fixed outcome window. Interest rate may capture economic conditions and risk pricing. State and industry may reflect portfolio composition rather than characteristics of an individual borrower.

## Model status and limitations

E06 is a research model for portfolio monitoring. It provides useful risk ranking, but its absolute probabilities are not fully calibrated and it is not deployment-ready.

Important limitations include:

* No credit scores or borrower financial statements
* No verified collateral or detailed underwriting information
* Only seven eligible FY2010+ approval vintages
* A locked test limited to FY2015 and eligible FY2016 approvals
* Uneven calibration across some states and portfolio segments
* Changes in programme and category definitions over time
* A strong relationship between original term and the fixed outcome window
* Predictive associations that should not be interpreted as causal effects

New outcome data would be required to reassess performance and calibration before the model could support any operational use.

The modelling code is stored in `src/modeling/`. Aggregate evaluation results are available in `reports/tables/modeling/`, with supporting figures in `reports/figures/modeling/`.

The public repository does not contain fitted model files, borrower-level predictions or individual risk scores.
