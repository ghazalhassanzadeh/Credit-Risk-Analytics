# Statistical analysis

## Purpose

The descriptive analysis showed how charge-off rates varied across loan and business characteristics. This section tests whether those patterns remained after accounting for approval year and differences in portfolio composition.

The analysis uses 307,957 loans approved from FY2010 onward, including 20,199 charge-offs within 120 months. It is explanatory rather than predictive. Its purpose is to compare observed and adjusted relationships, not to score individual loans or estimate causal effects.

## Approach

The analysis compares three types of results:

1. Observed charge-off rates from the portfolio analysis.
2. Rates adjusted for differences in approval vintage.
3. Estimates from a multivariable logistic regression.

The regression controls for approval vintage, industry, project state, business age, business type and processing method. Loan amount, original term, guarantee percentage and initial interest rate use flexible spline terms to capture non-linear relationships.

Adjusted risks are averaged across the analysis population, making them easier to compare with observed rates. Confidence intervals are included to show uncertainty, and industry and state comparisons are corrected for multiple testing.

These adjustments account only for information available in the SBA data. Important borrower and underwriting characteristics are not included.

## Summary of adjusted findings

| Factor                | Observed pattern                                                           | After adjustment                                                            |
| --------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Loan size             | Smaller loans had higher charge-off rates                                  | The direction reversed and remained non-linear                              |
| Original term         | Short-term loans had much higher rates                                     | A strong curved relationship remained                                       |
| Industry              | Several industries appeared clearly separated                              | Many differences became smaller                                             |
| Geography             | Florida and Texas were above the portfolio rate                            | Florida remained higher; the Texas difference weakened                      |
| Business age          | Startups and early-stage businesses had higher rates                       | Differences narrowed but remained                                           |
| Business type         | Individual businesses were higher and partnerships lower than corporations | The individual difference became small; the partnership difference remained |
| Processing method     | Pooled rates showed large differences                                      | Limited overlap and inconsistent time coverage weakened the comparison      |
| Guarantee percentage  | Some guarantee bands had higher rates                                      | A non-linear relationship remained                                          |
| Initial interest rate | Higher rates were associated with more charge-offs                         | A strong curved relationship remained                                       |

## Loan size

Loan size produced the clearest difference between the descriptive and adjusted results.

Observed charge-off rates were highest among smaller loans. After controlling for the available portfolio characteristics, estimated ten-year risk increased across the selected loan amounts:

| Gross approval | Adjusted risk |
| -------------: | ------------: |
|        $35,000 |         5.33% |
|       $113,700 |         6.86% |
|       $350,000 |         7.67% |
|       $965,000 |         8.24% |

This suggests that the higher observed rate among small loans was largely connected to other differences in the portfolio, including term, industry, processing method and guarantee percentage.

The adjusted relationship does not mean that larger approvals cause higher risk. It shows why observed loan-size bands should not be used as a simple risk rule.

## Original term

Original term remained the strongest adjusted relationship.

| Original term | Adjusted risk |
| ------------: | ------------: |
|     12 months |        50.08% |
|     60 months |        17.87% |
|     84 months |         4.98% |

The pattern was non-linear and remained visible when the outcome window was shortened from 120 to 96 months.

Part of this relationship may reflect differences in loan products and borrower profiles. The fixed observation window also covers the full life of a short loan but only part of a longer loan. For these reasons, term is treated as a useful risk indicator rather than a cause of charge-off.

## Interest rate and guarantee percentage

Initial interest rate retained a strong adjusted association with charge-off risk:

| Initial interest rate | Adjusted risk |
| --------------------: | ------------: |
|                 4.75% |         2.50% |
|                 6.00% |         7.22% |
|                 8.25% |        11.87% |

Interest rate may reflect risk pricing, economic conditions, lender practices and product structure. The analysis does not show that changing the interest rate would directly change the outcome.

Guarantee percentage also had a curved relationship with risk. Adjusted risk was 5.20% at a 50% guarantee and 8.73% at 75%. Estimates at 75% and 85% were nearly equal, so the results do not support using the original guarantee bands as fixed risk thresholds.

## Industry, geography and business characteristics

Several industry differences became smaller after adjustment. Sectors that appeared clearly separated in the descriptive analysis were not always statistically distinguishable after accounting for other portfolio characteristics.

Geographic differences also weakened. Florida remained above the comparison level, while Texas was no longer clearly different from California. State results should therefore be treated as portfolio context, not as a borrower-quality measure or a state risk ranking.

Startups and early-stage businesses remained associated with higher risk, although the differences narrowed. The difference between individual businesses and corporations became small, while partnerships continued to show a lower adjusted estimate.

## Processing method

Processing method showed large differences in pooled charge-off rates, but the comparison was not stable.

Some methods were available only during part of the analysis period, and comparable loan profiles were limited across methods. Most adjusted comparisons had wide uncertainty or included no clear difference.

For that reason, named processing-method rankings were excluded from the executive portfolio story and from the final model. This avoids presenting a visible historical pattern as a reliable risk relationship.

## Limitations

The analysis does not include credit scores, borrower financial statements, verified collateral values or detailed underwriting information. It is also based on only seven FY2010+ approval vintages with a complete ten-year observation window.

Because the dataset is large, small differences can be statistically significant without being practically important. The interpretation therefore focuses on the size, uncertainty and consistency of each relationship rather than p-values alone.

All adjusted estimates are statistical associations. They do not show that changing a loan characteristic would cause a change in charge-off risk.

The analysis can be reproduced with `src/analysis/run_statistical_analysis.py`. Aggregate results are stored in `reports/tables/statistics/`, and supporting figures are available in `reports/figures/statistics/`.
