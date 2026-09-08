# Portfolio findings

## Portfolio and vintage context

The analysis covers 1.96 million SBA 7(a) loan records from FY1991 through June 2026. Approval volume and exposure use the full history. Charge-off comparisons use only loans with a complete 120-month observation window.

The clearest pattern is the change across approval vintages. The ten-year charge-off rate increased from 11.47% for FY2002 approvals to 35.78% for FY2007, before falling to 8.74% in FY2010. Rates for FY2011 through FY2016 ranged from 5.60% to 6.92%.

This variation matters because results from the financial-crisis period do not necessarily represent the later portfolio.

| Eligible population |   Loans | Charge-offs within 120 months | Charge-off rate | Gross approval |
| ------------------- | ------: | ----------------------------: | --------------: | -------------: |
| FY2001 onward       | 878,511 |                       150,609 |          17.14% |      $206.22bn |
| FY2010 onward       | 308,134 |                        20,299 |           6.59% |      $107.87bn |

These figures describe historical portfolio performance. They do not prove that programme changes caused the difference between the two periods.

## Risk frequency and exposure

Charge-off frequency and approved-dollar exposure identify different monitoring priorities.

Within the FY2010+ population, loans up to $50,000 had the highest observed charge-off rate at 8.00%. Loans above $1 million had the lowest observed rate at 3.92%, but represented 51.36% of total gross approvals. Loans up to $50,000 accounted for only 2.82% of gross approvals.

Smaller loans therefore produced more charge-offs relative to their number, while larger loans represented the main concentration of approved dollars. However, the higher observed rate among small loans did not remain after controlling for vintage and other portfolio characteristics. The results do not support the simple conclusion that small loans are independently riskier.

## Original term

Original term produced the strongest descriptive separation in charge-off rates.

For FY2010+ loans, the observed ten-year rate was 31.60% for terms of 1 to 12 months and 16.09% for terms of 13 to 60 months. Rates for valid term groups above 60 months ranged from 1.36% to 4.83%.

This pattern requires careful interpretation. Loan term may reflect different products, purposes and borrower profiles. The fixed ten-year outcome window also covers the full contractual life of a short loan but only part of a longer loan. Term is useful for portfolio segmentation and prediction, but the analysis does not show that term causes charge-off.

## Industry and geography

Industry results combined risk frequency with exposure concentration. Among supported two-digit NAICS groups, sectors 45, 71, 48 and 72 had comparatively high observed charge-off rates. Accommodation and food services, sector 72, was also the largest industry exposure, representing 17.54% of gross approvals.

Geographic patterns were also mixed. Florida had an observed charge-off rate of 9.02% and Texas 8.33%, compared with 6.59% for the FY2010+ portfolio. California had the largest exposure share at 16.57%.

Some state differences became smaller after statistical adjustment. Geography should therefore be used as supporting portfolio context, not as a borrower-quality measure or a state risk ranking.

## Business characteristics

Startups and early-stage businesses had observed charge-off rates of 8.18% and 8.02%, compared with 5.66% for existing businesses. These differences narrowed after adjustment but remained relevant to the portfolio story.

Business type showed smaller differences. Corporations had an observed rate of 6.54%, compared with 7.38% for individual businesses and 3.68% for partnerships. After adjustment, the individual-versus-corporation difference became small, while the partnership difference remained.

## Interest rate and guarantee percentage

Observed charge-off rates generally increased across the main initial-interest-rate bands, from 2.88% for rates up to 5% to 10.20% for rates above 7.5% through 10%. Guarantee percentage showed a non-linear relationship with risk.

These fields may reflect product design, programme rules, economic conditions and selection at approval. They provide useful predictive information, but they should not be interpreted as causal effects.

## Recorded charge-off amounts

Recorded `GrossChargeOffAmount` represented 2.82% of gross approvals in the FY2010+ fixed-horizon population. Excluding 647 records flagged for financial anomalies reduced the ratio to 2.69%, without changing the main exposure patterns.

This field is used only as a supporting measure. It is not described as SBA loss, lender loss or taxpayer loss because the published data does not support those interpretations.

## Main takeaways

* Approval vintage strongly affects the observed risk picture.
* Charge-off frequency and approved-dollar exposure highlight different portfolio concerns.
* Descriptive differences can weaken or change after statistical adjustment.
* Original term provides important monitoring information, but its relationship with the fixed outcome window must be considered.
* Segment results describe historical associations, not causes.

Supporting charts are available in `reports/figures/portfolio/`. The aggregate data behind them is stored in `reports/tables/portfolio/`.
