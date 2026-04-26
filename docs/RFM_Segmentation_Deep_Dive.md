# RFM Segmentation Deep Dive

## 1) Economic Profile of the Cluster System

### Portfolio-level structure

- Customer base segmented: **1,338,785**
- Segment architecture: **Young Whales, Mature Champions, Young At-Risk, Mature Newcomers, Young Core**
- Revenue concentration is asymmetric by design and is the main economic signal captured by the model.

### Cluster profiles (in-depth)

#### Young Whales

- **Scale:** 296,078 customers (**22.12%**)
- **Behavioral medians:** R=72, F=37, M=1.036119, Age=27
- **Basket intensity:** 1.1079
- **Revenue contribution:** **55.05%**
- **Interpretation:** This is the principal profit engine. High frequency and high spend indicate low friction in repeat purchase behavior and high acceptance for premium baskets.
- **Operating implication:** Treat as a high-value exploitation segment; optimize for margin density, assortment depth, and attachment value per session.

#### Mature Champions

- **Scale:** 207,780 customers (**15.52%**)
- **Behavioral medians:** R=84, F=26, M=0.779186, Age=51
- **Basket intensity:** 1.1033
- **Revenue contribution:** **30.19%**
- **Interpretation:** High-value loyal cohort with stable cadence and lower volatility than Whales.
- **Operating implication:** Retention quality is more important than novelty pressure; emphasize repeatable relevance, premium essentials, and churn prevention at low incentive cost.

#### Young At-Risk

- **Scale:** 324,677 customers (**24.25%**)
- **Behavioral medians:** R=460, F=3, M=0.086373, Age=27
- **Basket intensity:** 1.0994
- **Revenue contribution:** **4.82%**
- **Interpretation:** Largest segment by headcount but weak by monetization due to extreme recency decay.
- **Operating implication:** Treat as reactivation inventory, not revenue core; use conversion-safe items and controlled promotion depth.

#### Mature Newcomers

- **Scale:** 268,604 customers (**20.06%**)
- **Behavioral medians:** R=385, F=3, M=0.101627, Age=53
- **Basket intensity:** 1.0941
- **Revenue contribution:** **4.17%**
- **Interpretation:** Low-history segment with limited behavioral footprint and delayed recurrence.
- **Operating implication:** Prioritize certainty-first recommendations (demographic-fit popularity, low-risk combinations) until interaction density is sufficient.

#### Young Core

- **Scale:** 241,646 customers (**18.05%**)
- **Behavioral medians:** R=94, F=7, M=0.186390, Age=25
- **Basket intensity:** 1.0770
- **Revenue contribution:** **5.77%**
- **Interpretation:** Mid-intensity demand baseline; neither premium-heavy nor churn-dominant.
- **Operating implication:** Main segment for incremental frequency uplift and cross-sell breadth optimization.

## 2) Validity Proof from Notebook-Consistent Results

### A. Data integrity proof

- Segmented rows: **1,338,785**
- Active train customers: **1,338,785**
- Duplicate customer_id: **0**
- Null values: **0**
- NaN values (R/F/M/age): **0**
- Export schema integrity: customer_id Int32, cluster_id Int32, age Int8

### B. Economic validity proof

- **Pareto test:** Whales + Champions represent **37.64%** of customers and **85.23%** of total monetary value.
- **Value concentration lift:** **2.2647x** revenue share versus population share.
- **Implication:** Clusters are not balanced by count; they are intentionally differentiated by economic utility.

### C. Distributional separation proof

- Monetary Q75 for lower-value pool (Core + Newcomers): **0.241492**
- Monetary Q25 for high-value pool (Whales + Champions): **0.589000**
- Absolute boundary gap: **0.347508**
- Boundary ratio (Q25 high / Q75 low): **2.4390x**
- **Implication:** High-value and lower-value monetary bands are materially separated; this supports segment-specific policy without major overlap risk.

### D. K-selection evidence (same pipeline method)

| K | WCSS | Silhouette |
|---|------:|-----------:|
| 3 | 171475.58 | 0.3383 |
| 4 | 143117.26 | 0.3082 |
| 5 | 120434.76 | 0.3029 |
| 6 | 103655.86 | 0.3141 |
| 7 | 92798.19 | 0.3111 |
| 8 | 84271.71 | 0.2952 |
| 9 | 76918.01 | 0.3001 |
| 10 | 71188.98 | 0.2961 |

Interpretation:

- Maximum silhouette appears at K=3, but K=5 is retained for decision granularity across economically distinct intervention groups.
- This is a controlled business trade-off, not a methodological defect.

## 3) Deployment Logic for RecSys and MBA

### Recommendation system routing

- **Whales:** maximize expected margin per session; prefer premium attach candidates and high-value complements.
- **Champions:** maximize retention-weighted CLV; emphasize relevance stability and repeat-quality ranking.
- **At-Risk:** maximize reactivation probability under strict discount governance; prioritize historically high-conversion low-friction SKUs.
- **Newcomers:** run age-conditioned popularity priors until sufficient interaction depth unlocks collaborative ranking.
- **Core:** optimize frequency and category expansion with moderate exploration budget.

### FP-Growth operating policy

- **Whales/Champions:** lower min_support for niche premium bundles; enforce higher confidence/lift floor to avoid noisy luxury-tail rules.
- **Core:** use higher support thresholds for scalable baseline rules.
- **At-Risk/Newcomers:** use conservative confidence rules oriented to reliability and immediate conversion, not novelty.

## 4) Final Assessment

- The cluster system is **economically coherent**, **distributionally separated**, and **operationally deployable**.
- For internal objectives, the strongest evidence is revenue concentration and quantile boundary separation, both of which are decisively positive in the current result set.