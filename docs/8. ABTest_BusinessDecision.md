## **1. Experimental Design & Traffic Allocation**

### **1.1. Purpose & Goal of the Simulation**

While offline metrics (MAP@12, Hit Rate) prove historical mathematical accuracy, they do not guarantee future economic lift. We designed a **Randomized Controlled Trial (RCT)** simulation to definitively prove **Causality** - verifying that conversion and revenue uplifts are driven by algorithmic changes rather than macroeconomic or seasonal noise. This controlled environment provides a secure foundation for C-level deployment decisions without exposing the entire customer base to unproven risks.

### **1.2. Randomization Unit & Arm Assignment**

The **Randomization Unit** is the **User-Level (`customer_id`)**. We strictly avoid session-level randomization to prevent cross-session psychological contamination ("network bleed"). We utilize a **34/33/33 split** to ensure equal sample sizes and maximize Statistical Power ($1-\beta$) across a **3-Arm Setup**:

*   **Control Arm (34%):** MBA FP-Growth (Business-as-usual baseline).
*   **Treatment A (33%):** ALS Collaborative Filtering.
*   **Treatment B (33%):** LightGBM Learning-to-Rank.

### **1.3. Metric Definition & Hierarchy**

To avoid the **"ROI Trap"** - where a feature looks statistically significant but loses the company money — we enforce a strict hierarchical metric framework before observing any results. The table below outlines the definitions, business purposes, and the specific statistical tests we will apply to each.

| Metric | Name | Type | Business Purpose | Statistical Test |
| :---: | :---: | :---: | :---: | :---: |
| **Primary** | **Hit Rate@12** | Binary | Measures algorithmic conversion propensity (did the user buy *any* recommended item?). | **Pairwise Proportion Test** (with Bonferroni Correction). |
| **Secondary** | **Incremental Rec. Revenue (IRR)** | Continuous | Measures direct monetary value generated exclusively by algorithmic hits, proving true economic generation. | **Welch's T-Test** (`equal_var=False`). |
| **Guardrail** | **Sample Ratio Mismatch (SRM)** | Categorical | Ensures the automated traffic router mathematically achieved the intended 34/33/33 split. | **Chi-Square Test** (`scipy.stats.chisquare`). |

### **1.4. Analysis of Traffic Router Output**

The router processed 216,479 unique evaluation customers into: Control (73,586), Treatment A (71,496), and Treatment B (71,397). To ensure this empirical split lacks hashing bias, it must be mathematically validated via SRM evaluation.

## **2. Guardrail Validity Checks**

### **2.1. Sample Ratio Mismatch (SRM) Test**

A Randomized Controlled Trial (RCT) assumes confounding variables are neutralized through perfect random assignment. If the assignment mechanism fails (due to hashing bias or latency), the arms become disproportionate, triggering a Sample Ratio Mismatch (SRM) that invalidates downstream metrics. We utilize Pearson's Chi-Square Goodness-of-Fit Test to measure the divergence between observed and expected traffic volumes:

$$
\begin{cases} 
H_0: \text{The empirical user distribution matches the theoretical } 34/33/33 \text{ allocation.} \\ 
H_1: \text{The empirical distribution significantly deviates from expected proportions, indicating systemic bias.} 
\end{cases}
$$

The mathematical computation for the Chi-Square test statistic ($\chi^2$) is executed using the following formula:

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

Where the elements are defined as:

$$
\begin{cases} 
k: \text{Total number of experimental arms (3 in this architecture)} \\
O_i: \text{Observed count of unique users assigned to arm } i \\ 
E_i: \text{Expected count based on target probability } (N \times p_i) \\ 
N: \text{Total population size across all arms} 
\end{cases}
$$

A resulting P-value below the standard significance threshold ($\alpha = 0.05$) necessitates the immediate suspension of the experiment, as it proves the presence of SRM. Conversely, a P-value above 0.05 indicates insufficient evidence to reject the null hypothesis, thereby validating the integrity of the assignment router.

### **2.2. CUPED Applicability Assessment**

Continuous metrics like Incremental Recommendation Revenue (IRR) suffer from high variance ($\sigma^2$), which diminishes Statistical Power ($1 - \beta$). To mitigate this in live environments, Controlled-experiment Using Pre-Experiment Data (CUPED) leverages historical covariates (e.g., pre-experiment monetary value from RFM clusters) to absorb explainable noise:

$$
\hat{Y}_i^{CUPED} = Y_i - \theta(X_i - \bar{X})
$$

$$
\begin{cases} 
Y_i: \text{Observed continuous metric (IRR) for user } i \text{ during the test} \\ 
X_i: \text{Pre-experiment covariate for user } i \text{ (e.g., historical monetary average)} \\ 
\bar{X}: \text{Global mean of the pre-experiment covariate across all users} \\ 
\theta: \text{Optimal covariance multiplier calculated as } \left( \frac{\text{cov}(X,Y)}{\text{var}(X)} \right) 
\end{cases}
$$

However, given our exceptionally large static holdout sample ($n=216,479$), the Standard Error of the Mean (SEM) is natively suppressed. Therefore, mathematical CUPED transformation is bypassed in favor of raw economic evaluation via Welch's T-Test. We acknowledge CUPED as a mandatory architectural design for future live-environment variance reduction.

### **2.3. Analysis of the Chi-Square Guardrail Output**

The test yielded $\chi^2 = 0.0744$ and $p = 0.9635$. Because $p \gg \alpha$ (0.05), the null hypothesis is retained. The router is mathematically validated and free of hashing bias, permitting causal statistical evaluation.

## **3. Formal Hypothesis Statement**

### **3.1. Primary Metric Hypotheses (Hit Rate Propensity)**

Evaluates if advanced architectures increase conversion probability compared to baseline rules.

$$H_0: p_{treatment} - p_{control} \le 0 \quad \text{vs.} \quad H_1: p_{treatment} - p_{control} > 0$$

*(Where $p$ = proportion of converting users; $\alpha = 0.05$; Power $1 - \beta = 0.80$)*

### **3.2. Secondary Metric Hypotheses (Incremental Recommendation Revenue)**

Evaluates the continuous monetary accumulation derived exclusively from successful recommendations.

$$H_0: \mu_{treatment} - \mu_{control} \le 0 \quad \text{vs.} \quad H_1: \mu_{treatment} - \mu_{control} > 0$$

*(Where $\mu$ = Mean IRR per user in the respective arm)*

## **4. Statistical Evaluation Methodology**

### **4.1. Methodological Framework (Theory & Logic)**

#### **4.1.1. Primary Metric: Pairwise Proportion Testing & Bonferroni Correction**

Because the Control Arm is concurrently tested against two independent variants (ALS and LightGBM), the Family-Wise Error Rate (FWER) inflates. To maintain strict mathematical rigor, the Bonferroni Correction is applied. We divide the global significance level ($\alpha = 0.05$) by $m=2$ comparative pairs to secure the 95% confidence boundary.

Consequently, the adjusted significance threshold for the primary metric becomes $\alpha_{adj} = 0.025$. A treatment is only deemed statistically superior if its resulting P-value falls below this penalized boundary. The evaluation utilizes the two-sample Z-test for proportions, structured as:

$$
Z = \frac{\hat{p}_t - \hat{p}_c}{\sqrt{\hat{p}_{pool}(1 - \hat{p}_{pool}) \left(\frac{1}{n_t} + \frac{1}{n_c}\right)}}
$$

To satisfy C-level reporting requirements, the exact magnitude of the algorithmic uplift must be quantified using a Confidence Interval (CI) representing the delta ($\Delta$) between the proportions:

$$
CI_{\Delta} = (\hat{p}_t - \hat{p}_c) \pm Z_{1-\alpha_{adj}/2} \sqrt{\frac{\hat{p}_t(1-\hat{p}_t)}{n_t} + \frac{\hat{p}_c(1-\hat{p}_c)}{n_c}}
$$

*Note: A two-sided 95% Confidence Interval is reported alongside the one-tailed hypothesis test as standard practice for executive-facing reporting. The CI quantifies the plausible range of the uplift delta, while the one-tailed P-value evaluates whether that delta is directionally positive. Both are mathematically valid and mutually complementary.*

#### **4.1.2. Secondary Metric: Welch's T-Test and Effect Size**

The continuous secondary metric (IRR) violates the equal variance assumption (homoscedasticity) required by a traditional Student’s T-Test. Fashion retail data is heavily right-skewed by high-frequency "Whale" consumers generating asymmetrical monetary density. To mathematically correct for this variance and differing sample sizes across arms, **Welch’s T-Test** is mandated:

$$t = \frac{\bar{X}_t - \bar{X}_c}{\sqrt{\frac{s_t^2}{n_t} + \frac{s_c^2}{n_c}}}$$

While the resulting P-value confirms statistical significance, it does not convey the magnitude of the economic shift. In an A/B test exhibiting unequal variances, pooling the standard deviations creates a mathematically biased effect size. Therefore, **Glass's Delta ($\Delta$)** is utilized instead of Cohen's $d$. Glass's Delta divides the mean difference strictly by the Control Group's standard deviation ($s_c$), anchoring the effect size to the stable baseline reference:

$$\Delta_{Glass} = \frac{\bar{X}_t - \bar{X}_c}{s_c}$$

### **4.2. Pre-Execution Artifact Analysis**
#### **4.2.1. Data Architecture & Schema Analysis**

Prior to executing the statistical engine, the physical structure of the experimental artifacts must be validated to ensure deterministic relational joins. The schema extraction yields the following architectural footprint across the evaluation environment:

| Artifact | Rows | Primary Keys | Target Value | Purpose / Status |
| :--- | :--- | :--- | :--- | :--- |
| `ab_traffic_allocation` | 216,479 | `customer_id` (Int32) | `experimental_arm` | Master router. **(Ready)** |
| `test_processed` | 913,103 | `customer_id`, `article_id` | `price` (Float64) | Ground truth. **(Ready)** |
| `ALS_predictions` | 2.59M | `customer_id`, `article_id` | `rank` (Int64) | Trt A vectors. **(Requires Int32 Downcast)** |
| `lgbm_test_features` | 17.0M | `customer_id`, `article_id` | `label`, `features` | Unscored pairs. **(Deprecated)** |
| `articles_processed` | 105K | `article_id` (Int32) | `index_name` | Master catalog. **(Ready)** |
| `MBA_predictions` | - | `customer_id`, `article_id` | `rank` (Int8) | Control vectors. **(Generate via Export)** |
| `lgbm_predictions` | - | `customer_id`, `article_id` | `rank` (Int8) | Trt B vectors. **(Generate via Export)** |

#### **4.2.2. Architectural Normalization & Artifact Export**

Analysis reveals three critical dependencies required before statistical execution:

1.  **ALS Type Misalignment:** `ALS_predictions` uses `Int64`. Requires downcast to `Int32` during ingestion to prevent memory bloat.

2.  **Missing LightGBM Ranks:** `lgbm_test_features` lacks Top-$K$ ordering.

3.  **Dynamic MBA Overload:** Computing MBA rules concurrently during the statistical loop causes RAM failure.

$\implies$ **Standardization Protocol:** We append new code blocks to Phase 2 notebooks to export `MBA_predictions.parquet` and `lgbm_predictions.parquet`. All three predictive arms are standardized to an identical, compressed schema: `[customer_id (Int32), article_id (Int32), rank (Int8)]`.

### **4.3. Simulated RCT Execution & Statistical Inference**

#### **4.3.1. Statistical Engine Implementation**

To definitively evaluate the algorithmic uplift, the statistical engine will fuse the master assignment router, the unified predictive artifacts, and the historical ground truth. 

**Data Fusion & Evaluation Logic**

1.  **Isolate & Map:** The prediction artifacts (`MBA`, `ALS`, `LightGBM`) are strictly filtered to contain only the users assigned to their respective experimental arms (Control, Treatment A, Treatment B).
2.  **Ground Truth Intersection:** The unified prediction matrix is joined against `test_processed.parquet` using `[customer_id, article_id]` as the composite key. A successful intersection represents an empirical "Hit."
3.  **User-Level Aggregation:** 
    *   **Hit Rate Propensity (Binary):** A user is assigned a `1` if the intersection yields $\ge 1$ item, otherwise `0`.
    *   **Price-Weighted Hit Score (Continuous):** The `price` column from the ground truth is summed for all intersecting items per user. Non-converting users receive a strict `0.0`.

**Mathematical Formulation (Programmatic Implementation)**

1. **Primary Metric (Proportion Z-Test):**

To adhere to the Bonferroni Correction ($\alpha_{adj} = 0.025$), the critical Z-value for the 95% Confidence Interval is calculated via the inverse Cumulative Distribution Function (CDF): 

$$Z_{crit} = \Phi^{-1}(1 - \alpha_{adj}/2) \approx 2.2414$$

The Confidence Interval for the delta ($\Delta = \hat{p}_{treatment} - \hat{p}_{control}$) is computed using the unpooled standard error:

$$SE_{\Delta} = \sqrt{\frac{\hat{p}_t(1-\hat{p}_t)}{n_t} + \frac{\hat{p}_c(1-\hat{p}_c)}{n_c}}$$

$$CI = \Delta \pm (Z_{crit} \times SE_{\Delta})$$

2. **Secondary Metric (Welch's T-Test & Cohen's d):**
The continuous arrays are evaluated using `scipy.stats.ttest_ind(equal_var=False, alternative='greater')`. To measure standard deviation shifts independent of sample size, **Cohen's $d$** is computed:

$$d = \frac{\bar{X}_t - \bar{X}_c}{\sqrt{\frac{s_t^2 + s_c^2}{2}}}$$


#### **4.3.2. Empirical Outcome Analysis**

The statistical engine successfully evaluated the simulated interactions of 216,479 unique test subjects against the historical ground truth. To ensure maximum analytical clarity, the empirical outputs are aggregated into three distinct evaluation matrices: separating the probability of conversion, the magnitude of economic generation, and the stratified behavioral variance across user clusters.

##### **4.3.2.1. Statistical Aggregation Matrices**

**Table 1: Primary Metric – Hit Rate Propensity (Binary Conversion)**

*Evaluation Threshold: $\alpha_{adj} = 0.025$ (One-Tailed Z-Test)*

| Experimental Arm | Sample Size ($N$) | Hit Rate | Absolute Uplift ($\Delta$) | Relative Uplift | Z-Score | P-Value | 95% Confidence Interval | Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** (MBA) | 73,586 | 4.3881% | — | — | — | — | — | Baseline |
| **Treatment A** (ALS) | 71,496 | 4.7989% | +0.4108% | +9.36% | 3.7380 | $9.27 \times 10^{-5}$ | $[0.1643\%, 0.6573\%]$ | **Significant** |
| **Treatment B** (LGBM) | 71,397 | 7.5073% | +3.1193% | +71.09% | 25.1523 | $6.66 \times 10^{-140}$ | $[2.8409\%, 3.3976\%]$ | **Significant** |

**Table 2: Secondary Metric – Incremental Recommendation Revenue (IRR)**

*Evaluation Threshold: $\alpha_{adj} = 0.025$ (Welch's T-Test)*

| Experimental Arm | Mean IRR | Std. Deviation | Absolute Shift | Welch's T-Score | P-Value | Glass's $\Delta$ | Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control** (MBA) | 0.001360 | 0.007174 | — | — | — | — | Baseline |
| **Treatment A** (ALS) | 0.001439 | 0.007897 | +0.000079 | 1.9874 | $2.34 \times 10^{-2}$ | 0.0110 | **Significant** |
| **Treatment B** (LGBM) | 0.002671 | 0.010267 | +0.001312 | 28.1194 | $9.66 \times 10^{-174}$| 0.1828 | **Significant** |

**Table 3: Segment-Stratified Empirical Hit Rates**

*Analysis of variance across pre-defined RFM clusters.*

| Segment Label | Control Hit Rate | Treatment A (ALS) | ALS Relative Δ | Treatment B (LGBM) | LGBM Relative Δ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Mature Champions** | 4.3359% | 5.0625% | **+16.76%** | 7.9483% | **+83.31%** |
| **Mature Newcomers** | 4.7848% | 3.8701% | **−19.11%** | 6.5616% | **+37.13%** |
| **Young At-Risk** | 4.4572% | 2.9616% | **−33.55%** | 6.1375% | **+37.69%** |
| **Young Core** | 4.2204% | 4.6514% | **+10.21%** | 6.4903% | **+53.78%** |
| **Young Whales** | 4.4231% | 4.9566% | **+12.06%** | 7.7705% | **+75.68%** |


##### **4.3.2.2. Analytical Interpretations & Statistical Implications**

**I. Treatment A (ALS) Paradigm: Quantifying "ROI Trap"**

The empirical data from Treatment A perfectly encapsulates the **"ROI Trap"** - a scenario where a statistical test yields mathematical significance but fails to deliver material economic utility. For the **Primary Metric**, the **Collaborative Filtering** architecture successfully utilized **Latent Factorization** to drive a **Relative Uplift** of 9.36% in conversion propensity, achieving a definitive **Z-Score** of 3.7380. 

Critically, following deduplication normalization, ALS also managed to clear the **Bonferroni-corrected threshold** ($\alpha_{adj} = 0.025$) for the **Secondary Metric**, returning a **Welch's T-Test P-Value** of $0.0234$. However, despite achieving formal statistical significance across both axes, the economic magnitude is practically negligible. The **Absolute IRR Shift** (+0.000079) equates to a microscopic **Glass's Delta** ($\Delta = 0.0110$). 

Unlike the LightGBM model which utilized **Expected Cross-Sell Revenue (ECR)** pricing thresholds and **Wallet-Matching**, the pure Matrix Factorization of ALS blindly optimized for structural co-occurrence. This made it fatally susceptible to **Popularity Bias**. It marginally increased the *frequency* of purchases (Hit Rate), but cannibalized margin density by recommending cheaper, high-velocity basics, proving that mathematical accuracy in a latent space does not natively guarantee increased basket scale.

**II. Simpson's Paradox and Segment Harm**

An examination of **Table 3** reveals a critical systemic flaw within the ALS architecture. The aggregate +9.36% Hit Rate uplift reported in Table 1 is mathematically misleading - it is an artifact of **Simpson's Paradox**. 

While ALS generated positive lifts for high-frequency segments like **Mature Champions** (+16.76%) and **Young Whales** (+12.06%), it actively destroyed value for dormant users. **Young At-Risk** users subjected to ALS converted at a rate **33.55% lower** than the MBA Control, and **Mature Newcomers** dropped by **19.11%**. Mechanically, Collaborative Filtering fails on lapsed/new users because their sparse interaction history is insufficient to anchor meaningful latent vectors. Consequently, ALS feeds them generic popularity lists that perform markedly worse than the demographically-aware **Association Rules** utilized by the Control arm.

**III. Treatment B (LightGBM) Dominance: Universal Optimization**

The performance of Treatment B establishes **Gradient Boosting** as the unequivocally superior algorithmic architecture. On the **Primary Metric**, LightGBM achieved an astronomical **Z-Score** of 25.1523, equating to a **P-value** approaching absolute zero ($6.66 \times 10^{-140}$). The algorithm increased the conversion probability by a staggering **Relative Uplift** of 71.09%, with a tightly constrained **95% Confidence Interval** that operates entirely above a 2.8% absolute gain.

Crucially, LightGBM successfully cleared the dual-metric optimization hurdle while demonstrating universal segment superiority. The **Secondary Metric** demonstrates that the **Incremental Recommendation Revenue (IRR)** effectively doubled (from 0.001360 to 0.002671). The **Welch's T-Score** of 28.1194 mathematically confirms this systemic shift. The calculated **Glass's Delta** ($\Delta = 0.1828$) represents a highly tangible, operationally impactful effect size, proving the revenue shift scales robustly relative to the control group's baseline variance. 

**IV. Validation of the Anti-Leakage Methodology**

These statistical milestones must be contextualized within the **Anti-Join Methodology** enforced during Phase 2. Because trivial repurchases were strictly barred from the candidate sets, the calculated **Uplifts** represent 100% pure **Novel Discovery**. The evaluation proves that advanced, multi-dimensional feature matrices possess the operational capacity to stimulate genuine, previously dormant consumer demand without relying on historical repurchasing crutches.

## **5. Business Impact & Deployment Decision**

### **5.1. ROI Trap Analysis & Annualized Revenue Projection**

The primary function of a Data-Driven Marketing strategy is to maximize the **Return on Investment (ROI)**. The **"ROI Trap"** occurs when an organization deploys a statistically significant algorithm whose computational infrastructure costs vastly exceed its marginal revenue generation. To make an objective deployment decision, the empirical **Absolute IRR Shift** calculated in Section 4 must be projected into an annualized economic forecast.

The simulation evaluated an active cohort of $216,479$ users over a strictly bounded 28-day evaluation window. By utilizing the total segmented customer base ($1,338,785$ users) mapped during the RFM classification phase, the **Annualized Incremental Revenue** is forecasted utilizing the following scaling formula:

$$
\text{Projected Annual Incremental Revenue} = \left( \Delta_{\mu} \times N_{Total} \right) \times \left( \frac{365}{28} \right)
$$

Where elements are defined as:

*   $\Delta_{\mu}$: **Absolute IRR Shift** (Continuous Incremental Recommendation Revenue per user).
*   $N_{Total}$: **Total Addressable Market** ($1.33$ million active users).
*   $365 / 28$: **Temporal Annualization Multiplier** ($\approx 13.035$).

**Table 4: Annualized Economic Projection**

| Algorithmic Architecture | Absolute IRR Shift ($\Delta_{\mu}$) | Statistical Validity | Projected Annual Incremental Yield | Computational Inference Cost |
| :---: | :---: | :---: | :---: | :---: |
| **ALS (Collaborative Filtering)** | +0.000079 | **Passed** ($p < \alpha_{adj}$) | +1,379 NRU | Medium (Latent Matrix Multiplication) |
| **LightGBM (Learning-to-Rank)** | +0.001312 | **Passed** ($p \ll \alpha_{adj}$) | +22,897 NRU | High (Multi-Dimensional Feature Trees) |

*Note: Monetary values are expressed in Normalized Revenue Units (NRU), corresponding to the normalized price scale [0, 0.591] used in the retail dataset. Conversion to absolute currency requires the original dataset normalization scalar. For executive reporting, the directional ratio (LightGBM generating ~16.6× the yield of ALS) remains economically valid regardless of unit scale.*

**Inference Cost Anchor:**

To complete the ROI evaluation, the projected annual incremental yield of +22,897 NRU must be weighed against LightGBM's inference costs. In production, this is measured in cloud compute units per batch scoring run (e.g., GPU/CPU-hour pricing per million candidate pairs). The scale of the yield differential - 22,897 NRU for LightGBM versus ALS's negligible 1,379 NRU - provides a massive economic margin of safety. Deployment is strictly justified provided that monthly inference infrastructure costs remain below the annualized yield divided by 12 ($\approx 1,908$ NRU/month). This threshold must be validated against live cloud billing data prior to enterprise-wide scaling.


### **5.2. Segment-Stratified Interpretation (RFM Lens)**

To understand *why* LightGBM overwhelmingly succeeded where ALS triggered the ROI Trap, the empirical metrics must be viewed through the RFM Cluster definitions established in Phase 1. 

**The ALS Simpson's Paradox:**

The aggregate Hit Rate uplift (+9.36%) produced by ALS masks severe structural failures. As demonstrated in the Segment Stratification matrix, ALS's aggregate positive performance is entirely driven by high-frequency segments (Mature Champions: +16.76%, Young Whales: +12.06%). Shockingly, ALS actively harmed dormant segments: **Young At-Risk** users converted at a rate **33.55% lower** than the Control arm, and **Mature Newcomers** dropped by **19.11%**. 

This phenomenon is caused by **Popularity Bias**. Because Collaborative Filtering treats all interactions uniformly within a latent matrix, it struggles with cold or lapsed users whose sparse history cannot anchor meaningful vectors. Consequently, ALS feeds them generic, cheap basics that fail to convert. It blindly optimizes for structural co-occurrence, entirely missing the margin constraints of the consumer.

**The LightGBM Prescriptive Personalization:**

Conversely, LightGBM generated positive uplifts in *every single segment*, effectively solving the sparse-history deficit. By incorporating **Wallet Matching** (`user_avg_price`) and **Expected Cross-Sell Revenue (ECR)** thresholds directly into the decision trees, the ranker actively optimized price acceptance:

*   **The "Whale" Optimization:** For **Young Whales** (who represent 54.35% of total revenue), LightGBM recognized their historically high `user_avg_price` and algorithmically filtered out low-margin basics, driving an astonishing 75.68% Hit Rate uplift while pushing the mean IRR to 0.002836.
*   **The "At-Risk" Reactivation:** For **Young At-Risk** consumers, LightGBM successfully avoided "sticker shock" by downgrading premium candidates, routing demographically appropriate items that secured a +37.69% Hit Rate reversal over the Control baseline.

### **5.3. C-Level Prescriptive Strategy & Deployment Mandate**

Based on the statistical, economic, and segment-stratified evidence compiled throughout this Randomized Controlled Trial, the following strategic mandates are issued:

#### **5.3.1. Deprecate ALS & Transition to Supervised Learning-to-Rank**

The organization must immediately deprecate purely implicit matrix factorization (ALS) as a primary ranking engine. ALS successfully triggers interaction density but actively harms margin density and misfires on sparse-history users. The retail catalog exhibits a 90-day shelf life, mandating the use of Supervised Learning-to-Rank architectures capable of utilizing dynamic, multi-dimensional pricing and demographic features.

#### **5.3.2. Enforce "Novel Discovery" Metrics System-Wide**

The success of the LightGBM model was predicated on an **Anti-Join Architecture** that strictly blocked trivial repurchases during candidate generation. Marketing KPIs must reflect this paradigm shift. Replace Gross Hit Rate with **Net-New Hit Rate**, and substitute standard AOV with **Incremental Recommendation Revenue (IRR)**. These metrics measure true demand generation rather than artificially inflated repurchase tracking.

#### **5.3.3. Deploy Dual-Engine Prescriptive Logic**

The final production environment will utilize a bipartite algorithmic approach tailored to computational latency constraints:

*   **Primary Navigation & Homepage (LightGBM):** Route traffic through the LightGBM L2R engine to govern the largest IRR opportunity. **Execution Protocol:** Initiate a **20% Phased Rollout**. Allocate 20% of live homepage traffic to LightGBM while monitoring real-time guardrail metrics (page latency, cart-to-checkout drop rate, and app crash rate) for a minimum of 14 days (capturing two full weekly seasonality cycles). Execute a 100% rollout only upon guardrail clearance.

*   **Shopping Cart & Checkout (MBA Baseline):** Retain the MBA FP-Growth Baseline utilizing the Expected Cross-Sell Revenue (ECR) heuristic. Because LightGBM requires heavy batch feature engineering, it is ideal for asynchronous Homepage generation. Conversely, MBA Association Rules (N-to-1) execute via instant $O(1)$ dictionary lookups, making them the computationally optimal choice for real-time, impulse-buy generation at checkout.