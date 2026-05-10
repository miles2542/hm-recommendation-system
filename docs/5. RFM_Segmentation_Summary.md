# RFM Segmentation Summary

## 1. Pipeline Overview (What was done)
- **Data scope:** Built customer segmentation from **train_processed.parquet only** to enforce strict **no-leakage**.
- **Feature engineering (Polars):** Aggregated customer-level **RFM** from active training transactions.
- **Definitions:** **R = min(days_ago)**, **F = count(distinct t_dat trips)**, **M = sum(price * quantity)**.
- **Preprocessing:** Applied **Yeo-Johnson** to R/F/M, then **StandardScaler** to transformed R/F/M plus age.
- **Clustering:** Trained **KMeans (K=5)** after K-search diagnostics.
- **Business mapping:** Converted clusters to interpretable segments using **rank-based deterministic labeling** (Whales, Champions, At-Risk, Newcomers, Core + Young/Mature age prefix).

## 2. Reliability & Quality Metrics (Why trust this?)
- **Coverage proof:** **1,338,785** output rows, exactly matching **1,338,785** unique active train customers.
- **Integrity proof:** **0 nulls**, **0 NaNs**, **0 duplicate customer_id** in final artifact.
- **Method proof:** **Elbow + Silhouette** were computed on a **100k sample**; **K=5** was chosen for better business-actionable granularity versus the raw silhouette peak at **K=3**.
- **Memory optimization:** Export schema uses **customer_id Int32**, **cluster_id Int32**, **age Int8** to reduce footprint and keep downstream joins stable.

## 3. Segment Profiles & Results (The Output)
- **Young At-Risk** (Share: **23.85%** | Medians: **R=469, F=1, M=0.093153, Age=27** | Basket Size: **3.2977** items)  
  Description: Low-frequency, stale customers with very weak recent activity. Prioritize win-back triggers and high-CTR staples.
- **Young Whales** (Share: **23.36%** | Medians: **R=72, F=11, M=0.960407, Age=26** | Basket Size: **3.7883** items)  
  Description: Highest-value and most active buyers. Prioritize premium cross-sell and margin-maximizing bundles.
- **Mature Newcomers** (Share: **19.98%** | Medians: **R=391, F=1, M=0.101644, Age=53** | Basket Size: **2.9319** items)  
  Description: Low-history older customers with limited engagement depth. Use onboarding and age-relevant trend recommendations.
- **Young Core** (Share: **17.21%** | Medians: **R=97, F=2, M=0.155644, Age=25** | Basket Size: **2.9070** items)  
  Description: Broad mid-value base with moderate repeat behavior. Use frequency-lift offers and personalized replenishment.
- **Mature Champions** (Share: **15.60%** | Medians: **R=82, F=9, M=0.791822, Age=51** | Basket Size: **3.3420** items)  
  Description: Loyal, strong spenders with consistent activity. Target curated assortments and premium retention campaigns.

## 4. Downstream Usage (For RecSys / MBA)
- **RecSys:** Treat segments as policy layers: prioritize high-margin candidate generation for **Whales/Champions**, and trend-plus-demographic cold-start ranking for **Newcomers**.
- **MBA:** Use segment-level **avg basket size** to tune FP-Growth support thresholds; **Core needs higher support (e.g., 0.005)** while **Whales/Champions need lower support (e.g., 0.001)**.