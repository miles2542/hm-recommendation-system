# Preprocessing Implementation

## Available Datasets (Artifacts)
The `data/processed/` directory contains the final optimized artifacts required for downstream pipelines (RFM analysis, clustering, and RecSys modeling):

*   **`train_processed.parquet`**: Chronological training set (`t_dat < 2020-08-25`). Use for model training, interaction history, and RFM calculation.
*   **`test_processed.parquet`**: Chronological evaluation set (Final 28 days). Use strictly for RecSys validation and hold-out metrics.
*   **`customers_processed.parquet`**: Cleansed demographic and loyalty metadata. Use for user embeddings and clustering.
*   **`articles_processed.parquet`**: Master product catalog. Use for content-based feature extraction.
*   **`id_mapping.json`**: Translation dictionary bridging original Hex/String IDs to contiguous `Int32` integers. Retained for deployment inference; not required for offline evaluation pipeline.

---

## Pipeline Execution Summary
- **Original Transactions**: 31,788,324 rows
- **Final Transactions (Train + Test)**: 28,793,010 rows (~3M rows collapsed/purged)
- **Train Set Size**: 27,803,789 interactions
- **Test Set Size**: 989,221 interactions

---

## Transformation Steps (Leakage-Proofed)

### 1. Temporal Train/Test Split (Primary Step)
*   **Action**: Partitioned the timeline using the global max date minus 28 days.
*   **Purpose**: Prevent data leakage (future predicting past). All subsequent filtering and imputation logic (Resellers, Age Imputation) are computed strictly from the Training segment.
*   **Impact**: Ensures model evaluation reflects a realistic production scenario without hindsight bias.

### 2. Memory Optimization & Deterministic ID Mapping
*   **Action**: Mapped `customer_id` and `article_id` to contiguous zero-indexed `Int32`. Applied lexicographical sorting prior to indexing.
*   **Purpose**: Reduce RAM footprint and ensure IDs remain stable and deterministic across pipeline reruns.
*   **Impact**: Facilitates reproducible model checkpoints and compatible sparse matrix lookups.

### 3. Transaction Deduplication & Aggregation
*   **Action**: Grouped transactions by `[t_dat, customer_id, article_id, price, sales_channel_id]` and aggregated using a row count.
*   **Purpose**: Resolve the 17.36% exact-duplicate anomaly (representing multi-unit purchases).
*   **Impact**: Preserves interaction strength through the new `quantity` feature while eliminating redundant target rows.

### 4. Reseller / Outlier Purge
*   **Action**: Filtered out customers with a **Training-set** lifetime `quantity > 1000`.
*   **Purpose**: Remove wholesale/reseller accounts that skew collaborative filtering via popularity bias.
*   **Impact**: Surgically removed extreme outliers while preserving high-engagement "whale" consumers.

### 5. RecSys Feature Engineering (Time Decay)
*   **Action**: Engineered `days_ago` relative to the **Training Max Date**, and computed `time_weight = exp(-0.0231 * days_ago)`.
*   **Purpose**: Provide sequential weighting. A rate of `-0.0231` assigns a precise 30-day half-life.
*   **Impact**: Anchoring to the training max date ensures the freshest training interactions maintain a weight of 1.0, rather than being pre-decayed relative to the test window.

### 6. Customer Profiling & Affinitive Imputation
*   **Action**: 
    *   **Binary Flags**: Imputed `FN` and `Active` nulls with `0` (Int8).
    *   **Geo-Masking**: Overwrote the 626k-transaction anomaly code (`2c29ae65...`) to `Unknown`.
    *   **Affinitive Age Imputation**: Imputed missing age using the most purchased product index calculated strictly from **Training-set** history.
*   **Purpose**: Ensure a dense feature matrix without introducing future-leakage into static demographics.
*   **Impact**: Maintains the bimodal demographic integrity discovered during EDA.
