# Feature Dictionary: Processed Datasets

This dictionary defines the features available in the optimized Parquet artifacts located in `data/processed/`. These files have been cleansed, imputed, and optimized for downstream modeling (RecSys, RFM, and Clustering).

## 1. Interaction Data (`train_processed.parquet` / `test_processed.parquet`)
*   **Purpose**: Transactional logs with interaction strength and temporal weighting.
*   **Split**: `train` (interactions before 2020-08-25), `test` (interactions from 2020-08-25 to 2020-09-22).

| Feature | Type | Null % | Range / Uniques | Sample | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `t_dat` | Date | 0.00% | 734 unique days | 2018-09-20 | Transaction Date |
| `customer_id` | Int32 | 0.00% | [0, 1,371,979] | 512 | Mapped Customer ID (Deterministic) |
| `article_id` | Int32 | 0.00% | [0, 105,541] | 2048 | Mapped Article ID (Deterministic) |
| `price` | Float64 | 0.00% | [0.00001, 0.591] | 0.0508 | Normalized Price |
| `sales_channel_id`| Int8 | 0.00% | [1, 2] | 2 | Channel (1: Online, 2: Store) |
| `quantity` | UInt32 | 0.00% | [1, 570] | 1 | Number of units in transaction |
| `days_ago` | Int64 | 0.00% | [0, 733] | 733 | Days since Training Max Date (Anchor: 2020-08-24) |
| `time_weight` | Float64 | 0.00% | [0.0, 1.0] | 0.00000004 | Exp decay weight (30d half-life, anchored to end of train) |

## 2. User Metadata (`customers_processed.parquet`)
*   **Purpose**: Cleansed and imputed customer profiles.

| Feature | Type | Null % | Range / Uniques | Sample | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customer_id` | Int32 | 0.00% | [0, 1,371,979] | 512 | Mapped Primary Key |
| `FN` | Int8 | 0.00% | [0, 1] | 1 | Fashion News flag (Imputed Null $\rightarrow$ 0) |
| `Active` | Int8 | 0.00% | [0, 1] | 1 | Active profile flag (Imputed Null $\rightarrow$ 0) |
| `club_member_status` | String | 0.44% | 4 unique | ACTIVE | Membership status |
| `fashion_news_frequency` | String | 1.17% | 5 unique | NONE | News frequency |
| `age` | Int8 | 0.00% | [16, 99] | 49 | Customer age (Leakage-Proofed Affinitive Imputation) |
| `postal_code` | String | 0.00% | 352,900 unique | Unknown | Hashed location (Masked Super-Code) |

## 3. Product Metadata (`articles_processed.parquet`)
*   **Purpose**: Master catalog with mapped IDs.

| Feature | Type | Null % | Range / Uniques | Sample | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `article_id` | Int32 | 0.00% | [0, 105,541] | 2048 | Mapped Primary Key |
| `product_code` | Int64 | 0.00% | [108775, 959461] | 108775 | High-level SKU identifier |
| `prod_name` | String | 0.00% | 45,875 unique | Strap top | Product Name |
| `product_group_name` | String | 0.00% | 19 unique | Garment Upper body | High-level category |
| `colour_group_name` | String | 0.00% | 50 unique | Black | Primary color |
| `index_name` | String | 0.00% | 10 unique | Ladieswear | Macro department |
| `section_name` | String | 0.00% | 56 unique | Womens Everyday Basics | Store section |
| `garment_group_name` | String | 0.00% | 21 unique | Jersey Basic | Manufacturing group |
| `detail_desc` | String | 0.39% | 43,405 unique | Jersey top... | Product description |
