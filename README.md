# DSEB 65A - Group 4 — H&M Recommendation System & Customer Intelligence

<div align="center">

[![Final Report](https://img.shields.io/badge/Final_Report-PDF-2b3137?style=flat)](docs/Final_Report.pdf)
[![Interactive Dashboard](https://img.shields.io/badge/PowerBI-Interactive_Dashboard-EB1E25?style=flat&logo=powerbi&logoColor=white)](https://drive.google.com/drive/folders/17R1_ZM1MguQdwQFNwbqP0yBn5N8KA0_t?usp=sharing)
[![H&M Personalized Fashion Competition](https://img.shields.io/badge/-H&M_Personalized_Fashion_Competition-20BEFF?style=flat&logo=Kaggle&logoColor=white)](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
![MAP@12](https://img.shields.io/badge/MAP@12-0.0075_(28_days)-success?style=flat)
<br />
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-CD792C?style=flat&logo=polars&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-000000?style=flat&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-D22128?style=flat)

</div>

A comprehensive data-driven marketing architecture built on ~31.7 million H&M transactions. This project bridges raw behavioral data and actionable customer segmentation to optimize product discovery and maximize customer lifetime value.

The final LightGBM learning-to-rank model demonstrates a 71% uplift in conversion propensity while strictly enforcing novel discovery (preventing trivial repurchase predictions).

---

## System Architecture

```mermaid
graph TD
    %% Pastel Color Palette Definitions
    classDef data fill:#E3F2FD,stroke:#1565C0,stroke-width:1px,color:#1565C0;
    classDef proc fill:#E8F5E9,stroke:#2E7D32,stroke-width:1px,color:#2E7D32;
    classDef model fill:#F3E5F5,stroke:#7B1FA2,stroke-width:1px,color:#7B1FA2;
    classDef biz fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#E65100;

    A[Raw H&M Data<br/>31.7M Rows]
    B(Temporal Split &<br/>Strict Isolation)
    C[Train Set<br/>27.8M Interactions]
    D["Holdout Test Set<br/>28 Days | 989k Interactions"]
    
    C --> E[RFM Segmentation<br/>K-Means]
    C --> F[Feature Engineering<br/>Word2Vec, Stats]
    C --> G[MBA Baseline<br/>FP-Growth]
    
    E --> H(Candidate Gen<br/>Popularity + Item-CF)
    F --> H
    
    H --> I[LightGBM Ranker]
    
    D -.->|Ground Truth Evaluation| I
    D -.->|Ground Truth Evaluation| G
    
    I --> J((Business<br/>Simulation))
    G --> J

    A --> B
    B --> C
    B --> D

    %% Categorize Nodes
    class A,C,D data;
    class B,E,F proc;
    class G,H,I model;
    class J biz;
```

## 1. Data Foundation & Strict Temporal Isolation

Building a reliable recommender for fast fashion requires confronting severe popularity bias, a 99.98% interaction matrix sparsity, and extremely short item shelf lives (most products are active for < 90 days).

### The 28-Day Holdout Philosophy

```mermaid
%%{init: { 'theme': 'base', 'themeVariables': { 
    'primaryColor': '#1565C0',
    'primaryTextColor': '#efefef',
    'primaryBorderColor': '#1565C0',
    'lineColor': '#efefef',
    'secondaryColor': '#2E7D32',
    'tertiaryColor': '#E65100',
    'ganttBarBkgColor': '#1565C0',
    'ganttBarBorderColor': '#efefef',
    'ganttBarLabelColor': '#efefef',
    'ganttCriticalBarFill': '#E65100',
    'ganttCriticalBarStroke': '#efefef',
    'ganttActiveBarFill': '#2E7D32',
    'ganttActiveBarStroke': '#efefef',
    'ganttSectionBkgColor': '#222',
    'ganttSectionBkgColor2': '#333',
    'titleColor': '#efefef',
    'sectionTitleColor': '#efefef',
    'gridLineColor': '#444',
    'axisColor': '#efefef'
}}}%%
gantt
    title Temporal Architecture & Time Decay
    dateFormat  YYYY-MM-DD
    axisFormat  %b %Y
    
    section Training Data
    Historical Interactions  :a1, 2018-09-20, 2020-08-24
    Exponential Time Decay   :crit, active, 2020-07-24, 2020-08-24
    
    section Evaluation
    Strict Holdout Test Set :active, 2020-08-25, 2020-09-22
```

We sequestered the final **28 days** of the dataset as an immutable holdout evaluation set. While standard fast-fashion benchmarks often use a 7-day window, a 28-day holdout captures nearly **1 million interactions (989,221)** across 216,479 unique active users. This massive evaluation volume natively suppresses variance and proves the architecture's stability against mid-term temporal drift, ensuring our metrics are highly statistically significant rather than an artifact of a single "lucky" week.

### Preprocessing & Leakage-Prevention
*   **Time-Decay Anchoring:** An exponential time decay (30-day half-life) was applied to the interaction matrix. This decay was anchored to the training period's cutoff (`2020-08-24`) to ensure obsolete seasonal inventory was phased out naturally without artificially discounting the freshest training data relative to the test set.
*   **Affinitive Imputation:** Rather than filling missing age data (1.16%) with global medians, demographics were inferred using the user's dominant purchase category history (e.g., users dominating the `Divided` youth line were imputed as 29). This preserved the bimodal variance necessary for accurate K-Means clustering.
*   **Reseller Purge:** Wholesale accounts (e.g., users purchasing >1,000 items) were purged strictly based on training-set history to prevent popularity bias inflation.

### Processed Feature Dictionary

<details>
<summary>Click to expand Data Dictionaries</summary>

**Interaction Matrix (`train_processed.parquet`)**
| Feature       | Type    | Purpose                                                |
| :------------ | :------ | :----------------------------------------------------- |
| `t_dat`       | Date    | Transaction Timestamp                                  |
| `customer_id` | Int32   | Mapped Primary Key (Zero-indexed)                      |
| `article_id`  | Int32   | Mapped SKU Identifier                                  |
| `price`       | Float64 | Normalized Purchase Price                              |
| `quantity`    | UInt32  | Aggregated Unit Volume (resolves 17% exact duplicates) |
| `days_ago`    | Int64   | Days since Training Max Date                           |
| `time_weight` | Float64 | Exponential Decay (30d half-life)                      |

**Demographic Matrix (`customers_processed.parquet`)**
| Feature         | Type   | Purpose                                              |
| :-------------- | :----- | :--------------------------------------------------- |
| `age`           | Int8   | Affinitively Imputed Age                             |
| `FN` / `Active` | Int8   | Binary flags (Nulls imputed to 0)                    |
| `postal_code`   | String | Masked Location Hash (Mega-node mapped to `Unknown`) |
</details>

## 2. Customer Intelligence & RFM Segmentation

The prescriptive marketing logic relies on segmenting the 1.37M customer base into distinct economic clusters using Yeo-Johnson transformed K-Means. We prioritize economic utility over pure geometric symmetry.

| Segment              | Headcount | Revenue Share | Median Profile   | Operating Tactic                                                      |
| :------------------- | :-------- | :------------ | :--------------- | :-------------------------------------------------------------------- |
| **Young Whales**     | 23.36%    | **54.35%**    | Age 26, Freq: 11 | Zero-discount upselling; margin density focus.                        |
| **Mature Champions** | 15.60%    | **31.08%**    | Age 51, Freq: 9  | Retention and replenishment; relevance stability.                     |
| **Young At-Risk**    | 23.85%    | 5.29%         | Age 27, Freq: 1  | Margin-controlled win-back; historically high-conversion SKU routing. |
| **Mature Newcomers** | 19.98%    | 4.41%         | Age 53, Freq: 1  | Certainty-first demographic priors until history builds.              |
| **Young Core**       | 17.21%    | 4.87%         | Age 25, Freq: 2  | Basket stretching and category cross-selling.                         |

*Finding:* A Pareto-like concentration exists where **Whales + Champions represent 39% of headcount but command 85% of total revenue**. Furthermore, Markov chain analysis revealed a ~61% "stickiness" in `Ladieswear` and a clear evolutionary pipeline mapping users aging out of the `Divided` youth index directly into mature lines.

## 3. Predictive Architectures & Pipeline Engineering

To establish a definitive performance hierarchy, we evaluated models strictly on **Novel Discovery**. By enforcing an anti-join against a user's full purchase history during candidate generation, the system was barred from artificially inflating Hit Rates via trivial repurchases.

**Evaluation Set:** 216,479 warm users (Final 28 days holdout)

| Model Architecture              | Methodology                    | MAP@12     | Hit Rate@12 |
| :------------------------------ | :----------------------------- | :--------- | :---------- |
| **MBA Baseline**                | Explainable FP-Growth rules    | 0.0045     | 4.44%       |
| **ALS (Matrix Factorization)**  | 1-to-1 implicit latent factors | 0.0063     | 4.83%       |
| **LightGBM (Learning-to-Rank)** | Multi-feature boosted ranking  | **0.0075** | **7.66%**   |

### 3.1 Checkout Impulse (MBA Baseline)
To solve the fast-fashion "Dead-Stock Trap" (recommending obsolete items), we implemented a dual-window temporal heuristic. This engine serves as our high-speed, $O(1)$ dictionary lookup for active shopping carts, utilizing an Expected Cross-Sell Revenue (ECR) heuristic to push premium bundles to VIP segments.

<div align="center">
  <img src="docs/assets/mba_pipeline.png" alt="MBA FP-Growth Pipeline" width="800">
</div>

### 3.2 Collaborative Filtering (Matrix Factorization)
The ALS model transitions from generalized rules to true 1-to-1 latent personalization. While mathematically rigorous, simulation revealed it suffers from extreme popularity bias—relying heavily on dense interaction histories, which causes it to fail on newly acquired or dormant consumers.

<div align="center">
  <img src="docs/assets/als_pipeline.png" alt="ALS Matrix Factorization Pipeline" width="800">
</div>

### 3.3 The Winning Engine: LightGBM (Learning-to-Rank)
The L2R framework reframes recommendation as a supervised classification problem. By evaluating features like semantic similarity (`w2v_cosine_sim`) and wallet-matching (`user_avg_price`), this context-rich approach overcomes the sparse-user deficit found in other models.

<div align="center">
  <img src="docs/assets/lgbm_pipeline.png" alt="LightGBM Ranking Pipeline" width="800">
</div>

## 4. Randomized Controlled Trial & The "ROI Trap"

Through a simulated deployment environment, we analyzed the economic impact of three recommendation architectures. The goal was to maximize Incremental Recommendation Revenue (IRR) across all customer segments without falling into the "ROI Trap" of mathematically significant but financially negligible models. While ALS showed a statistically significant +9.36% aggregate Hit Rate uplift over the baseline, segment-stratified analysis revealed a severe **Simpson's Paradox**: it boosted "Whales" (+12%) but **actively harmed dormant users**—"Young At-Risk" dropped by -33.5% and "Mature Newcomers" dropped by -19.1%. It indiscriminately pushed cheap basics instead of culturally relevant fashion, resulting in zero actual economic revenue shift (The ROI Trap).

### LightGBM Dominance & Deployment Mandate
Conversely, LightGBM generated massive uplifts in *every single segment*, effectively doubling the overall IRR. We mandate a **Dual-Engine Prescriptive Logic**:
1.  **Homepage / Navigation (LightGBM):** Handles asynchronous, computationally heavy personalization to govern the largest IRR opportunity.
2.  **Checkout Cart (MBA FP-Growth):** Retained for real-time impulse generation, capitalizing on high-speed association rules for active sessions.

## 5. Executive BI Dashboard

To operationalize these findings, we developed a 4-tab interactive dashboard allowing stakeholders to explore customer segments, product affinity networks, and live A/B test results.

| Tab | Focus | Key Insight |
| :--- | :--- | :--- |
| **RFM Explorer** | Descriptive Analytics | Visualizes the Pareto concentration where 39% of users drive 85% of revenue. |
| **MBA Network** | Association Rules | Maps the "Gravity Well" of Ladieswear using interactive chord diagrams. |
| **A/B Test Results** | Business Impact | Surfaces the Simpson's Paradox and quantifies the $22.9K NRU annual yield. |
| **Virtual Assistant** | Model Inference | A live demo showing personalized Top-12 recommendations for any User ID. |

<div align="center">
  <a href="[POWERBI_LINK](https://drive.google.com/drive/folders/17R1_ZM1MguQdwQFNwbqP0yBn5N8KA0_t?usp=sharing)">
    <img src="docs/assets/powerbi_preview.png" alt="PowerBI Dashboard Preview" width="800">
    <br/>
    <b>View Live Interactive Dashboard</b>
  </a>
</div>

## 6. Project Structure

![Project Structure](docs/assets/project-structure.png)

## 7. Contributions

*   **Class:** DSEB 65A
*   **Course:** Data Driven Marketing
*   **Group:** Group 4

| Student ID | Full Name | Task Assignment |
| :---: | :---: | :---: |
| 11230517 | Vũ Ngọc Hồng Anh | Customer Intelligence, RFM Segmentation & K-Means Profiling |
| 11230527 | Đỗ Tuấn Đạt | Algorithmic Architecture, LightGBM Ranker & Personalization |
| 11230553 | Hàn Chí Kiên | Exploratory Data Analysis, Temporal Isolation & Pipeline Engineering |
| 11230570 | Phạm Hồng Minh | Association Discovery, MBA Baseline & ECR Heuristics, PowerBI (Present Ver) |
| 11230584 | Chu Bích Phương | Data Integrity, Validation & Final Report Compiler |
| 11230588 | Nguyễn Thanh Thảo **(Leader)** | All Tasks Code Optimization, A/B Testing - RCT Simulation & Business Analysis, Business Action Plan, PowerBI (Advanced Ver) |
