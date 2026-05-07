# H&M Recommendation System — Model Comparison Report

**Project:** H&M Fashion Retail Personalization Engine  
**Dataset:** H&M Kaggle Competition Dataset (~31.7M transactions, 1.37M customers, ~105K articles)  
**Evaluation Metric:** MAP@12 (Mean Average Precision at 12) and Hit Rate@12  
**Evaluation Set:** 239,951 test-period customers (final 28-day holdout window)

---

## 1. Overview

This report introduces and compares three recommendation models developed for the H&M fashion retail recommendation system. The models form a progressive hierarchy—from simple association-based rules to full personalized Learning-to-Rank—demonstrating increasing sophistication and predictive power.

| # | Model | Category | MAP@12 | Hit Rate@12 | Users Evaluated |
|---|-------|----------|--------|-------------|-----------------|
| 1 | **MBA Baseline** (FP-Growth + Demographics) | Association Rules | 0.003427 | 4.32% | 239,951 |
| 2 | **ALS Collaborative Filtering** (Matrix Factorization) | Memory-Based CF | 0.011692 | 7.81% | 239,951 |
| 3 | **LightGBM Learning-to-Rank** (Gradient Boosting) | Learning-to-Rank | 0.013956 | 9.33% | 239,951 |

> **Key Takeaway:** The LightGBM model achieves the highest MAP@12 of **0.013956**, representing a **4.07×** uplift over the baseline and a **19.4%** improvement over ALS.

---

## 2. Model 1: MBA Baseline — Market Basket Analysis & FP-Growth

### 2.1 Introduction

The MBA Baseline serves as the non-personalized benchmark. It extracts statistically significant cross-selling patterns from historical transactions using the **FP-Growth algorithm** and serves cold-start users via demographically-aware popularity recommendations.

**Key Innovation:** Transitions from purely predictive analytics to *prescriptive analytics* by integrating RFM segmentation — prioritizing high-margin product bundles for VIP customer segments (Expected Cross-Sell Revenue heuristic).

### 2.2 Pipeline Architecture

```
Raw Transactions
      │
      ▼
┌─────────────────────────────────────────┐
│  1. Dual-Window Temporal Filtering      │
│     • Active Catalog: Last 30 days      │
│     • Historical Context: Last 90 days  │
│       (restricted to active items)      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  2. Native Polars Mathematical Pruning  │
│     • min_support threshold = 0.1%      │
│     • C++ Lazy evaluation graph         │
│     • Eliminates OOM risk (>3GB crash)  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  3. FP-Growth Frequent Itemset Mining   │
│     • 146,460 multi-item baskets        │
│     • Matrix shape: (146460, 362)       │
│     • Found: 539 frequent itemsets      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  4. Association Rule Extraction         │
│     • Confidence ≥ 0.20                 │
│     • Lift > 1.0                        │
│     • N-to-1 rule architecture          │
│     • 186 enriched rules exported       │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  5. Recommendation Engine               │
│     • Cold-Start: Age-bucketed trending │
│       (<25, 25-35, 36-50, >50)          │
│     • Warm Users: Cart-based rule match │
│     • VIP: ECR = Lift × Conf × Price   │
│     • Fallback padding to Top-K=12      │
└─────────────────────────────────────────┘
      │
      ▼
  Top-12 Recommendations per User
```

### 2.3 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **FP-Growth over Apriori** | 3.8× faster on sparse datasets (compressed FP-Tree vs. iterative candidate generation) |
| **Dual-Window Strategy** | Prevents dead-stock recommendations — fast-fashion has ~90-day shelf-life |
| **Native Polars Pruning** | Bypasses mlxtend's dense matrix densification that caused >3GB RAM crash |
| **Repurchase tolerance** | Repurchase is a dominant behavioral signal in H&M ecosystem (basics/underwear) |
| **ECR for VIP segments** | Prevents "ROI Trap" — avoids suggesting cheap items to high-value customers |

### 2.4 Performance

| Metric | Value |
|--------|-------|
| MAP@12 | 0.003427 |
| Hit Rate@12 | 4.32% (10,365 / 239,951 users) |
| Evaluation Time | 7.3 seconds |

---

## 3. Model 2: ALS Collaborative Filtering — Matrix Factorization

### 3.1 Introduction

The ALS model transitions from generalized association rules to **true 1-to-1 personalization**. By applying the Alternating Least Squares algorithm (Hu, Koren & Volinsky, 2008) for implicit feedback, the model learns latent preference vectors for every user based on their full purchase history.

**Objective Function:**
$$\min_{X,Y} \sum_{u,i} c_{ui}(p_{ui} - x_u^T y_i)^2 + \lambda(\|x_u\|^2 + \|y_i\|^2)$$

### 3.2 Pipeline Architecture

```
train_processed.parquet
      │
      ▼
┌─────────────────────────────────────────┐
│  1. Confidence Matrix Construction      │
│     c_ui = Σ(quantity × time_weight)   │
│     • 1,338,785 users × 100,959 items  │
│     • ~26.3M non-zero interactions      │
│     • Sparsity: 99.98%                  │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  2. ALS Model Training                  │
│     • Library: implicit (C++ backend)   │
│     • Latent Factors: 200               │
│     • Regularization (λ): 0.02          │
│     • Alpha (α): 60                     │
│     • Iterations: 25                    │
│     • Training time: ~88.6 seconds      │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  3. Memory-Efficient Batch Prediction   │
│     • Chunked matmul: 2,000 users/chunk │
│     • argpartition O(N) top-K selection │
│     • Aggressive gc.collect() per chunk │
│     • Warm: 216,475 users → ALS scores  │
│     • Cold: 23,476 users → Popularity   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  4. Offline Evaluation (MAP@12)         │
│     • 239,951 test users evaluated      │
│     • Separate warm vs. cold metrics    │
└─────────────────────────────────────────┘
      │
      ▼
  ALS_predictions.parquet (239,951 users × Top-12)
```

### 3.3 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Implicit feedback (not explicit ratings)** | Retail data only shows purchases, not dislikes |
| **time_weight decay** | Penalizes old transactions — discontinues SKUs from past seasons |
| **200 latent factors** | Balances expressive power vs. overfitting at this scale |
| **C++ backend (implicit library)** | ~10× faster vs. pure NumPy ALS (with NumPy fallback for reproducibility) |
| **argpartition for top-K** | O(N) partial sort vs. O(N log N) full sort — critical for 100K items |
| **Chunked prediction (2K users/chunk)** | Prevents OOM from dense 1.3M × 100K matrix materialization |

### 3.4 Performance

| Metric | Overall | Warm Users (216,475) | Cold Users (23,476) |
|--------|---------|----------------------|---------------------|
| MAP@12 | 0.011692 | 0.012522 | 0.004040 |
| Hit Rate@12 | 7.81% (18,745 users) | — | — |
| Training Time | 88.6 seconds | — | — |

**Lift over Baseline:** MAP@12 improved **3.41×** (0.003427 → 0.011692).

---

## 4. Model 3: LightGBM Learning-to-Rank

### 4.1 Introduction

The LightGBM model reframes recommendation as a **supervised binary classification problem**: given a (user, item) candidate pair, predict the probability that the user will purchase the item. This enables the model to leverage rich multi-dimensional features beyond simple interaction counts.

A two-stage pipeline is used:
1. **Feature Engineering** (`05_feature_engineering.py`) — offline, run once
2. **LightGBM Training & Scoring** (`05_LightGBM_RecSys.ipynb`) — fast at inference

### 4.2 Pipeline Architecture

#### Stage 1: Feature Engineering (Pre-computation)

```
train_processed.parquet + test_processed.parquet
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Candidate Generation (3 strategies per user)       │
│  ┌───────────────┐ ┌────────────────┐ ┌──────────┐ │
│  │  Repurchase   │ │   Popularity   │ │  Item-CF │ │
│  │ (User history)│ │ (Top-50 items, │ │ (Cosine  │ │
│  │               │ │   last 7 days) │ │ similarity│ │
│  └───────────────┘ └────────────────┘ └──────────┘ │
│         ↓                  ↓               ↓        │
│              Deduplicated, capped at 100/user       │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Feature Engineering (21 features per candidate)    │
│                                                     │
│  User-Item Interaction:                             │
│    ui_purchase_count, ui_days_since_last            │
│    ui_tw_1w, ui_tw_1m, ui_tw_1s (time-weighted)    │
│                                                     │
│  Item Aggregation:                                  │
│    item_price_mean/max/min, item_total_sales        │
│    item_age_mean/max/min (buyer demographics)       │
│                                                     │
│  User Aggregation:                                  │
│    user_total_purchases, user_unique_items          │
│    user_avg_price, user_active_days, age            │
│                                                     │
│  Semantic Similarity:                               │
│    w2v_cosine_sim (Word2Vec item embeddings)        │
│                                                     │
│  Cumulative Features:                               │
│    cum_purchase_count, cum_avg_price                │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Label Assignment & Downsampling                    │
│  Train: label=1 if bought in last 7 days of train   │
│  Test: label=1 if bought in test_processed (28 days)│
│  Negatives downsampled to 1,500,000 for training    │
└─────────────────────────────────────────────────────┘
      │
      ▼
  lgbm_train_features.parquet  (1,515,896 rows × 24 cols)
  lgbm_test_features.parquet   (21,825,652 rows × 24 cols)
```

#### Stage 2: LightGBM Training & Inference

```
lgbm_train_features.parquet
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  LightGBM Binary Classifier Training               │
│  • Objective: binary (cross-entropy)                │
│  • Metric: AUC                                      │
│  • Learning rate: 0.01                              │
│  • num_leaves: 255                                  │
│  • feature_fraction: 0.8                            │
│  • bagging_fraction: 0.8, bagging_freq: 5           │
│  • Early stopping: 100 rounds patience              │
│  • Best iteration: 840 | Best AUC: 0.8596           │
│  • Training time: 84.9 seconds                      │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Score All Test Candidates                          │
│  • 21.8M candidate pairs scored                     │
│  • Sort by score descending per user                │
│  • Extract Top-12 per user                          │
└─────────────────────────────────────────────────────┘
      │
      ▼
  Top-12 Recommendations per User (239,951 users)
```

### 4.3 Feature Importance (Top-10)

| Rank | Feature | Description | Type |
|------|---------|-------------|------|
| 1 | `item_total_sales` | Total historical units sold | Item popularity |
| 2 | `w2v_cosine_sim` | Word2Vec user-item semantic similarity | Semantic |
| 3 | `ui_tw_1s` | Time-weighted interaction (last 90 days) | Interaction |
| 4 | `user_avg_price` | User's average purchase price | User profile |
| 5 | `item_age_mean` | Mean age of item buyers | Item profile |
| 6 | `age` | User's age | User profile |
| 7 | `item_price_mean` | Item's mean price | Item profile |
| 8 | `ui_days_since_last` | Days since last purchase of this item | Recency |
| 9 | `user_total_purchases` | User's total purchase count | User activity |
| 10 | `item_price_min` | Item's minimum historical price | Item profile |

### 4.4 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **3-strategy candidate generation** | Ensures coverage: repurchase (recency), popularity (cold-start), Item-CF (discovery) |
| **Word2Vec item embeddings** | Captures semantic co-purchase patterns beyond explicit interaction counts |
| **Time-windowed features (1w/1m/1s)** | Models purchase recency at multiple granularities |
| **Negative downsampling (1.5M cap)** | Balances positive-negative ratio while maintaining training efficiency |
| **LightGBM binary classification** | AUC optimization → probability ranking naturally solves top-K retrieval |

### 4.5 Performance

| Metric | Value |
|--------|-------|
| MAP@12 | 0.013956 |
| Hit Rate@12 | 9.33% (22,392 / 239,951 users) |
| Best AUC (validation) | 0.8596 |
| Evaluation Time | 194.2 seconds |

**Lift over Baseline:** MAP@12 improved **4.07×** (0.003427 → 0.013956).  
**Lift over ALS:** MAP@12 improved **19.4%** (0.011692 → 0.013956).

---

## 5. Model Comparison & Analysis

### 5.1 Performance Summary

```
MAP@12 Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MBA Baseline  ████░░░░░░░░░░░░░░░░░░░░░░░  0.003427  (1.0×)
ALS (CF)      ████████████████░░░░░░░░░░░  0.011692  (3.41×)
LightGBM      ████████████████████░░░░░░░  0.013956  (4.07×)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Hit Rate@12 Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MBA Baseline  ████░░░░░░░░░░░░░░░░░░░░░░░  4.32%
ALS (CF)      ████████░░░░░░░░░░░░░░░░░░░  7.81%
LightGBM      █████████░░░░░░░░░░░░░░░░░░  9.33%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.2 Head-to-Head Comparison

| Dimension | MBA Baseline | ALS | LightGBM |
|-----------|-------------|-----|----------|
| **Personalization** | Demographic-only | True per-user latent factors | Per-user + multi-feature |
| **Cold-start handling** | Age-bucketed trending | Global popularity fallback | Popularity candidates + features |
| **Scalability** | Low (rule lookup) | Medium (chunked matmul) | High (pre-built feature file) |
| **Interpretability** | High (Lift, Confidence) | Low (latent space) | Medium (feature importance) |
| **Training data required** | Association rules | Full interaction matrix | Labeled candidate pairs |
| **Feature richness** | Statistical rules | Interaction history only | 21 multi-dimensional features |
| **Inference speed** | Very fast (O(1) lookup) | Medium (matmul per user) | Fast (model.predict on pre-built) |
| **Business prescriptiveness** | High (ECR, VIP logic) | None | Low (pure ranking) |

### 5.3 Key Findings

1. **Personalization drives the biggest gain:** Moving from the non-personalized MBA Baseline to ALS delivers a **3.41× improvement** in MAP@12, confirming that individual preference modeling is far more effective than demographic averaging.

2. **Rich features further close the gap:** LightGBM's additional 21 features — especially `w2v_cosine_sim` (semantic similarity) and `item_total_sales` (global popularity signal) — push MAP@12 **19.4% beyond ALS**.

3. **Word2Vec semantic similarity is the 2nd most important feature**, suggesting that the "purchase sequence" contains rich item-relationship signals that matrix factorization alone cannot capture.

4. **Cold-start remains a bottleneck:** ALS warm users achieve MAP@12 = 0.012522 vs. 0.004040 for cold users — a **3.1× performance gap**, indicating cold-start resolution is a high-priority improvement area.

5. **The MBA Baseline still adds value as a business tool:** Despite lower MAP@12, it provides explainable cross-selling rules and prescriptive VIP optimization that purely statistical models cannot offer.

---

## 6. Architecture Overview

### End-to-End System Pipeline

```
RAW DATA (H&M Kaggle Dataset)
├── transactions_train.csv  (~31.7M rows)
├── customers.csv           (~1.37M users)
└── articles.csv            (~105K items)
        │
        ▼
┌───────────────────────────────────────────┐
│  Phase 1: Data Engineering & EDA          │
│  • Chronological train/test split         │
│  • Feature preprocessing & normalization │
│  • RFM segmentation (K-Means)             │
│  → train_processed.parquet                │
│  → test_processed.parquet                 │
│  → customers_segmented.parquet            │
└───────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────────────┐
        │                                                  │
        ▼                                                  ▼
┌─────────────────────────────────┐        ┌──────────────────────────────────┐
│  Model Track A: MBA Baseline    │        │  Model Track B: ALS + LightGBM   │
│  • FP-Growth rule mining        │        │                                  │
│  • Demographic cold-start       │        │  ALS: Matrix Factorization       │
│  • ECR VIP prescriptive ranking │        │  → ALS_predictions.parquet       │
│  MAP@12: 0.003427               │        │  MAP@12: 0.011692                │
└─────────────────────────────────┘        │                                  │
                                           │  LightGBM: Learning-to-Rank     │
                                           │  → Feature Engineering Script    │
                                           │  → lgbm_train_features.parquet   │
                                           │  → lgbm_test_features.parquet    │
                                           │  MAP@12: 0.013956                │
                                           └──────────────────────────────────┘
```

---

## 7. Future Improvements

| Improvement | Expected Impact | Complexity |
|-------------|-----------------|------------|
| **Two-Tower Neural Network** | Higher MAP via deep feature interactions | High |
| **LightGBM LambdaRank** | Direct MAP optimization vs. AUC proxy | Medium |
| **Sequential models (GRU4Rec)** | Capture purchase sequence dynamics | High |
| **Better cold-start** (content-based) | Close warm/cold MAP gap (3.1×) | Medium |
| **Ensemble ALS + LightGBM** | Combine latent factors + rich features | Low |
| **Online learning / real-time updates** | React to intra-session signals | Very High |

---

## 8. Conclusion

The three-model progression demonstrates a clear and measurable improvement hierarchy for the H&M recommendation system:

- **MBA Baseline** (MAP@12: 0.003427): Establishes a rules-based, explainable baseline with prescriptive VIP business logic.
- **ALS Collaborative Filtering** (MAP@12: 0.011692): Introduces true personalization via latent factor modeling, delivering a 3.41× improvement.
- **LightGBM Learning-to-Rank** (MAP@12: 0.013956): Achieves the best offline performance by combining rich multi-dimensional features with gradient-boosted ranking, reaching a 4.07× improvement over the baseline.

The results confirm that personalization and feature richness are the primary drivers of recommendation quality in the H&M fashion retail context.

---

*Report generated from experimental results logged in:*  
- `notebooks/03_MBA_Baseline.ipynb`  
- `notebooks/4_RecSys_ALS.ipynb`  
- `notebooks/05_LightGBM_RecSys.ipynb`  
- `scripts/05_feature_engineering.py`
