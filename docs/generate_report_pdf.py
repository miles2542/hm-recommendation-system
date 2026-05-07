"""
Generate Model_Comparison_Report.pdf from the markdown report.
Uses fpdf2 library.
Run: python docs/generate_report_pdf.py
"""
from fpdf import FPDF
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def s(text):
    """Sanitize text to latin-1 safe ASCII."""
    return (
        str(text)
        .replace("\u2014", "-").replace("\u2013", "-").replace("\u2022", "*")
        .replace("\u2192", "->").replace("\u00d7", "x").replace("\u2265", ">=")
        .replace("\u2264", "<=").replace("\u00b2", "^2").replace("\u00b3", "^3")
        .encode("latin-1", errors="replace").decode("latin-1")
    )

ROOT  = Path(__file__).parent
IMG   = ROOT.parent / "figures" / "lgbm_feature_importance.png"
OUT   = ROOT / "Model_Comparison_Report.pdf"

BLUE  = (30, 100, 200)
DBLUE = (10, 50, 130)
LGRAY = (240, 240, 245)
MGRAY = (160, 160, 160)
BLACK = (30, 30, 30)
GREEN = (34, 139, 34)

# ── helpers ──────────────────────────────────────────────────────────────────
def h1(pdf, text):
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*DBLUE)
    pdf.ln(4)
    pdf.cell(0, 9, s(text), ln=True)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.6)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 175, pdf.get_y())
    pdf.ln(3)

def h2(pdf, text):
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*BLUE)
    pdf.ln(3)
    pdf.cell(0, 7, s(text), ln=True)
    pdf.set_text_color(*BLACK)

def h3(pdf, text):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*DBLUE)
    pdf.ln(2)
    pdf.cell(0, 6, s(text), ln=True)
    pdf.set_text_color(*BLACK)

def body(pdf, text, size=9):
    pdf.set_font("Helvetica", "", size)
    pdf.set_text_color(*BLACK)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, s(text))
    pdf.ln(1)

def bullet(pdf, text, indent=5):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*BLACK)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, s(f"*  {text}"))

def kv(pdf, key, val):
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*DBLUE)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, s(key))
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*BLACK)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, s("    " + val))
    pdf.ln(1)

def table_row(pdf, cols, widths, bold=False, bg=None):
    if bg:
        pdf.set_fill_color(*bg)
    style = "B" if bold else ""
    pdf.set_font("Helvetica", style, 8)
    pdf.set_text_color(*BLACK)
    x0, y0 = pdf.get_x(), pdf.get_y()
    max_h = 5
    for i, (c, w) in enumerate(zip(cols, widths)):
        pdf.set_xy(x0 + sum(widths[:i]), y0)
        pdf.cell(w, max_h, s(c), border=1, fill=bool(bg))
    pdf.set_xy(x0, y0 + max_h)

def section_header(pdf, title, subtitle=""):
    pdf.set_fill_color(*DBLUE)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, s(title), ln=True, fill=True)
    if subtitle:
        pdf.set_fill_color(*BLUE)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, s(subtitle), ln=True, fill=True)
    pdf.set_text_color(*BLACK)
    pdf.ln(2)

def pipeline_box(pdf, steps):
    pdf.set_fill_color(*LGRAY)
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(0.3)
    for step in steps:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*DBLUE)
        pdf.cell(0, 5, s(step["title"]), ln=True, fill=True, border=1)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*BLACK)
        for line in step["lines"]:
            pdf.set_x(pdf.get_x() + 5)
            pdf.cell(0, 4, s(line), ln=True)
        # arrow
        cx = pdf.get_x() + 87
        y  = pdf.get_y()
        pdf.set_draw_color(*MGRAY)
        pdf.line(cx, y, cx, y + 3)
        pdf.ln(3)

def metric_badge(pdf, label, value, color):
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(55, 7, s(f"{label}: {value}"), fill=True, border=0)
    pdf.set_text_color(*BLACK)

# ── build PDF ─────────────────────────────────────────────────────────────────
pdf = FPDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_margins(15, 15, 15)

# ════════════════════════════════════════════════
# PAGE 1 — COVER
# ════════════════════════════════════════════════
pdf.add_page()

pdf.set_fill_color(*DBLUE)
pdf.rect(0, 0, 210, 60, "F")
pdf.set_text_color(255, 255, 255)
pdf.set_font("Helvetica", "B", 22)
pdf.set_y(15)
pdf.cell(0, 12, "H&M Recommendation System", ln=True, align="C")
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "Model Comparison Report", ln=True, align="C")
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 7, "MBA Baseline  |  ALS Collaborative Filtering  |  LightGBM Learning-to-Rank", ln=True, align="C")
pdf.set_y(65)
pdf.set_text_color(*BLACK)

body(pdf, "Project: H&M Fashion Retail Personalization Engine")
body(pdf, "Dataset: H&M Kaggle Competition (~31.7M transactions, 1.37M customers, ~105K articles)")
body(pdf, "Evaluation: MAP@12 & Hit Rate@12 on 239,951 test-period customers (final 28-day holdout)")

h1(pdf, "Executive Summary")
body(pdf, (
    "This report introduces and compares three recommendation models forming a progressive hierarchy "
    "from association-based rules to full personalized Learning-to-Rank. Each model is presented with "
    "its pipeline architecture, technical decisions, and offline evaluation results."
))

# Summary table
pdf.ln(2)
h2(pdf, "Performance Overview")
hdrs = ["#", "Model", "Category", "MAP@12", "Hit Rate@12", "Lift vs Baseline"]
wids = [8, 55, 40, 22, 26, 30]
table_row(pdf, hdrs, wids, bold=True, bg=DBLUE)
# Override color for header
pdf.set_text_color(255,255,255)
pdf.set_xy(15, pdf.get_y()-5)
pdf.set_font("Helvetica","B",8)
for i,(c,w) in enumerate(zip(hdrs,wids)):
    pdf.cell(w,5,c,border=1,fill=True)
pdf.ln()
pdf.set_text_color(*BLACK)

rows = [
    ["1", "MBA Baseline (FP-Growth)", "Association Rules", "0.003427", "4.32%", "1.00x (baseline)"],
    ["2", "ALS Collaborative Filtering", "Matrix Factorization", "0.011692", "7.81%", "3.41x"],
    ["3", "LightGBM Learning-to-Rank", "Gradient Boosting", "0.013956", "9.33%", "4.07x"],
]
for i, r in enumerate(rows):
    bg = LGRAY if i % 2 == 0 else (255,255,255)
    table_row(pdf, r, wids, bg=bg)

pdf.ln(4)
pdf.set_fill_color(*GREEN)
pdf.set_text_color(255,255,255)
pdf.set_font("Helvetica","B",9)
pdf.cell(0, 7, "  Key Takeaway: LightGBM achieves MAP@12=0.013956, a 4.07x uplift over the baseline and 19.4% above ALS", fill=True, ln=True)
pdf.set_text_color(*BLACK)

# ════════════════════════════════════════════════
# PAGE 2 — MBA Baseline
# ════════════════════════════════════════════════
pdf.add_page()
section_header(pdf, "Model 1: MBA Baseline", "Market Basket Analysis & FP-Growth + Demographics")

h2(pdf, "1.1 Introduction")
body(pdf, (
    "The MBA Baseline is the non-personalized benchmark. It extracts cross-selling patterns from historical "
    "transactions using the FP-Growth algorithm and handles cold-start users via age-bucketed popularity "
    "recommendations. A key innovation is the ECR (Expected Cross-Sell Revenue) metric that prescriptively "
    "prioritizes high-margin bundles for VIP customers (\"Mature Champions\", \"Young Whales\")."
))

h2(pdf, "1.2 Pipeline")
steps = [
    {"title": "Step 1: Dual-Window Temporal Filtering",
     "lines": ["Active Catalog: last 30 days (proxy for live inventory)",
               "Historical Context: last 90 days restricted to active items"]},
    {"title": "Step 2: Native Polars Mathematical Pruning (C++ backend)",
     "lines": ["min_support = 0.1%  |  eliminated >3GB RAM crash from mlxtend densification",
               "1,240,764 historical baskets  |  146,460 multi-item baskets  |  362 active items"]},
    {"title": "Step 3: FP-Growth Frequent Itemset Mining",
     "lines": ["539 frequent itemsets found  |  confidence >= 0.20  |  lift > 1.0"]},
    {"title": "Step 4: N-to-1 Association Rule Extraction",
     "lines": ["186 enriched rules exported to Association_Rules.parquet",
               "Antecedent = multi-item cart subset  |  Consequent = single item"]},
    {"title": "Step 5: Recommendation Engine",
     "lines": ["Cold-start: age buckets (<25, 25-35, 36-50, >50) -> last-30-day trending",
               "Warm: cart-based rule matching via set(antecedents).issubset(cart)",
               "VIP: ECR = Lift x Confidence x Median_Price  |  Fallback pad to K=12"]},
]
pipeline_box(pdf, steps)

h2(pdf, "1.3 Key Technical Decisions")
decisions = [
    ("FP-Growth over Apriori", "3.8x faster on sparse data - compressed FP-Tree vs. iterative generation"),
    ("Dual-Window Strategy", "Prevents dead-stock recommendations (fast-fashion ~90-day shelf-life)"),
    ("Native Polars Pruning", "Bypasses mlxtend dense-matrix OOM; runs on local hardware safely"),
    ("Repurchase tolerance", "Repurchase is a dominant signal in H&M (basics/underwear categories)"),
    ("ECR for VIP segments", "Prevents recommending cheap items to high-value customers"),
]
for k, v in decisions:
    kv(pdf, k + ":", v)

h2(pdf, "1.4 Performance")
pdf.ln(1)
metric_badge(pdf, "MAP@12", "0.003427", DBLUE)
pdf.ln(1)
metric_badge(pdf, "Hit Rate@12", "4.32%  (10,365 / 239,951 users)", BLUE)
pdf.ln(1)
metric_badge(pdf, "Eval time", "7.3 seconds", MGRAY)
pdf.ln(5)

# ════════════════════════════════════════════════
# PAGE 3 — ALS
# ════════════════════════════════════════════════
pdf.add_page()
section_header(pdf, "Model 2: ALS Collaborative Filtering", "Alternating Least Squares — Hu, Koren & Volinsky (2008)")

h2(pdf, "2.1 Introduction")
body(pdf, (
    "The ALS model transitions to true 1-to-1 personalization. By factorizing the implicit User-Item "
    "confidence matrix, the model learns latent preference vectors for every user. "
    "Objective: min X,Y  sum_ui c_ui(p_ui - x_u^T y_i)^2 + lambda(||x_u||^2 + ||y_i||^2)"
))

h2(pdf, "2.2 Implicit Confidence Signal")
body(pdf, "c_ui = sum(quantity_ui * time_weight_ui)   where time_weight decays older transactions (half-life decay).")
body(pdf, "Matrix scale: 1,338,785 users x 100,959 items  |  ~26.3M non-zero interactions  |  Sparsity: 99.98%")

h2(pdf, "2.3 Pipeline")
steps = [
    {"title": "Step 1: Confidence Matrix Construction",
     "lines": ["quantity * time_weight aggregated per (user, item)",
               "Stored as scipy CSR sparse matrix  |  Build time: 5.7s"]},
    {"title": "Step 2: ALS Model Training (implicit C++ backend)",
     "lines": ["Factors=200  |  Regularization=0.02  |  Alpha=60  |  Iterations=25",
               "User factors: (1,338,785 x 200)  |  Item factors: (100,959 x 200)  |  88.6s"]},
    {"title": "Step 3: Memory-Efficient Batch Prediction",
     "lines": ["Chunk size: 2,000 users  |  argpartition O(N) top-K extraction",
               "Warm users (216,475): ALS dot product  |  Cold users (23,476): global popularity"]},
    {"title": "Step 4: Offline Evaluation",
     "lines": ["Ground truth: test_processed.parquet (239,951 users)",
               "Separate MAP@12 tracked for warm vs. cold cohorts"]},
]
pipeline_box(pdf, steps)

h2(pdf, "2.4 Key Technical Decisions")
decisions = [
    ("Implicit feedback", "Retail data only shows purchases, not dislikes — requires confidence weighting"),
    ("time_weight decay", "Penalizes discontinued SKUs from past seasons"),
    ("200 latent factors", "Balances expressiveness vs. overfitting at 1.3M user scale"),
    ("C++ backend", "~10x faster than pure NumPy ALS (NumPy fallback kept for reproducibility)"),
    ("argpartition", "O(N) partial sort vs. O(N log N) — critical for 100K items per user"),
    ("2K-user chunks", "Prevents OOM from dense 1.3M x 100K matrix materialization"),
]
for k, v in decisions:
    kv(pdf, k + ":", v)

h2(pdf, "2.5 Performance")
hdrs2 = ["Cohort", "Users", "MAP@12", "Hit Rate@12"]
wids2 = [45, 35, 30, 40]
pdf.set_fill_color(*DBLUE)
pdf.set_text_color(255,255,255)
pdf.set_font("Helvetica","B",8)
for c,w in zip(hdrs2,wids2):
    pdf.cell(w,5,c,border=1,fill=True)
pdf.ln()
pdf.set_text_color(*BLACK)
rows2 = [
    ["Overall","239,951","0.011692","7.81% (18,745 users)"],
    ["Warm users","216,475","0.012522","-"],
    ["Cold users","23,476","0.004040","-"],
]
for i,r in enumerate(rows2):
    bg = LGRAY if i%2==0 else (255,255,255)
    table_row(pdf,r,wids2,bg=bg)
pdf.ln(3)
body(pdf, "Lift over Baseline: 3.41x  |  Cold-start gap: warm MAP@12 is 3.1x higher than cold users")

# ════════════════════════════════════════════════
# PAGE 4 — LightGBM
# ════════════════════════════════════════════════
pdf.add_page()
section_header(pdf, "Model 3: LightGBM Learning-to-Rank", "Gradient Boosting Binary Classifier for Top-K Retrieval")

h2(pdf, "3.1 Introduction")
body(pdf, (
    "LightGBM reframes recommendation as supervised binary classification: given a (user, item) candidate "
    "pair, predict the purchase probability. A two-stage pipeline — offline feature engineering then "
    "fast LightGBM inference — enables 21 multi-dimensional features including semantic Word2Vec similarity."
))

h2(pdf, "3.2 Stage 1 — Candidate Generation (05_feature_engineering.py)")
steps = [
    {"title": "Candidate Generation — 3 strategies, capped at 100/user",
     "lines": ["Repurchase: items user already bought in training window",
               "Popularity: Top-50 globally purchased items (last 7 days)",
               "Item-CF: batched cosine similarity on item-user sparse matrix (last 6 weeks)"]},
    {"title": "Feature Engineering — 21 features per (user, item) pair",
     "lines": ["User-Item: ui_purchase_count, ui_days_since_last, ui_tw_1w/1m/1s",
               "Item: item_price_mean/max/min, item_total_sales, item_age_mean/max/min",
               "User: user_total_purchases, user_unique_items, user_avg_price, user_active_days, age",
               "Semantic: w2v_cosine_sim (Word2Vec 64-dim, window=5, min_count=3, epochs=5)",
               "Cumulative: cum_purchase_count, cum_avg_price (inactive users)"]},
    {"title": "Label Assignment & Downsampling",
     "lines": ["Train label=1: bought in last 7 days of train period",
               "Test label=1: bought in test_processed (final 28 days)",
               "Negatives downsampled to 1,500,000  |  Train: (1,515,896 x 24)  |  Test: (21,825,652 x 24)"]},
]
pipeline_box(pdf, steps)

h2(pdf, "3.3 Stage 2 — LightGBM Training & Inference")
steps2 = [
    {"title": "LightGBM Binary Classifier Training",
     "lines": ["Objective: binary  |  Metric: AUC  |  Learning rate: 0.01  |  num_leaves: 255",
               "feature_fraction: 0.8  |  bagging_fraction: 0.8  |  Early stopping: 100 rounds",
               "Best iteration: 840  |  Best val AUC: 0.8596  |  Training time: 84.9s"]},
    {"title": "Scoring & Top-K Extraction",
     "lines": ["Score all 21.8M test candidates  |  Sort per user  |  Extract Top-12"]},
]
pipeline_box(pdf, steps2)

h2(pdf, "3.4 Top-10 Feature Importance (by Gain)")
hdrs3 = ["Rank","Feature","Description","Type"]
wids3 = [12,45,80,30]
pdf.set_fill_color(*DBLUE)
pdf.set_text_color(255,255,255)
pdf.set_font("Helvetica","B",8)
for c,w in zip(hdrs3,wids3):
    pdf.cell(w,5,c,border=1,fill=True)
pdf.ln()
pdf.set_text_color(*BLACK)
feat_rows = [
    ["1","item_total_sales","Total historical units sold","Item popularity"],
    ["2","w2v_cosine_sim","Word2Vec user-item semantic similarity","Semantic"],
    ["3","ui_tw_1s","Time-weighted interaction (90 days)","Interaction"],
    ["4","user_avg_price","User's average purchase price","User profile"],
    ["5","item_age_mean","Mean age of item buyers","Item profile"],
    ["6","age","User's age","User profile"],
    ["7","item_price_mean","Item's mean price","Item profile"],
    ["8","ui_days_since_last","Days since last purchase of item","Recency"],
    ["9","user_total_purchases","User's total purchase count","User activity"],
    ["10","item_price_min","Item's minimum historical price","Item profile"],
]
for i,r in enumerate(feat_rows):
    bg = LGRAY if i%2==0 else (255,255,255)
    table_row(pdf,r,wids3,bg=bg)

pdf.ln(3)
h2(pdf, "3.5 Performance")
metric_badge(pdf, "MAP@12", "0.013956", GREEN)
pdf.ln(1)
metric_badge(pdf, "Hit Rate@12", "9.33%  (22,392 / 239,951 users)", BLUE)
pdf.ln(1)
metric_badge(pdf, "Best AUC", "0.8596", DBLUE)
pdf.ln(1)
metric_badge(pdf, "Lift vs Baseline", "4.07x  |  Lift vs ALS: +19.4%", (80,0,0))
pdf.ln(5)

# ════════════════════════════════════════════════
# PAGE 5 — Comparison + Feature Importance Chart
# ════════════════════════════════════════════════
pdf.add_page()
section_header(pdf, "Model Comparison & Key Findings")

h2(pdf, "4.1 Head-to-Head Comparison")
hdrs4 = ["Dimension","MBA Baseline","ALS","LightGBM"]
wids4 = [48,44,40,45]
pdf.set_fill_color(*DBLUE)
pdf.set_text_color(255,255,255)
pdf.set_font("Helvetica","B",8)
for c,w in zip(hdrs4,wids4):
    pdf.cell(w,5,c,border=1,fill=True)
pdf.ln()
pdf.set_text_color(*BLACK)
comp_rows = [
    ["Personalization","Demographic-only","Per-user latent factors","Per-user multi-feature"],
    ["Cold-start","Age-bucketed trending","Global popularity","Popularity + features"],
    ["Interpretability","High (Lift/Confidence)","Low (latent space)","Medium (feat importance)"],
    ["Feature richness","Statistical rules","Interaction history","21 multi-dim features"],
    ["Business prescriptive","High (ECR, VIP logic)","None","Low (pure ranking)"],
    ["Inference speed","Very fast (O(1))","Medium (matmul)","Fast (pre-built features)"],
    ["Training time","~seconds","88.6s","84.9s"],
    ["MAP@12","0.003427","0.011692","0.013956"],
    ["Hit Rate@12","4.32%","7.81%","9.33%"],
]
for i,r in enumerate(comp_rows):
    bg = LGRAY if i%2==0 else (255,255,255)
    table_row(pdf,r,wids4,bg=bg)

pdf.ln(4)
h2(pdf, "4.2 Key Findings")
findings = [
    "Personalization drives the biggest gain: ALS delivers 3.41x MAP@12 uplift over the non-personalized baseline.",
    "Rich features further close the gap: LightGBM's 21 features push MAP@12 19.4% beyond ALS.",
    "Word2Vec semantic similarity is the 2nd most important feature — purchase sequences contain rich item signals.",
    "Cold-start remains a bottleneck: ALS warm MAP@12 (0.012522) is 3.1x higher than cold (0.004040).",
    "MBA Baseline still adds business value: explainable rules + ECR prescriptive VIP logic unavailable in ML models.",
]
for f in findings:
    bullet(pdf, f)

pdf.ln(3)
h2(pdf, "4.3 Feature Importance Chart (LightGBM)")
if IMG.exists():
    pdf.image(str(IMG), x=15, w=175)

# ════════════════════════════════════════════════
# PAGE 6 — Future Work & Conclusion
# ════════════════════════════════════════════════
pdf.add_page()
section_header(pdf, "Future Improvements & Conclusion")

h2(pdf, "5.1 Future Improvements")
hdrs5 = ["Improvement","Expected Impact","Complexity"]
wids5 = [65,70,35]
pdf.set_fill_color(*DBLUE)
pdf.set_text_color(255,255,255)
pdf.set_font("Helvetica","B",8)
for c,w in zip(hdrs5,wids5):
    pdf.cell(w,5,c,border=1,fill=True)
pdf.ln()
pdf.set_text_color(*BLACK)
future_rows = [
    ["Two-Tower Neural Network","Higher MAP via deep feature interactions","High"],
    ["LightGBM LambdaRank","Direct MAP optimization vs. AUC proxy","Medium"],
    ["Sequential models (GRU4Rec)","Capture purchase sequence dynamics","High"],
    ["Content-based cold-start","Close warm/cold MAP gap (currently 3.1x)","Medium"],
    ["Ensemble ALS + LightGBM","Combine latent factors + rich features","Low"],
    ["Online / real-time learning","React to intra-session signals","Very High"],
]
for i,r in enumerate(future_rows):
    bg = LGRAY if i%2==0 else (255,255,255)
    table_row(pdf,r,wids5,bg=bg)

pdf.ln(5)
h1(pdf, "6. Conclusion")
body(pdf, (
    "The three-model progression demonstrates a clear and measurable improvement hierarchy for the H&M "
    "recommendation system, evaluated on an identical 239,951-user holdout set:"
))
bullet(pdf, "MBA Baseline (MAP@12: 0.003427) — Establishes a rules-based, explainable benchmark with ECR-driven prescriptive VIP business logic.")
bullet(pdf, "ALS Collaborative Filtering (MAP@12: 0.011692) — Introduces true personalization via implicit matrix factorization, delivering a 3.41x improvement.")
bullet(pdf, "LightGBM Learning-to-Rank (MAP@12: 0.013956) — Achieves best offline performance through rich multi-dimensional features and gradient boosting, reaching 4.07x over baseline.")
pdf.ln(3)
body(pdf, (
    "Results confirm that personalization and feature richness are the primary drivers of recommendation "
    "quality in H&M's fashion retail context. The warm/cold user gap (3.1x) highlights cold-start "
    "resolution as the highest-priority area for future work."
))

pdf.ln(4)
pdf.set_fill_color(*LGRAY)
pdf.set_font("Helvetica","I",8)
pdf.set_text_color(*MGRAY)
pdf.multi_cell(0,5,(
    "Report generated from: notebooks/03_MBA_Baseline.ipynb  |  notebooks/4_RecSys_ALS.ipynb  |  "
    "notebooks/05_LightGBM_RecSys.ipynb  |  scripts/05_feature_engineering.py"
), fill=True)

pdf.output(str(OUT))
print(f"PDF saved to: {OUT}")
