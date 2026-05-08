"""
05_feature_engineering.py
=========================
Builds LightGBM feature matrices for ALL users in train_processed and test_processed.

Outputs (saved to data/processed/):
  lgbm_train_features.parquet  — candidates + features for users in train split
                                  label = 1 if bought in last 7 days of train
  lgbm_test_features.parquet   — candidates + features for users in test_processed
                                  label = 1 if bought in test_processed (final 28 days)

Run from project root:
  python scripts/05_feature_engineering.py
"""

import warnings, time, gc
from pathlib import Path
from datetime import timedelta

import numpy as np
import polars as pl
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from rich.console import Console

warnings.filterwarnings("ignore")
np.random.seed(42)

console = Console()

# ── Paths ──────────────────────────────────────────────────────────────────
cwd = Path.cwd()
if (cwd / "data" / "processed").exists():
    ROOT = cwd
elif (cwd.parent / "data" / "processed").exists():
    ROOT = cwd.parent
else:
    raise FileNotFoundError("Cannot find data/processed from current directory")

DATA = ROOT / "data" / "processed"
TRAIN_PATH = DATA / "train_processed.parquet"
TEST_PATH = DATA / "test_processed.parquet"
CUST_PATH = DATA / "customers_processed.parquet"
OUT_TRAIN = DATA / "lgbm_train_features.parquet"
OUT_TEST = DATA / "lgbm_test_features.parquet"

# ── Constants ───────────────────────────────────────────────────────────────
N_CAND = 100  # max candidates per user
N_SIM = 20  # similar items per item in CF
BATCH = 500  # item batch size for cosine similarity
NEG_SIZE = 1_500_000  # max negatives to keep in train split
RS = 42  # random state


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Candidate Generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_candidates(
    feat_df: pl.DataFrame,
    target_users: list[int],
) -> pl.DataFrame:
    """
    Generates up to N_CAND (customer_id, article_id, src) candidates per user.

    Strategies:
      - repurchase : items user already bought in feat_df
      - popularity : Top-50 globally popular items from last 7 days of feat_df
      - cf         : Item-to-Item cosine similarity (batched, memory-safe)
    """
    feat_max_date = feat_df["t_dat"].max()
    console.print(
        f"  [dim]feat_max_date = {feat_max_date}, target_users = {len(target_users):,}[/dim]"
    )

    # 1a. Repurchase
    repurch = (
        feat_df.filter(pl.col("customer_id").is_in(target_users))
        .group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("article_id"))
        .explode("article_id")
        .with_columns(pl.lit("repurchase").alias("src"))
    )
    console.print(f"  Repurchase: {len(repurch):,} pairs")

    # 1b. Popularity (last 7 days of feature period)
    last_wk = feat_max_date - timedelta(days=6)
    pop_items = (
        feat_df.filter(pl.col("t_dat") >= last_wk)
        .group_by("article_id")
        .agg(pl.col("quantity").sum().alias("cnt"))
        .sort("cnt", descending=True)
        .head(50)["article_id"]
        .to_list()
    )
    pop = (
        pl.DataFrame({"customer_id": target_users}, schema={"customer_id": pl.Int32})
        .join(
            pl.DataFrame({"article_id": pop_items}, schema={"article_id": pl.Int32}),
            how="cross",
        )
        .with_columns(pl.lit("popularity").alias("src"))
    )
    console.print(f"  Popularity: {len(pop):,} pairs")

    # 1c. Item-CF (last 6 weeks of feat_df → batched cosine sim)
    feat_6w = feat_df.filter(pl.col("t_dat") >= feat_max_date - timedelta(days=41))
    feat_items = feat_6w["article_id"].unique().sort().to_list()
    feat_users_all = feat_6w["customer_id"].unique().sort().to_list()
    i2idx = {a: i for i, a in enumerate(feat_items)}
    u2idx = {u: i for i, u in enumerate(feat_users_all)}

    rows = feat_6w["customer_id"].replace_strict(u2idx).to_numpy().astype(np.int32)
    cols = feat_6w["article_id"].replace_strict(i2idx).to_numpy().astype(np.int32)
    ui_sp = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(feat_users_all), len(feat_items)),
    )
    iu_sp = ui_sp.T.tocsr()
    del rows, cols, feat_6w
    gc.collect()

    # Batched cosine → top-N similar items per item
    n_items_f = len(feat_items)
    sim_dict: dict[int, list[tuple[int, float]]] = {}
    for s in range(0, n_items_f, BATCH):
        e = min(s + BATCH, n_items_f)
        sims = sk_cosine(iu_sp[s:e], iu_sp)
        for i in range(e - s):
            row = sims[i]
            row[s + i] = 0.0
            if row.max() == 0:
                continue
            topk = np.argpartition(row, -N_SIM)[-N_SIM:]
            topk = topk[row[topk] > 0]
            sim_dict[s + i] = [(int(j), float(row[j])) for j in topk]
        del sims
        gc.collect()
        if (s // BATCH) % 20 == 0:
            console.print(f"  [dim]CF batch {s // BATCH}/{n_items_f // BATCH}[/dim]")
    del iu_sp, ui_sp
    gc.collect()

    # For each target user, collect CF candidates
    user_hist = (
        feat_df.filter(pl.col("customer_id").is_in(target_users))
        .group_by("customer_id")
        .agg(pl.col("article_id").unique().alias("hist"))
    )
    cf_rows: list[dict] = []
    for r in user_hist.iter_rows(named=True):
        uid, hist = r["customer_id"], set(r["hist"])
        scored: dict[int, float] = {}
        for aid in hist:
            if aid not in i2idx:
                continue
            for j, sc in sim_dict.get(i2idx[aid], []):
                real_j = feat_items[j]
                if real_j not in hist:
                    scored[real_j] = max(scored.get(real_j, 0.0), sc)
        for a in sorted(scored, key=scored.get, reverse=True)[:30]:
            cf_rows.append({"customer_id": uid, "article_id": a, "src": "cf"})
    del sim_dict
    gc.collect()

    cf_cands = (
        pl.DataFrame(cf_rows)
        if cf_rows
        else pl.DataFrame(
            schema={"customer_id": pl.Int32, "article_id": pl.Int32, "src": pl.Utf8}
        )
    )
    cf_cands = cf_cands.with_columns(
        [
            pl.col("customer_id").cast(pl.Int32),
            pl.col("article_id").cast(pl.Int32),
        ]
    )
    console.print(f"  Item-CF: {len(cf_cands):,} pairs")

    # Merge & dedup, cap at N_CAND
    cast_expr = [
        pl.col("customer_id").cast(pl.Int32),
        pl.col("article_id").cast(pl.Int32),
    ]
    all_cands = pl.concat(
        [
            repurch.select(["customer_id", "article_id", "src"]).with_columns(
                cast_expr
            ),
            pop.select(["customer_id", "article_id", "src"]).with_columns(cast_expr),
            cf_cands.select(["customer_id", "article_id", "src"]).with_columns(
                cast_expr
            ),
        ]
    ).unique(subset=["customer_id", "article_id"], keep="first")

    all_cands = (
        all_cands.with_columns(
            pl.col("article_id").cum_count().over("customer_id").alias("rn")
        )
        .filter(pl.col("rn") <= N_CAND)
        .drop("rn")
    )
    console.print(f"  Total candidates: {len(all_cands):,}")
    return all_cands


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
def build_features(
    all_cands: pl.DataFrame,
    feat_df: pl.DataFrame,
    customers_df: pl.DataFrame,
) -> pd.DataFrame:
    """Attaches interaction, aggregation, W2V similarity and cumulative features."""
    feat_max_date = feat_df["t_dat"].max()
    w1 = feat_max_date - timedelta(days=6)
    m1 = feat_max_date - timedelta(days=29)
    s1 = feat_max_date - timedelta(days=89)

    # 2.1 User-Item interaction
    ui_full = feat_df.group_by(["customer_id", "article_id"]).agg(
        [
            pl.len().alias("ui_purchase_count"),
            pl.col("days_ago").min().alias("ui_days_since_last"),
        ]
    )
    ui_1w = (
        feat_df.filter(pl.col("t_dat") >= w1)
        .group_by(["customer_id", "article_id"])
        .agg(
            (pl.col("quantity").cast(pl.Float32) * pl.col("time_weight"))
            .sum()
            .alias("ui_tw_1w")
        )
    )
    ui_1m = (
        feat_df.filter(pl.col("t_dat") >= m1)
        .group_by(["customer_id", "article_id"])
        .agg(
            (pl.col("quantity").cast(pl.Float32) * pl.col("time_weight"))
            .sum()
            .alias("ui_tw_1m")
        )
    )
    ui_1s = (
        feat_df.filter(pl.col("t_dat") >= s1)
        .group_by(["customer_id", "article_id"])
        .agg(
            (pl.col("quantity").cast(pl.Float32) * pl.col("time_weight"))
            .sum()
            .alias("ui_tw_1s")
        )
    )

    cands = all_cands.join(ui_full, on=["customer_id", "article_id"], how="left")
    cands = cands.join(ui_1w, on=["customer_id", "article_id"], how="left")
    cands = cands.join(ui_1m, on=["customer_id", "article_id"], how="left")
    cands = cands.join(ui_1s, on=["customer_id", "article_id"], how="left")
    del ui_full, ui_1w, ui_1m, ui_1s
    gc.collect()

    # 2.2 Item aggregation
    item_price = feat_df.group_by("article_id").agg(
        [
            pl.col("price").mean().cast(pl.Float32).alias("item_price_mean"),
            pl.col("price").max().cast(pl.Float32).alias("item_price_max"),
            pl.col("price").min().cast(pl.Float32).alias("item_price_min"),
            pl.col("quantity").sum().alias("item_total_sales"),
        ]
    )
    buyer_age = (
        feat_df.select(["customer_id", "article_id"])
        .unique()
        .join(customers_df.select(["customer_id", "age"]), on="customer_id", how="left")
        .group_by("article_id")
        .agg(
            [
                pl.col("age").mean().cast(pl.Float32).alias("item_age_mean"),
                pl.col("age").max().cast(pl.Float32).alias("item_age_max"),
                pl.col("age").min().cast(pl.Float32).alias("item_age_min"),
            ]
        )
    )
    cands = cands.join(item_price, on="article_id", how="left")
    cands = cands.join(buyer_age, on="article_id", how="left")
    del item_price, buyer_age
    gc.collect()

    # 2.3 User aggregation
    user_agg = feat_df.group_by("customer_id").agg(
        [
            pl.col("quantity").sum().alias("user_total_purchases"),
            pl.col("article_id").n_unique().alias("user_unique_items"),
            pl.col("price").mean().cast(pl.Float32).alias("user_avg_price"),
            pl.col("t_dat").n_unique().alias("user_active_days"),
        ]
    )
    cands = cands.join(user_agg, on="customer_id", how="left")
    cands = cands.join(
        customers_df.select(["customer_id", "age"]), on="customer_id", how="left"
    )
    del user_agg
    gc.collect()

    # 2.4 Word2Vec similarity
    console.print("  [dim]Training Word2Vec...[/dim]")
    sequences = (
        feat_df.sort("t_dat")
        .group_by("customer_id")
        .agg(pl.col("article_id").alias("seq"))
    )
    sentences = [list(map(str, s)) for s in sequences["seq"].to_list()]
    w2v = Word2Vec(
        sentences,
        vector_size=64,
        window=5,
        min_count=3,
        sg=1,
        workers=4,
        seed=RS,
        epochs=5,
    )
    item_embs = {int(w): w2v.wv[w] for w in w2v.wv.key_to_index}
    user_emb_dict: dict[int, np.ndarray] = {}
    for r in sequences.iter_rows(named=True):
        vecs = [item_embs[a] for a in r["seq"] if a in item_embs]
        if vecs:
            user_emb_dict[r["customer_id"]] = np.mean(vecs, axis=0)
    del sentences, sequences
    gc.collect()

    cands_pd = cands.to_pandas()
    cos_sims = np.zeros(len(cands_pd), dtype=np.float32)
    for i, (uid, aid) in enumerate(
        zip(cands_pd["customer_id"], cands_pd["article_id"])
    ):
        if uid in user_emb_dict and aid in item_embs:
            u_v, i_v = user_emb_dict[uid], item_embs[aid]
            norm = np.linalg.norm(u_v) * np.linalg.norm(i_v)
            cos_sims[i] = float(np.dot(u_v, i_v) / norm) if norm > 0 else 0.0
    cands_pd["w2v_cosine_sim"] = cos_sims
    del cos_sims, user_emb_dict, item_embs, w2v
    gc.collect()

    # 2.5 Cumulative features (inactive users = no purchase in last 30 days)
    recent_users = set(
        feat_df.filter(pl.col("t_dat") >= m1)["customer_id"].unique().to_list()
    )
    inactive_mask = ~cands_pd["customer_id"].isin(recent_users)
    full_hist_pd = (
        feat_df.group_by(["customer_id", "article_id"])
        .agg(
            [
                pl.len().alias("cum_purchase_count"),
                pl.col("price").mean().cast(pl.Float32).alias("cum_avg_price"),
            ]
        )
        .to_pandas()
        .set_index(["customer_id", "article_id"])
    )

    cands_pd["cum_purchase_count"] = 0.0
    cands_pd["cum_avg_price"] = 0.0
    if inactive_mask.any():
        for idx in cands_pd.index[inactive_mask]:
            key = (
                int(cands_pd.at[idx, "customer_id"]),
                int(cands_pd.at[idx, "article_id"]),
            )
            if key in full_hist_pd.index:
                cands_pd.at[idx, "cum_purchase_count"] = full_hist_pd.at[
                    key, "cum_purchase_count"
                ]
                cands_pd.at[idx, "cum_avg_price"] = full_hist_pd.at[
                    key, "cum_avg_price"
                ]
    del full_hist_pd
    gc.collect()

    return cands_pd


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Downsample & dtype reduction
# ═══════════════════════════════════════════════════════════════════════════
def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        col = df[c]
        if col.dtype == "float64":
            df[c] = col.astype(np.float32)
        elif col.dtype in ("int64", "int32"):
            mn, mx = col.min(), col.max()
            if mn >= np.iinfo(np.int8).min and mx <= np.iinfo(np.int8).max:
                df[c] = col.astype(np.int8)
            elif mn >= np.iinfo(np.int16).min and mx <= np.iinfo(np.int16).max:
                df[c] = col.astype(np.int16)
            else:
                df[c] = col.astype(np.int32)
    return df


def finalize(cands_pd: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """Label-encode src, downsample negatives (train only), reduce dtypes."""
    cands_pd["src"] = LabelEncoder().fit_transform(cands_pd["src"]).astype(np.int8)
    if is_train:
        pos = cands_pd[cands_pd["label"] == 1]
        neg = cands_pd[cands_pd["label"] == 0]
        if len(neg) > NEG_SIZE:
            neg = neg.sample(n=NEG_SIZE, random_state=RS)
        cands_pd = (
            pd.concat([pos, neg], ignore_index=True)
            .sample(frac=1, random_state=RS)
            .reset_index(drop=True)
        )
    return reduce_mem(cands_pd)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    console.rule("[bold cyan]LightGBM Feature Engineering Pipeline[/bold cyan]")
    total_t = time.time()

    console.print("[cyan]Loading data...[/cyan]")
    train_full = pl.read_parquet(TRAIN_PATH)
    test_full = pl.read_parquet(TEST_PATH)
    customers_df = pl.read_parquet(CUST_PATH)

    train_max_date = train_full["t_dat"].max()
    console.print(
        f"  Train date range : {train_full['t_dat'].min()} → {train_max_date}"
    )
    console.print(
        f"  Test date range  : {test_full['t_dat'].min()} → {test_full['t_dat'].max()}"
    )
    console.print(f"  Train users      : {train_full['customer_id'].n_unique():,}")
    console.print(f"  Test users       : {test_full['customer_id'].n_unique():,}")

    # ── TRAIN SPLIT ─────────────────────────────────────────────────────────
    console.rule("[bold yellow]Building TRAIN features[/bold yellow]")
    t0 = time.time()

    # Label window = last 7 days of train
    label_start = train_max_date - timedelta(days=6)
    train_feat = train_full.filter(pl.col("t_dat") < label_start)
    train_labels = train_full.filter(pl.col("t_dat") >= label_start)

    # Target users = ALL users in train_processed who appear in the label window
    train_target_users = train_labels["customer_id"].unique().to_list()
    console.print(f"  Train label users: {len(train_target_users):,}")

    train_cands = generate_candidates(train_feat, train_target_users)
    label_pairs = (
        train_labels.select(["customer_id", "article_id"])
        .unique()
        .with_columns(pl.lit(1).alias("label").cast(pl.Int8))
    )
    train_cands = train_cands.join(
        label_pairs, on=["customer_id", "article_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))

    train_pd = build_features(train_cands, train_feat, customers_df)
    train_pd = finalize(train_pd, is_train=True)

    train_pd.to_parquet(OUT_TRAIN, index=False, compression="zstd")
    console.print(f"[green]✓ Train features saved → {OUT_TRAIN.name}[/green]")
    console.print(
        f"  Shape : {train_pd.shape}  |  Pos: {(train_pd['label'] == 1).sum():,}  |  Neg: {(train_pd['label'] == 0).sum():,}"
    )
    console.print(f"  Time  : {time.time() - t0:.1f}s")
    del train_feat, train_labels, train_cands, train_pd
    gc.collect()

    # ── TEST SPLIT ──────────────────────────────────────────────────────────
    console.rule("[bold yellow]Building TEST features[/bold yellow]")
    t0 = time.time()

    # feat_df = entire train_full, label_df = test_processed (final 28 days)
    # Target users = ALL users in test_processed (must match ALS: 239,951)
    test_target_users = test_full["customer_id"].unique().to_list()
    console.print(f"  Test target users: {len(test_target_users):,}")

    test_cands = generate_candidates(train_full, test_target_users)
    label_pairs_test = (
        test_full.select(["customer_id", "article_id"])
        .unique()
        .with_columns(pl.lit(1).alias("label").cast(pl.Int8))
    )
    test_cands = test_cands.join(
        label_pairs_test, on=["customer_id", "article_id"], how="left"
    ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))

    test_pd = build_features(test_cands, train_full, customers_df)
    test_pd = finalize(test_pd, is_train=False)  # no downsampling for test

    test_pd.to_parquet(OUT_TEST, index=False, compression="zstd")
    console.print(f"[green]✓ Test features saved → {OUT_TEST.name}[/green]")
    console.print(
        f"  Shape : {test_pd.shape}  |  Users: {test_pd['customer_id'].nunique():,}"
    )
    console.print(f"  Time  : {time.time() - t0:.1f}s")

    console.rule(f"[bold green]Done — Total: {time.time() - total_t:.0f}s[/bold green]")


if __name__ == "__main__":
    main()
