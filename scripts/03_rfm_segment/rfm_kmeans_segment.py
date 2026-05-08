from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
FINAL_K = 5


def resolve_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        return cwd
    if (cwd.parent / "data" / "processed").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate data/processed from current working directory.")


def load_rfm(train_path: Path, customers_path: Path) -> pl.DataFrame:
    train_lf = pl.scan_parquet(str(train_path))
    customers_lf = pl.scan_parquet(str(customers_path)).select(["customer_id", "age"])

    rfm_lf = (
        train_lf
        .group_by("customer_id")
        .agg([
            pl.col("days_ago").min().alias("R"),
            pl.col("t_dat").n_unique().alias("F"),
            (pl.col("price") * pl.col("quantity")).sum().alias("M"),
        ])
        .join(customers_lf, on="customer_id", how="left")
        .select(["customer_id", "R", "F", "M", "age"])
    )

    rfm_df = rfm_lf.collect()

    # Defensive age handling in case unexpected nulls are present.
    if rfm_df.get_column("age").null_count() > 0:
        non_null_age = rfm_df.get_column("age").drop_nulls()
        if len(non_null_age) == 0:
            raise ValueError("Cannot compute fallback age: all active customers have null age.")
        fallback_age = int(non_null_age.median())
        rfm_df = rfm_df.with_columns(pl.col("age").fill_null(fallback_age))
        print(f"Filled null age values with fallback median age: {fallback_age}")

    rfm_df = rfm_df.with_columns([
        pl.col("R").cast(pl.Float64),
        pl.col("F").cast(pl.Float64),
        pl.col("M").cast(pl.Float64),
        pl.col("age").cast(pl.Float64),
    ])

    return rfm_df


def build_feature_matrix(rfm_df: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    model_df = rfm_df.select(["customer_id", "R", "F", "M", "age"])

    rfm_matrix = model_df.select(["R", "F", "M"]).to_numpy()
    age_matrix = model_df.select(["age"]).to_numpy()

    power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    rfm_transformed = power_transformer.fit_transform(rfm_matrix)

    X_pre_scaled = np.hstack([rfm_transformed, age_matrix])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pre_scaled)

    return model_df, X_scaled


def assign_clusters(X_scaled: np.ndarray) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=FINAL_K,
        init="k-means++",
        random_state=SEED,
        n_init=10,
    )
    return kmeans.fit_predict(X_scaled).astype(np.int32)


def label_clusters(model_df: pl.DataFrame, cluster_id: np.ndarray) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    clustered_df = model_df.with_columns(pl.Series(name="cluster_id", values=cluster_id))

    cluster_profile_df = (
        clustered_df
        .group_by("cluster_id")
        .agg([
            pl.col("R").mean().alias("R_mean"),
            pl.col("R").median().alias("R_median"),
            pl.col("F").mean().alias("F_mean"),
            pl.col("F").median().alias("F_median"),
            pl.col("M").mean().alias("M_mean"),
            pl.col("M").median().alias("M_median"),
            pl.col("age").mean().alias("age_mean"),
            pl.col("age").median().alias("age_median"),
            pl.len().alias("n_customers"),
        ])
        .sort("cluster_id")
    )

    global_medians = clustered_df.select([
        pl.col("R").median().alias("global_R_median"),
        pl.col("F").median().alias("global_F_median"),
        pl.col("M").median().alias("global_M_median"),
        pl.col("age").median().alias("global_age_median"),
    ]).to_dicts()[0]

    profile_df = cluster_profile_df.select([
        "cluster_id", "R_median", "F_median", "M_median", "age_median"
    ]).sort("cluster_id")

    global_f = float(global_medians["global_F_median"])
    global_age = float(global_medians["global_age_median"])

    remaining = set(profile_df.get_column("cluster_id").to_list())
    base_labels = {int(cid): "Core" for cid in profile_df.get_column("cluster_id").to_list()}

    def sort_remaining(df: pl.DataFrame, remaining_ids: set[int]) -> pl.DataFrame:
        return df.filter(pl.col("cluster_id").is_in(list(remaining_ids)))

    def pick_first(df: pl.DataFrame) -> int | None:
        if df.height == 0:
            return None
        return int(df["cluster_id"][0])

    whales_candidates = (
        sort_remaining(profile_df, remaining)
        .sort(["M_median", "R_median", "F_median"], descending=[True, False, True])
    )
    whales_cid = pick_first(whales_candidates)
    if whales_cid is not None:
        base_labels[whales_cid] = "Whales"
        remaining.remove(whales_cid)

    risk_candidates = (
        sort_remaining(profile_df, remaining)
        .sort(["R_median", "F_median", "M_median"], descending=[True, False, False])
    )
    at_risk_cid = pick_first(risk_candidates)
    if at_risk_cid is not None:
        base_labels[at_risk_cid] = "At-Risk"
        remaining.remove(at_risk_cid)

    champ_pool = sort_remaining(profile_df, remaining).filter(pl.col("F_median") >= global_f)
    if champ_pool.height == 0:
        champ_pool = sort_remaining(profile_df, remaining)
    champions_candidates = champ_pool.sort(["R_median", "M_median", "F_median"], descending=[False, True, True])
    champions_cid = pick_first(champions_candidates)
    if champions_cid is not None:
        base_labels[champions_cid] = "Champions"
        remaining.remove(champions_cid)

    newcomers_candidates = (
        sort_remaining(profile_df, remaining)
        .sort(["F_median", "R_median", "M_median"], descending=[False, False, False])
    )
    newcomers_cid = pick_first(newcomers_candidates)
    if newcomers_cid is not None:
        base_labels[newcomers_cid] = "Newcomers"
        remaining.remove(newcomers_cid)

    for cid in sorted(remaining):
        base_labels[cid] = "Core"

    age_by_cluster = {
        int(row["cluster_id"]): float(row["age_median"])
        for row in profile_df.iter_rows(named=True)
    }

    final_label_map = {}
    for cid, base in base_labels.items():
        prefix = "Young" if age_by_cluster[cid] < global_age else "Mature"
        final_label_map[cid] = f"{prefix} {base}"

    label_df = pl.DataFrame({
        "cluster_id": list(final_label_map.keys()),
        "segment_label": list(final_label_map.values()),
    })

    segmented_df = clustered_df.join(label_df, on="cluster_id", how="left")

    return segmented_df, cluster_profile_df, global_medians


def compute_metrics(segmented_df: pl.DataFrame) -> None:
    revenue_by_segment = (
        segmented_df
        .group_by("segment_label")
        .agg(pl.col("M").sum().alias("segment_revenue"))
    )
    total_revenue = float(revenue_by_segment.select(pl.col("segment_revenue").sum()).item())

    revenue_share_df = (
        revenue_by_segment
        .with_columns((pl.col("segment_revenue") / total_revenue * 100).alias("revenue_share_pct"))
        .sort("revenue_share_pct", descending=True)
    )

    print("Revenue share by segment (% of total M):")
    print(revenue_share_df)

    high_value_mask = pl.col("segment_label").str.contains("Whales|Champions")
    low_value_mask = pl.col("segment_label").str.contains("Core|Newcomers")

    high_value_q25 = segmented_df.filter(high_value_mask).select(pl.col("M").quantile(0.25)).item()
    low_value_q75 = segmented_df.filter(low_value_mask).select(pl.col("M").quantile(0.75)).item()

    print("High-value segments (Whales/Champions) M 25th percentile:", high_value_q25)
    print("Low-value segments (Core/Newcomers) M 75th percentile:", low_value_q75)


def main() -> None:
    project_root = resolve_project_root()
    data_dir = project_root / "data" / "processed"
    train_path = data_dir / "train_processed.parquet"
    customers_path = data_dir / "customers_processed.parquet"
    output_path = data_dir / "customers_segmented.parquet"

    required_files = [train_path, customers_path]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required processed artifacts:\n"
            + "\n".join(missing)
            + "\n\nPlease generate these files before running this script."
        )

    np.random.seed(SEED)

    rfm_df = load_rfm(train_path, customers_path)
    model_df, X_scaled = build_feature_matrix(rfm_df)
    cluster_id = assign_clusters(X_scaled)
    segmented_df, _, _ = label_clusters(model_df, cluster_id)

    final_df = (
        segmented_df
        .select(["customer_id", "R", "F", "M", "age", "cluster_id", "segment_label"])
        .with_columns([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("R").cast(pl.Float64),
            pl.col("F").cast(pl.Float64),
            pl.col("M").cast(pl.Float64),
            pl.col("age").round(0).cast(pl.Int8),
            pl.col("cluster_id").cast(pl.Int32),
        ])
    )

    assert final_df.height == rfm_df.height, "Mismatch in customer counts after segmentation."
    assert final_df.get_column("customer_id").is_duplicated().sum() == 0, "Duplicate customer_id in output."

    null_counts = final_df.select([pl.col(c).is_null().sum().alias(c) for c in final_df.columns])
    total_nulls = int(sum(null_counts.row(0)))
    assert total_nulls == 0, f"Output contains null values: {null_counts}"

    inf_counts = final_df.select([
        pl.col("R").is_infinite().sum().alias("R_inf"),
        pl.col("F").is_infinite().sum().alias("F_inf"),
        pl.col("M").is_infinite().sum().alias("M_inf"),
    ])
    assert int(sum(inf_counts.row(0))) == 0, f"Output contains inf values: {inf_counts}"

    assert final_df.get_column("segment_label").null_count() == 0, "Missing segment labels in final output."

    final_df.write_parquet(output_path)

    print("Export complete:", output_path)
    print("Rows exported:", final_df.height)
    print("Distinct segments:", final_df.get_column("segment_label").n_unique())
    print("Output schema:", final_df.schema)

    compute_metrics(segmented_df)


if __name__ == "__main__":
    main()
