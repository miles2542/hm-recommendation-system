import polars as pl
import json
from pathlib import Path
from rich.console import Console

# --- Configuration & Constants ---
PROCESSED_DIR = Path(r"F:\PYTHON\hm-recommendation-system\data\processed")
OUTPUT_DIR = Path(r"F:\PYTHON\hm-recommendation-system\data\processed")
ID_MAPPING_PATH = OUTPUT_DIR / "id_mapping.json"

SUPER_CODE = "2c29ae653a9282cce4151bd87643c911a68d276429a91f8ca0d4b6efa8100"
RESELLER_CUTOFF = 1000
TEST_WINDOW_DAYS = 28
DECAY_HALF_LIFE_RATE = 0.055

AGE_MAPPING = {
    "Lingeries/Tights": 29,
    "Divided": 29,
    "Ladies Accessories": 30,
    "Sport": 31,
    "Baby Sizes 50-98": 33,
    "Ladieswear": 33,
    "Menswear": 34,
    "Children Sizes 92-140": 38,
    "Children Accessories": 39,
    "Children Sizes 134-170": 45,
}

pl.Config.set_ascii_tables(True)
console = Console()


def create_id_mappings(
    cust: pl.LazyFrame, art: pl.LazyFrame
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Generates contiguous Int32 IDs for memory efficiency. Evaluates eagerly."""
    console.print("[cyan]Creating ID Mappings...[/cyan]")

    # Evaluate to get mappings
    cust_map = (
        cust.select("customer_id")
        .unique()
        .collect()
        .with_row_index("customer_int_id")
        .with_columns(pl.col("customer_int_id").cast(pl.Int32))
    )
    art_map = (
        art.select("article_id")
        .unique()
        .collect()
        .with_row_index("article_int_id")
        .with_columns(pl.col("article_int_id").cast(pl.Int32))
    )

    mapping_dict = {
        "customers": dict(
            zip(
                cust_map["customer_id"].to_list(), cust_map["customer_int_id"].to_list()
            )
        ),
        "articles": dict(
            zip(art_map["article_id"].to_list(), art_map["article_int_id"].to_list())
        ),
    }

    with open(ID_MAPPING_PATH, "w") as f:
        json.dump(mapping_dict, f)

    return cust_map.lazy(), art_map.lazy(), mapping_dict


def process_customers(
    cust: pl.LazyFrame, tx: pl.LazyFrame, art: pl.LazyFrame, cust_map: pl.LazyFrame
) -> pl.LazyFrame:
    """Imputes missing values and applies ID mappings."""
    console.print("[cyan]Processing Customers...[/cyan]")

    cust = cust.with_columns(
        pl.when(pl.col("postal_code") == SUPER_CODE)
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("postal_code"))
        .alias("postal_code")
    ).with_columns(
        [
            pl.col("FN").fill_null(0.0).cast(pl.Int8),
            pl.col("Active").fill_null(0.0).cast(pl.Int8),
        ]
    )

    tx_art = tx.join(
        art.select(["article_id", "index_name"]), on="article_id", how="left"
    )
    top_indices = (
        tx_art.group_by(["customer_id", "index_name"])
        .len()
        .sort(["customer_id", "len"], descending=[False, True])
        .group_by("customer_id")
        .first()
        .select(["customer_id", "index_name"])
    )

    age_map_df = pl.DataFrame(
        {
            "index_name": list(AGE_MAPPING.keys()),
            "imputed_age": list(AGE_MAPPING.values()),
        }
    ).lazy()

    cust = (
        cust.join(top_indices, on="customer_id", how="left")
        .join(age_map_df, on="index_name", how="left")
        .with_columns(
            pl.col("age").fill_null(pl.col("imputed_age")).fill_null(33).cast(pl.Int8)
        )
        .drop(["index_name", "imputed_age"])
    )

    cust = (
        cust.join(cust_map, on="customer_id", how="inner")
        .drop("customer_id")
        .rename({"customer_int_id": "customer_id"})
    )
    return cust


def process_articles(art: pl.LazyFrame, art_map: pl.LazyFrame) -> pl.LazyFrame:
    """Applies ID mappings to articles."""
    console.print("[cyan]Processing Articles...[/cyan]")
    return (
        art.join(art_map, on="article_id", how="inner")
        .drop("article_id")
        .rename({"article_int_id": "article_id"})
    )


def process_transactions(
    tx: pl.LazyFrame, cust_map: pl.LazyFrame, art_map: pl.LazyFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Cleans, deduplicates, purges resellers, calculates time weights, and splits train/test."""
    console.print("[cyan]Processing Transactions...[/cyan]")

    tx = tx.with_columns(pl.col("t_dat").str.to_date())
    tx = (
        tx.group_by(["t_dat", "customer_id", "article_id", "price", "sales_channel_id"])
        .len()
        .rename({"len": "quantity"})
    )

    cust_totals = tx.group_by("customer_id").agg(
        pl.col("quantity").sum().alias("total_items")
    )
    valid_custs = cust_totals.filter(pl.col("total_items") <= RESELLER_CUTOFF).select(
        "customer_id"
    )
    tx = tx.join(valid_custs, on="customer_id", how="inner")

    # Time Decay Feature using window function
    tx = tx.with_columns(
        [(pl.col("t_dat").max() - pl.col("t_dat")).dt.total_days().alias("days_ago")]
    ).with_columns(
        [(-DECAY_HALF_LIFE_RATE * pl.col("days_ago")).exp().alias("time_weight")]
    )


    tx = (
        tx.join(cust_map, on="customer_id", how="inner")
        .join(art_map, on="article_id", how="inner")
        .drop(["customer_id", "article_id"])
        .rename({"customer_int_id": "customer_id", "article_int_id": "article_id"})
    )

    # Train/Test Split (Window function requires max over full column)
    split_date = pl.col("t_dat").max() - pl.duration(days=TEST_WINDOW_DAYS)
    train_tx = tx.filter(pl.col("t_dat") < split_date)
    test_tx = tx.filter(pl.col("t_dat") >= split_date)

    return train_tx, test_tx


def main():
    console.print(
        "[bold magenta]Starting Data Preprocessing Pipeline (Lazy Evaluation)[/bold magenta]"
    )

    cust_lazy = pl.scan_parquet(PROCESSED_DIR / "customers.parquet")
    art_lazy = pl.scan_parquet(PROCESSED_DIR / "articles.parquet")
    tx_lazy = pl.scan_parquet(PROCESSED_DIR / "transactions_train.parquet")

    cust_map, art_map, _ = create_id_mappings(cust_lazy, art_lazy)

    cust_processed = process_customers(cust_lazy, tx_lazy, art_lazy, cust_map)
    art_processed = process_articles(art_lazy, art_map)
    train_tx, test_tx = process_transactions(tx_lazy, cust_map, art_map)

    console.print("[cyan]Executing pipeline and exporting datasets...[/cyan]")
    # Execute lazy graphs
    train_df = train_tx.collect()
    test_df = test_tx.collect()

    train_df.write_parquet(OUTPUT_DIR / "train_processed.parquet", compression="zstd")
    test_df.write_parquet(OUTPUT_DIR / "test_processed.parquet", compression="zstd")
    cust_processed.collect().write_parquet(
        OUTPUT_DIR / "customers_processed.parquet", compression="zstd"
    )
    art_processed.collect().write_parquet(
        OUTPUT_DIR / "articles_processed.parquet", compression="zstd"
    )

    console.print(
        f"[bold green]✓ Pipeline Complete. Train rows: {len(train_df)}, Test rows: {len(test_df)}[/bold green]"
    )


if __name__ == "__main__":
    main()
