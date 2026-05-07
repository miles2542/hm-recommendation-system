import polars as pl
import json
from datetime import timedelta
from pathlib import Path
from rich.console import Console

# --- Configuration & Constants ---
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")
ID_MAPPING_PATH = OUTPUT_DIR / "id_mapping.json"

SUPER_CODE = "2c29ae653a9282cce4151bd87643c911a68d276429a91f8ca0d4b6efa8100"
RESELLER_CUTOFF = 1000
TEST_WINDOW_DAYS = 28

# RESTORED FIX #3: 0.0231 correctly yields a 30-day half-life.
DECAY_HALF_LIFE_RATE = 0.0231 

AGE_MAPPING = {
    "Lingeries/Tights": 29, "Divided": 29, "Ladies Accessories": 30,
    "Sport": 31, "Baby Sizes 50-98": 33, "Ladieswear": 33,
    "Menswear": 34, "Children Sizes 92-140": 38, "Children Accessories": 39,
    "Children Sizes 134-170": 45,
}

pl.Config.set_ascii_tables(True)
console = Console()

def create_id_mappings(cust: pl.LazyFrame, art: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame, dict]:
    """Generates contiguous Int32 IDs for memory efficiency. Evaluates eagerly."""
    console.print("[cyan]Creating ID Mappings...[/cyan]")
    
    # Evaluate to get mappings
    # (Note: Using sort here to ensure determinism as suggested by feedback #4, 
    # even though we are focusing on #1-#3, it's good practice for stability)
    cust_map = cust.select("customer_id").unique().collect().sort("customer_id").with_row_index("customer_int_id").with_columns(pl.col("customer_int_id").cast(pl.Int32))
    art_map = art.select("article_id").unique().collect().sort("article_id").with_row_index("article_int_id").with_columns(pl.col("article_int_id").cast(pl.Int32))
    
    mapping_dict = {
        "customers": dict(zip(cust_map["customer_id"].to_list(), cust_map["customer_int_id"].to_list())),
        "articles": dict(zip(art_map["article_id"].to_list(), art_map["article_int_id"].to_list()))
    }
    
    with open(ID_MAPPING_PATH, "w") as f:
        json.dump(mapping_dict, f)
        
    return cust_map.lazy(), art_map.lazy(), mapping_dict

def process_customers(cust: pl.LazyFrame, train_tx_unmapped: pl.LazyFrame, art: pl.LazyFrame, cust_map: pl.LazyFrame) -> pl.LazyFrame:
    """Imputes missing values and applies ID mappings. FIX #1: Uses only training data for affinity."""
    console.print("[cyan]Processing Customers (Leakage-Proofed)...[/cyan]")
    
    cust = cust.with_columns(
        pl.when(pl.col("postal_code") == SUPER_CODE).then(pl.lit("Unknown")).otherwise(pl.col("postal_code")).alias("postal_code")
    ).with_columns([
        pl.col("FN").fill_null(0.0).cast(pl.Int8),
        pl.col("Active").fill_null(0.0).cast(pl.Int8)
    ])
    
    # FIX #1 & #5: Find top index per customer using ONLY train interactions and correct aggregation pattern
    tx_art = train_tx_unmapped.join(art.select(["article_id", "index_name"]), on="article_id", how="left")
    
    top_indices = (
        tx_art.group_by(["customer_id", "index_name"])
        .agg(pl.len().alias("cnt"))
        .group_by("customer_id")
        .agg(pl.col("index_name").sort_by("cnt", descending=True).first())
        .select(["customer_id", "index_name"])
    )
    
    age_map_df = pl.DataFrame({"index_name": list(AGE_MAPPING.keys()), "imputed_age": list(AGE_MAPPING.values())}).lazy()
    
    cust = (
        cust.join(top_indices, on="customer_id", how="left")
        .join(age_map_df, on="index_name", how="left")
        .with_columns(pl.col("age").fill_null(pl.col("imputed_age")).fill_null(33).cast(pl.Int8))
        .drop(["index_name", "imputed_age"])
    )
    
    cust = cust.join(cust_map, on="customer_id", how="inner").drop("customer_id").rename({"customer_int_id": "customer_id"})
    return cust

def process_articles(art: pl.LazyFrame, art_map: pl.LazyFrame) -> pl.LazyFrame:
    """Applies ID mappings to articles."""
    console.print("[cyan]Processing Articles...[/cyan]")
    return art.join(art_map, on="article_id", how="inner").drop("article_id").rename({"article_int_id": "article_id"})

def split_and_process_transactions(tx: pl.LazyFrame, cust_map: pl.LazyFrame, art_map: pl.LazyFrame, split_date) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """FIX #1 & #2: Splits first, then purges resellers and anchors time decay to train max date."""
    console.print("[cyan]Processing Transactions (Leakage-Proofed)...[/cyan]")
    
    tx = tx.with_columns(pl.col("t_dat").str.to_date())
    
    train_tx = tx.filter(pl.col("t_dat") < split_date)
    test_tx = tx.filter(pl.col("t_dat") >= split_date)
    
    # Deduplicate
    train_tx = train_tx.group_by(["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]).len().rename({"len": "quantity"})
    test_tx = test_tx.group_by(["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]).len().rename({"len": "quantity"})
    
    # FIX #1: Reseller Purge calculated strictly on Train Set
    cust_totals = train_tx.group_by("customer_id").agg(pl.col("quantity").sum().alias("total_items"))
    valid_custs = cust_totals.filter(pl.col("total_items") <= RESELLER_CUTOFF).select("customer_id")
    
    train_tx = train_tx.join(valid_custs, on="customer_id", how="inner")
    test_tx = test_tx.join(valid_custs, on="customer_id", how="inner")
    
    # FIX #2: Time Decay anchored to train_max_date
    train_max_date = split_date - timedelta(days=1)
    train_tx = train_tx.with_columns([
        (pl.lit(train_max_date) - pl.col("t_dat")).dt.total_days().alias("days_ago")
    ]).with_columns([
        (-DECAY_HALF_LIFE_RATE * pl.col("days_ago")).exp().alias("time_weight")
    ])
    
    # Test interactions get weight 1.0
    test_tx = test_tx.with_columns([
        pl.lit(0).cast(pl.Int64).alias("days_ago"),
        pl.lit(1.0).cast(pl.Float64).alias("time_weight")
    ])
    
    # Apply Mapping
    train_tx = (
        train_tx.join(cust_map, on="customer_id", how="inner")
        .join(art_map, on="article_id", how="inner")
        .drop(["customer_id", "article_id"])
        .rename({"customer_int_id": "customer_id", "article_int_id": "article_id"})
    )
    test_tx = (
        test_tx.join(cust_map, on="customer_id", how="inner")
        .join(art_map, on="article_id", how="inner")
        .drop(["customer_id", "article_id"])
        .rename({"customer_int_id": "customer_id", "article_int_id": "article_id"})
    )
    
    return train_tx, test_tx

def main():
    console.print("[bold magenta]Starting Data Preprocessing Pipeline (Leakage-Proofed)[/bold magenta]")
    
    cust_lazy = pl.scan_parquet(PROCESSED_DIR / "customers.parquet")
    art_lazy = pl.scan_parquet(PROCESSED_DIR / "articles.parquet")
    tx_lazy = pl.scan_parquet(PROCESSED_DIR / "transactions_train.parquet")
    
    cust_map, art_map, _ = create_id_mappings(cust_lazy, art_lazy)
    
    # Pre-extract dates for splitting
    max_date = tx_lazy.select(pl.col("t_dat").str.to_date().max()).collect().item()
    split_date = max_date - timedelta(days=TEST_WINDOW_DAYS)
    
    # Prepare train set for customer imputation (unmapped)
    train_tx_unmapped = tx_lazy.with_columns(pl.col("t_dat").str.to_date()).filter(pl.col("t_dat") < split_date)
    
    cust_processed = process_customers(cust_lazy, train_tx_unmapped, art_lazy, cust_map)
    art_processed = process_articles(art_lazy, art_map)
    train_tx, test_tx = split_and_process_transactions(tx_lazy, cust_map, art_map, split_date)
    
    console.print("[cyan]Executing pipeline and exporting datasets...[/cyan]")
    train_df = train_tx.collect()
    test_df = test_tx.collect()
    
    train_df.write_parquet(OUTPUT_DIR / "train_processed.parquet", compression="zstd")
    test_df.write_parquet(OUTPUT_DIR / "test_processed.parquet", compression="zstd")
    cust_processed.collect().write_parquet(OUTPUT_DIR / "customers_processed.parquet", compression="zstd")
    art_processed.collect().write_parquet(OUTPUT_DIR / "articles_processed.parquet", compression="zstd")
    
    console.print(f"[bold green]✓ Pipeline Complete. Train rows: {len(train_df)}, Test rows: {len(test_df)}[/bold green]")

if __name__ == "__main__":
    main()
