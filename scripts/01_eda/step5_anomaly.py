import polars as pl
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runAnomaly() -> None:
    processedDir = Path("data/processed")
    tx = pl.read_parquet(processedDir / "transactions_train.parquet")

    # 1. High Volume Customers (Potential Resellers)
    print("\n--- Top Customers by Purchase Count ---")
    topCusts = tx.group_by("customer_id").len().sort("len", descending=True).head(10)
    print(topCusts)

    # 2. Extreme Price Outliers
    print("\n--- Extreme Price Items ---")
    meanPrice = tx["price"].mean()
    stdPrice = tx["price"].std()
    threshold = meanPrice + 10 * stdPrice
    outliers = (
        tx.filter(pl.col("price") > threshold)
        .select(["article_id", "price"])
        .unique()
        .sort("price", descending=True)
        .head(10)
    )
    print(outliers)

    # 3. Article Price Volatility
    print("\n--- Article Price Range (Max/Min ratio) ---")
    priceVol = (
        tx.group_by("article_id")
        .agg(
            [pl.col("price").min().alias("min_p"), pl.col("price").max().alias("max_p")]
        )
        .with_columns((pl.col("max_p") / pl.col("min_p")).alias("ratio"))
        .filter(pl.col("min_p") > 0)
        .sort("ratio", descending=True)
        .head(10)
    )
    print(priceVol)


if __name__ == "__main__":
    runAnomaly()
