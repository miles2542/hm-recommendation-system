import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runRefinement() -> None:
    processedDir = Path("data/processed")
    figDir = Path("figures/phase6")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load transactions and convert t_dat to date
    tx = pl.read_parquet(processedDir / "transactions_train.parquet").with_columns(
        pl.col("t_dat").str.to_date()
    )
    cust = pl.read_parquet(processedDir / "customers.parquet")
    art = pl.read_parquet(processedDir / "articles.parquet")

    # 1. Article Shelf Life
    print("Analyzing Article Shelf Life...")
    shelfLife = (
        tx.group_by("article_id")
        .agg(
            [
                pl.col("t_dat").min().alias("first_sold"),
                pl.col("t_dat").max().alias("last_sold"),
            ]
        )
        .with_columns(
            (pl.col("last_sold") - pl.col("first_sold"))
            .dt.total_days()
            .alias("shelf_life_days")
        )
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(shelfLife["shelf_life_days"], bins=50, color="darkorange")
    plt.title("Distribution of Article Shelf Life (Days)")
    plt.savefig(figDir / "article_shelf_life.png")
    plt.close()

    # 2. Postal Code Concentration
    print("Analyzing Postal Code Concentration...")
    postalSales = tx.join(
        cust.select(["customer_id", "postal_code"]), on="customer_id", how="left"
    )
    topPostal = (
        postalSales.group_by("postal_code").len().sort("len", descending=True).head(10)
    )
    print("Top 10 Postal Codes by Sales Volume:")
    print(topPostal)

    # 3. New Entity Inflow (Monthly)
    print("Analyzing New Entity Inflow...")
    # First purchase per customer
    custInflow = tx.group_by("customer_id").agg(
        pl.col("t_dat").min().alias("join_date")
    )
    monthlyCusts = (
        custInflow.with_columns(pl.col("join_date").dt.truncate("1mo"))
        .group_by("join_date")
        .len()
        .sort("join_date")
    )

    # First sale per article
    artInflow = tx.group_by("article_id").agg(pl.col("t_dat").min().alias("intro_date"))
    monthlyArts = (
        artInflow.with_columns(pl.col("intro_date").dt.truncate("1mo"))
        .group_by("intro_date")
        .len()
        .sort("intro_date")
    )

    plt.figure(figsize=(15, 6))
    plt.plot(
        monthlyCusts["join_date"],
        monthlyCusts["len"],
        label="New Customers",
        marker="o",
    )
    plt.plot(
        monthlyArts["intro_date"], monthlyArts["len"], label="New Articles", marker="x"
    )
    plt.title("Entity Inflow (Monthly)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(figDir / "entity_inflow.png")
    plt.close()

    # 4. Dead Stock (Unsold Articles)
    totalArts = art["article_id"].unique()
    soldArts = tx["article_id"].unique()
    unsold = totalArts.filter(~totalArts.is_in(soldArts))
    print(f"Total Articles: {len(totalArts)}")
    print(
        f"Articles Never Sold: {len(unsold)} ({len(unsold) / len(totalArts) * 100:.2f}%)"
    )


if __name__ == "__main__":
    runRefinement()
