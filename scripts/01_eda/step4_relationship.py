import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runRelationship() -> None:
    processedDir = Path("data/processed")
    figDir = Path("figures/phase4")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load data
    cust = pl.read_parquet(processedDir / "customers.parquet")
    art = pl.read_parquet(processedDir / "articles.parquet")
    tx = pl.read_parquet(processedDir / "transactions_train.parquet")

    # 1. Average Spend by Age Group
    print("Analyzing Spend by Age...")
    # Bin ages: 10s, 20s, 30s, 40s, 50s, 60s, 70s+
    custAge = cust.select(["customer_id", "age"]).with_columns(
        pl.when(pl.col("age") < 20)
        .then(pl.lit("10s"))
        .when(pl.col("age") < 30)
        .then(pl.lit("20s"))
        .when(pl.col("age") < 40)
        .then(pl.lit("30s"))
        .when(pl.col("age") < 50)
        .then(pl.lit("40s"))
        .when(pl.col("age") < 60)
        .then(pl.lit("50s"))
        .when(pl.col("age") < 70)
        .then(pl.lit("60s"))
        .otherwise(pl.lit("70+"))
        .alias("age_group")
    )

    # Join with transactions
    txWithAge = tx.join(custAge, on="customer_id", how="left")
    ageSpend = (
        txWithAge.group_by("age_group")
        .agg(pl.col("price").sum().alias("total_spend"))
        .sort("age_group")
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=ageSpend["age_group"], y=ageSpend["total_spend"], palette="magma")
    plt.title("Total Spend by Age Group")
    plt.savefig(figDir / "spend_by_age.png")
    plt.close()

    # 2. Price Distribution per Main Index
    print("Analyzing Price by Index...")
    txWithArt = tx.join(
        art.select(["article_id", "index_name"]), on="article_id", how="left"
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=txWithArt["price"], y=txWithArt["index_name"], palette="Set2")
    plt.title("Price Distribution by Product Index")
    plt.xscale("log")
    plt.savefig(figDir / "price_by_index_box.png")
    plt.close()

    # 3. Age Group vs Index Preference (Heatmap)
    print("Analyzing Demographic Preferences...")
    pref = txWithAge.join(
        art.select(["article_id", "index_name"]), on="article_id", how="left"
    )
    prefMatrix = (
        pref.group_by(["age_group", "index_name"])
        .len()
        .pivot(on="index_name", index="age_group", values="len")
        .fill_null(0)
    )

    # Convert to pandas for easier heatmap plotting
    prefPandas = prefMatrix.to_pandas().set_index("age_group")
    # Normalize by row (age group) to see relative preference
    prefPandasNorm = prefPandas.div(prefPandas.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(prefPandasNorm, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Product Index Preference by Age Group (Normalized)")
    plt.tight_layout()
    plt.savefig(figDir / "age_category_heatmap.png")
    plt.close()


if __name__ == "__main__":
    runRelationship()
