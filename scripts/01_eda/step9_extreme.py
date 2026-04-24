import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runExtreme() -> None:
    processedDir = Path("data/processed")
    figDir = Path("figures/phase9")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load data
    tx = pl.read_parquet(processedDir / "transactions_train.parquet").with_columns(
        pl.col("t_dat").str.to_date()
    )
    art = pl.read_parquet(processedDir / "articles.parquet").select(
        ["article_id", "index_name"]
    )
    cust = pl.read_parquet(processedDir / "customers.parquet").select(
        ["customer_id", "postal_code"]
    )

    # 1. Sequential Analysis (Next Purchase)
    print("Analyzing Sequential Paths...")
    # Join with index names and sort
    seq = tx.join(art, on="article_id").sort(["customer_id", "t_dat"])
    # Get the NEXT category bought by the same user
    seq = seq.with_columns(
        pl.col("index_name").shift(-1).over("customer_id").alias("next_index")
    ).filter(pl.col("next_index").is_not_null())

    # Transition Matrix
    trans = seq.group_by(["index_name", "next_index"]).len()
    transPivot = trans.pivot(
        on="next_index", index="index_name", values="len"
    ).fill_null(0)
    transMatrix = transPivot.to_pandas().set_index("index_name")
    transNorm = transMatrix.div(transMatrix.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(transNorm, annot=True, fmt=".2f", cmap="YlGn")
    plt.title("Sequential Index Transitions (What they buy next)")
    plt.tight_layout()
    plt.savefig(figDir / "sequential_transitions.png")
    plt.close()

    # 2. Spend Velocity (Monetary per month of tenure)
    print("Analyzing Spend Velocity...")
    custStats = (
        tx.group_by("customer_id")
        .agg(
            [
                pl.col("t_dat").min().alias("start"),
                pl.col("t_dat").max().alias("end"),
                pl.col("price").sum().alias("total_monetary"),
            ]
        )
        .with_columns(
            ((pl.col("end") - pl.col("start")).dt.total_days() / 30)
            .clip(lower_bound=1)
            .alias("tenure_months")
        )
        .with_columns(
            (pl.col("total_monetary") / pl.col("tenure_months")).alias("velocity")
        )
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(
        custStats.filter(pl.col("velocity") < 0.5)["velocity"], bins=50, color="indigo"
    )
    plt.title("Customer Spend Velocity (Monetary/Month)")
    plt.savefig(figDir / "spend_velocity.png")
    plt.close()

    # 3. Price Elasticity Proxy (Top 1000 Articles)
    print("Analyzing Price Elasticity...")
    topArts = (
        tx.group_by("article_id")
        .len()
        .sort("len", descending=True)
        .head(1000)["article_id"]
    )
    elasticityData = (
        tx.filter(pl.col("article_id").is_in(topArts))
        .with_columns(pl.col("t_dat").dt.truncate("1w").alias("week"))
        .group_by(["article_id", "week"])
        .agg(
            [
                pl.col("price").mean().alias("avg_p"),
                pl.col("article_id").len().alias("vol"),
            ]
        )
    )

    # Calculate correlation per article
    # We'll just look at the distribution of correlations
    corrs = (
        elasticityData.group_by("article_id")
        .agg(pl.corr("avg_p", "vol").alias("elasticity"))
        .drop_nulls()
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(corrs["elasticity"], bins=30, color="crimson", kde=True)
    plt.title("Distribution of Price Elasticity (Corrs) for Top 1000 SKUs")
    plt.axvline(0, color="black", linestyle="--")
    plt.savefig(figDir / "price_elasticity_dist.png")
    plt.close()

    # 4. Geographical Gini (Sales across Postal Codes)
    print("Analyzing Geographical Gini...")
    geoSales = (
        tx.join(cust, on="customer_id")
        .group_by("postal_code")
        .len()
        .sort("len", descending=True)
    )
    cumGeoSales = geoSales["len"].cum_sum().to_numpy() / geoSales["len"].sum()
    cumCodes = np.linspace(0, 1, len(geoSales))

    plt.figure(figsize=(8, 8))
    plt.plot(cumCodes, cumGeoSales, color="teal", label="Geo Concentration")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Geographical Concentration (Sales per Postal Code)")
    plt.savefig(figDir / "geo_lorenz.png")
    plt.close()


if __name__ == "__main__":
    runExtreme()
