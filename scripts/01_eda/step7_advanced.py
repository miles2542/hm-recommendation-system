import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runAdvanced() -> None:
    processedDir = Path("data/processed")
    figDir = Path("figures/phase7")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load data
    tx = pl.read_parquet(processedDir / "transactions_train.parquet").with_columns(
        pl.col("t_dat").str.to_date()
    )

    # 1. RFM Calculation
    print("Calculating RFM...")
    lastDate = tx["t_dat"].max()

    rfm = tx.group_by("customer_id").agg(
        [
            (lastDate - pl.col("t_dat").max()).dt.total_days().alias("recency"),
            pl.col("article_id").len().alias("frequency"),
            pl.col("price").sum().alias("monetary"),
        ]
    )

    # Plot RFM Distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(rfm["recency"], bins=50, ax=axes[0], color="blue").set_title(
        "Recency (Days)"
    )
    sns.histplot(rfm["frequency"], bins=50, ax=axes[1], color="green").set_title(
        "Frequency (Items)"
    )
    axes[1].set_yscale("log")
    sns.histplot(rfm["monetary"], bins=50, ax=axes[2], color="red").set_title(
        "Monetary (Spend)"
    )
    axes[2].set_yscale("log")
    plt.savefig(figDir / "rfm_distributions.png")
    plt.close()

    # 2. RecSys Sparsity & Popularity Bias
    print("Analyzing RecSys Readiness...")
    numUsers = tx["customer_id"].n_unique()
    numItems = tx["article_id"].n_unique()
    numInteractions = len(tx)
    sparsity = 1 - (numInteractions / (numUsers * numItems))
    print(f"Sparsity: {sparsity * 100:.6f}%")

    # Lorenz Curve for Popularity Bias
    itemSales = tx.group_by("article_id").len().sort("len", descending=True)
    cumSales = itemSales["len"].cum_sum().to_numpy() / itemSales["len"].sum()
    cumItems = np.linspace(0, 1, len(itemSales))

    plt.figure(figsize=(8, 8))
    plt.plot(cumItems, cumSales, label="Sales Concentration", color="purple")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Equality")
    plt.xlabel("Fraction of Items")
    plt.ylabel("Fraction of Total Sales")
    plt.title("Lorenz Curve of Sales (Popularity Bias)")
    plt.legend()
    plt.savefig(figDir / "popularity_lorenz.png")
    plt.close()

    # 3. Churn Analysis (Proxy)
    print("Analyzing Churn Proxy...")
    # Churn defined as no purchase in the last 6 months (180 days)
    atRisk = len(rfm.filter(pl.col("recency") > 180))
    churned = len(rfm.filter(pl.col("recency") > 365))
    print(f"At-Risk Customers (>180 days): {atRisk} ({atRisk / len(rfm) * 100:.2f}%)")
    print(f"Likely Churned (>365 days): {churned} ({churned / len(rfm) * 100:.2f}%)")

    # 4. Basket Analysis (Items per Purchase Event)
    print("Analyzing Basket Size...")
    basket = tx.group_by(["customer_id", "t_dat"]).len()
    print("Average items per purchase event:")
    print(basket["len"].mean())

    plt.figure(figsize=(10, 6))
    sns.countplot(x=basket.filter(pl.col("len") < 20)["len"], color="salmon")
    plt.title("Items per Transaction Event")
    plt.savefig(figDir / "basket_size_dist.png")
    plt.close()


if __name__ == "__main__":
    runAdvanced()
