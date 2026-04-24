import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console

pl.Config.set_ascii_tables(True)


def runUnivariate() -> None:
    console = Console()
    processedDir = Path("data/processed")
    figDir = Path("figures/phase2")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load data
    cust = pl.read_parquet(processedDir / "customers.parquet")
    art = pl.read_parquet(processedDir / "articles.parquet")
    tx = pl.read_parquet(processedDir / "transactions_train.parquet")

    # 1. Customer Age Distribution
    console.print("[cyan]Analyzing Customer Age...[/cyan]")
    plt.figure(figsize=(10, 6))
    sns.histplot(cust["age"].drop_nulls(), bins=50, kde=True, color="teal")
    plt.title("Distribution of Customer Age")
    plt.savefig(figDir / "customer_age_dist.png")
    plt.close()

    # 2. Transaction Price Distribution
    console.print("[cyan]Analyzing Transaction Price...[/cyan]")
    plt.figure(figsize=(10, 6))
    # Prices are usually small decimals in this dataset, log scale helps
    sns.histplot(tx["price"], bins=100, kde=False, color="purple")
    plt.yscale("log")
    plt.title("Distribution of Prices (Log Scale)")
    plt.savefig(figDir / "price_dist_log.png")
    plt.close()

    # 3. Top Article Categories
    console.print("[cyan]Analyzing Article Categories...[/cyan]")
    # index_name is a good high-level category
    topIndices = (
        art["index_name"].value_counts().sort("count", descending=True).head(10)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(x=topIndices["count"], y=topIndices["index_name"], palette="viridis")
    plt.title("Top 10 Article Indices")
    plt.tight_layout()
    plt.savefig(figDir / "top_indices.png")
    plt.close()

    # 4. Sales Channel
    console.print("[cyan]Analyzing Sales Channels...[/cyan]")
    channelCounts = tx["sales_channel_id"].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(
        channelCounts["count"],
        labels=[f"Channel {i}" for i in channelCounts["sales_channel_id"]],
        autopct="%1.1f%%",
        colors=["skyblue", "salmon"],
    )
    plt.title("Sales Channel Distribution")
    plt.savefig(figDir / "sales_channel_pie.png")
    plt.close()

    # 5. Investigate Duplicates in Transactions
    console.print("[cyan]Investigating Duplicates...[/cyan]")
    dupeSample = tx.filter(tx.is_duplicated()).head(5)
    print("Sample of duplicated rows:")
    print(dupeSample)


if __name__ == "__main__":
    runUnivariate()
