import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pl.Config.set_ascii_tables(True)


def verifyAgeAffinity():
    processedDir = Path("data/processed")
    figDir = Path("figures/verification")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load data
    tx = pl.read_parquet(processedDir / "transactions_train.parquet")
    art = pl.read_parquet(processedDir / "articles.parquet").select(
        ["article_id", "index_name"]
    )
    cust = pl.read_parquet(processedDir / "customers.parquet").select(
        ["customer_id", "age"]
    )

    # Merge
    # We only care about transactions where age is known
    print("Merging data for age verification...")
    df = tx.join(art, on="article_id").join(cust, on="customer_id").drop_nulls("age")

    # Calculate Stats
    print("\n--- Age Stats per Product Index ---")
    stats = (
        df.group_by("index_name")
        .agg(
            [
                pl.col("age").mean().alias("mean_age"),
                pl.col("age").median().alias("median_age"),
                pl.col("age").std().alias("std_age"),
                pl.len().alias("transaction_count"),
            ]
        )
        .sort("median_age")
    )

    print(stats)

    # Visualization
    plt.figure(figsize=(12, 8))
    # Sample a portion for the boxplot to keep it fast/clean if needed,
    # but with 32GB ram and 30M rows, seaborn might be slow.
    # Let's use a sample of 1M rows for the plot.
    sns.boxplot(
        data=df.sample(n=1_000_000).to_pandas(), x="age", y="index_name", palette="husl"
    )
    plt.title("Age Distribution per Product Index (1M Transaction Sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figDir / "age_affinity_verification.png")
    plt.close()

    print(f"\nPlot saved to {figDir / 'age_affinity_verification.png'}")


if __name__ == "__main__":
    verifyAgeAffinity()
