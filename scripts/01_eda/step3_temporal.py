import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runTemporal() -> None:
    processedDir = Path("data/processed")
    figDir = Path("figures/phase3")
    figDir.mkdir(parents=True, exist_ok=True)

    # Load transactions and convert t_dat to date
    tx = pl.read_parquet(processedDir / "transactions_train.parquet").with_columns(
        pl.col("t_dat").str.to_date()
    )

    # 1. Daily Transaction Volume
    print("Analyzing Daily Volume...")
    daily = tx.group_by("t_dat").count().sort("t_dat")
    plt.figure(figsize=(15, 6))
    plt.plot(daily["t_dat"], daily["count"], color="steelblue", linewidth=1)
    plt.title("Daily Transaction Volume (2018-2020)")
    plt.xlabel("Date")
    plt.ylabel("Number of Transactions")
    plt.grid(True, alpha=0.3)
    plt.savefig(figDir / "daily_volume.png")
    plt.close()

    # 2. Monthly Seasonality
    print("Analyzing Monthly Seasonality...")
    # Group by Month-Year
    monthly = (
        tx.with_columns(pl.col("t_dat").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .count()
        .sort("month")
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=monthly["month"].dt.strftime("%Y-%m"), y=monthly["count"], color="seagreen"
    )
    plt.title("Monthly Transaction Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figDir / "monthly_volume.png")
    plt.close()

    # 3. Day of Week Patterns
    print("Analyzing Day of Week...")
    dow = (
        tx.with_columns(pl.col("t_dat").dt.weekday().alias("dow"))
        .group_by("dow")
        .count()
        .sort("dow")
    )

    dowLabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dowLabels, y=dow["count"], palette="coolwarm")
    plt.title("Transactions by Day of Week")
    plt.savefig(figDir / "dow_distribution.png")
    plt.close()

    # 4. Price Evolution (Mean Daily Price)
    print("Analyzing Price Evolution...")
    dailyPrice = (
        tx.group_by("t_dat")
        .agg(pl.col("price").mean().alias("avg_price"))
        .sort("t_dat")
    )
    plt.figure(figsize=(15, 6))
    plt.plot(dailyPrice["t_dat"], dailyPrice["avg_price"], color="darkred", linewidth=1)
    plt.title("Average Transaction Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean Price")
    plt.grid(True, alpha=0.3)
    plt.savefig(figDir / "price_evolution.png")
    plt.close()


if __name__ == "__main__":
    runTemporal()
