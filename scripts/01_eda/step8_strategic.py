import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

pl.Config.set_ascii_tables(True)


def runPhase8() -> None:
    processedDir = Path("data/processed")
    figDir = Path("figures/phase8")
    figDir.mkdir(parents=True, exist_ok=True)

    tables = ["articles", "customers", "transactions_train"]
    data = {t: pl.read_parquet(processedDir / f"{t}.parquet") for t in tables}

    # 1. File Metadata (Basic Info)
    print("\n--- File Metadata ---")
    for name, df in data.items():
        print(f"Table: {name}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns}")
        print(f"  Size: {df.estimated_size('mb'):.2f} MB")

    tx = data["transactions_train"].with_columns(pl.col("t_dat").str.to_date())
    txWithArt = tx.join(
        data["articles"].select(["article_id", "index_name"]), on="article_id"
    )

    # 2. Cohort Analysis (Retention)
    print("\nAnalyzing Cohorts...")
    # Get join month per customer
    custJoin = (
        tx.group_by("customer_id")
        .agg(pl.col("t_dat").min().alias("join_month"))
        .with_columns(pl.col("join_month").dt.truncate("1mo"))
    )

    # Merge back and calculate "months since join"
    txCohort = tx.join(custJoin, on="customer_id")
    txCohort = txCohort.with_columns(
        [pl.col("t_dat").dt.truncate("1mo").alias("active_month")]
    ).with_columns(
        [
            (
                (pl.col("active_month").dt.year() - pl.col("join_month").dt.year()) * 12
                + (pl.col("active_month").dt.month() - pl.col("join_month").dt.month())
            ).alias("month_diff")
        ]
    )

    # Aggregate
    cohortCounts = (
        txCohort.group_by(["join_month", "month_diff"])
        .agg(pl.col("customer_id").n_unique().alias("unique_customers"))
        .sort(["join_month", "month_diff"])
    )

    # Pivot for heatmap
    cohortPivot = cohortCounts.to_pandas().pivot(
        index="join_month", columns="month_diff", values="unique_customers"
    )
    # Normalize by cohort size (month_diff = 0)
    cohortSize = cohortPivot[0]
    retention = cohortPivot.divide(cohortSize, axis=0)

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        retention,
        annot=True,
        fmt=".0%",
        cmap="YlGnBu",
        cbar_kws={"label": "Retention Rate"},
    )
    plt.title("Customer Cohort Retention Heatmap")
    plt.xlabel("Months Since First Purchase")
    plt.ylabel("Join Month")
    plt.savefig(figDir / "cohort_retention.png")
    plt.close()

    # 3. Category Transition Matrix
    print("\nAnalyzing Category Transitions...")
    # Get unique (customer, index) pairs
    userIndexPairs = txWithArt.select(["customer_id", "index_name"]).unique()

    # Self-join to find pairs bought by same user
    cooc = (
        userIndexPairs.join(userIndexPairs, on="customer_id")
        .group_by(["index_name", "index_name_right"])
        .len()
    )

    # Pivot to matrix
    matrixPl = cooc.pivot(
        on="index_name_right", index="index_name", values="len"
    ).fill_null(0)
    matrix = matrixPl.to_pandas().set_index("index_name")

    # Normalize by diagonal (Total customers who bought index X)
    diag = np.diag(matrix)
    matrixNorm = matrix.div(diag, axis=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrixNorm, annot=True, fmt=".2f", cmap="Purples")
    plt.title("Index Affinity: 'Customers who bought X also bought Y'")
    plt.tight_layout()
    plt.savefig(figDir / "category_affinity.png")
    plt.close()


if __name__ == "__main__":
    runPhase8()
