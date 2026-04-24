import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rich.console import Console
from rich.table import Table


def checkIntegrity() -> None:
    console = Console()
    processedDir = Path("data/processed")
    figDir = Path("figures/phase1")
    figDir.mkdir(parents=True, exist_ok=True)

    tables = ["articles", "customers", "transactions_train"]
    data = {t: pl.read_parquet(processedDir / f"{t}.parquet") for t in tables}

    # 1. Null Analysis
    console.print("\n[bold blue]--- Null Analysis ---[/bold blue]")
    for name, df in data.items():
        nullCounts = df.null_count()
        totalRows = len(df)

        nullTable = Table(title=f"Nulls in {name}", header_style="bold blue")
        nullTable.add_column("Column", style="cyan")
        nullTable.add_column("Null Count", style="magenta")
        nullTable.add_column("Null %", style="yellow")

        hasNulls = False
        nullData = []
        for col in df.columns:
            count = nullCounts[col][0]
            if count > 0:
                hasNulls = True
                pct = (count / totalRows) * 100
                nullTable.add_row(col, str(count), f"{pct:.2f}%")
                nullData.append({"col": col, "pct": pct})

        if hasNulls:
            console.print(nullTable)
            # Plot nulls
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=[d["col"] for d in nullData],
                y=[d["pct"] for d in nullData],
                color="royalblue",
            )
            plt.title(f"Null Percentage in {name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(figDir / f"{name}_nulls.png")
            plt.close()
        else:
            console.print(f"[green]No nulls found in {name}[/green]")

    # 2. Cardinality Analysis
    console.print("\n[bold blue]--- Cardinality Analysis ---[/bold blue]")
    cardTable = Table(title="Uniqueness Profiling", header_style="bold blue")
    cardTable.add_column("Table", style="cyan")
    cardTable.add_column("Column", style="magenta")
    cardTable.add_column("Unique Count", style="green")
    cardTable.add_column("Total Rows", style="white")

    for name, df in data.items():
        total = len(df)
        for col in df.columns:
            # Skip high-memory columns for unique count if too large? No, Polars is fast.
            # But let's focus on IDs and Categoricals
            uniqueCount = df[col].n_unique()
            if uniqueCount < total or "id" in col.lower():
                cardTable.add_row(name, col, str(uniqueCount), str(total))
    console.print(cardTable)

    # 3. Referential Integrity
    console.print("\n[bold blue]--- Referential Integrity ---[/bold blue]")
    tx = data["transactions_train"]
    cust = data["customers"]
    art = data["articles"]

    # Missing customers in transactions
    txCusts = tx["customer_id"].unique()
    missingCusts = txCusts.filter(~txCusts.is_in(cust["customer_id"]))
    console.print(
        f"Transactions with unknown customer_id: [red]{len(missingCusts)}[/red]"
    )

    # Missing articles in transactions
    txArts = tx["article_id"].unique()
    missingArts = txArts.filter(~txArts.is_in(art["article_id"]))
    console.print(
        f"Transactions with unknown article_id: [red]{len(missingArts)}[/red]"
    )

    # Orphan customers (no purchases)
    orphanCusts = cust.filter(~cust["customer_id"].is_in(tx["customer_id"]))
    console.print(
        f"Customers with 0 purchases (Orphans): [yellow]{len(orphanCusts)}[/yellow] ({len(orphanCusts) / len(cust) * 100:.2f}%)"
    )

    # 4. Duplicates
    console.print("\n[bold blue]--- Duplicates Analysis ---[/bold blue]")
    for name, df in data.items():
        dupes = df.is_duplicated().sum()
        console.print(
            f"Duplicate rows in {name}: [red]{dupes}[/red] ({dupes / len(df) * 100:.4f}%)"
        )


if __name__ == "__main__":
    checkIntegrity()
