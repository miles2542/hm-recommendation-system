import polars as pl
from rich.console import Console
from rich.table import Table
from pathlib import Path


def analyzeSchema(filePath: Path) -> None:
    console = Console()
    console.print(f"\n[bold blue]Analyzing: {filePath.name}[/bold blue]")

    # Read head to infer types and see sample
    try:
        # Use scan_csv for large files to be safe/efficient
        df = pl.scan_csv(filePath).head(5).collect()

        schemaTable = Table(
            title=f"Schema: {filePath.name}", show_header=True, header_style="bold blue"
        )
        schemaTable.add_column("Field", style="cyan")
        schemaTable.add_column("Type", style="green")
        schemaTable.add_column("Sample", style="white")

        for col in df.columns:
            schemaTable.add_row(col, str(df.schema[col]), str(df[col][0]))

        console.print(schemaTable)

    except Exception as e:
        console.print(f"[bold red]Error analyzing {filePath.name}: {e}[/bold red]")


def main() -> None:
    rawDataDir = Path("data/raw")
    csvFiles = list(rawDataDir.glob("*.csv"))

    if not csvFiles:
        print("No CSV files found in data/raw")
        return

    for csvFile in csvFiles:
        analyzeSchema(csvFile)


if __name__ == "__main__":
    main()
