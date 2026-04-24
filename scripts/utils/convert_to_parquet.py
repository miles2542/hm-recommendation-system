import polars as pl
from pathlib import Path
from rich.console import Console


def convertToParquet() -> None:
    console = Console()
    rawDir = Path("data/raw")
    processedDir = Path("data/processed")
    processedDir.mkdir(parents=True, exist_ok=True)

    csvFiles = list(rawDir.glob("*.csv"))

    if not csvFiles:
        console.print("[bold red]No CSV files found in data/raw[/bold red]")
        return

    for csvFile in csvFiles:
        targetFile = processedDir / f"{csvFile.stem}.parquet"
        console.print(f"[cyan]Converting {csvFile.name}...[/cyan]")

        try:
            # Use sink_parquet for streaming to avoid OOM on large files
            pl.scan_csv(csvFile).sink_parquet(targetFile, compression="zstd")
            console.print(f"[bold green][DONE] Finished {csvFile.name}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to convert {csvFile.name}: {e}[/bold red]")


if __name__ == "__main__":
    convertToParquet()
