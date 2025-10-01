"""Enhanced command line interface with Click and Rich.

This module provides a modern CLI experience with:
- Click framework for better command structure
- Rich for beautiful terminal output
- Prediction commands
- Analysis commands
- Interactive mode (future)
"""

import json
from pathlib import Path
import sys
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from f1_predict import logging_config
from f1_predict.data.collector import F1DataCollector

# Initialize Rich console
console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="f1-predict")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config file",
    default=None,
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """F1 Race Prediction System - Enhanced CLI.

    A comprehensive command-line interface for F1 race predictions,
    data management, and analysis.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config or "~/.f1predict/config.yaml"

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logging_config.configure_logging(
        log_level=log_level,
        log_format="console",
        enable_colors=True,
    )


# ============================================================================
# DATA MANAGEMENT COMMANDS
# ============================================================================


@cli.group()
def data() -> None:
    """Manage F1 data collection and updates."""
    pass


@data.command()
@click.option(
    "--type",
    "-t",
    "data_type",
    type=click.Choice(["race-results", "qualifying", "schedules", "all"]),
    default="all",
    help="Data types to collect",
)
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data",
    help="Data directory",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force refresh existing data",
)
@click.pass_context
def update(ctx: click.Context, data_type: str, data_dir: str, force: bool) -> None:
    """Update F1 data from external sources.

    \b
    Examples:
        f1-predict data update --type all
        f1-predict data update --type race-results --force
        f1-predict data update --type qualifying --data-dir custom_data
    """
    console.print(f"[bold blue]Updating {data_type} data...[/bold blue]")

    try:
        collector = F1DataCollector(data_dir=data_dir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting data...", total=None)

            if data_type == "all":
                results = collector.collect_all_data()
            elif data_type == "race-results":
                results = {"race_results": collector.collect_race_results()}
            elif data_type == "qualifying":
                results = {"qualifying_results": collector.collect_qualifying_results()}
            elif data_type == "schedules":
                results = {"race_schedules": collector.collect_race_schedules()}
            else:
                console.print(f"[red]Unknown data type: {data_type}[/red]")
                sys.exit(1)

            progress.update(task, completed=True)

        # Display results
        table = Table(title="Data Collection Results", show_header=True)
        table.add_column("Data Type", style="cyan", width=20)
        table.add_column("Status", style="green")
        table.add_column("File Path", style="yellow")

        for data_name, file_path in results.items():
            if file_path:
                table.add_row(data_name, "[green]✓ Success[/green]", str(file_path))
            else:
                table.add_row(data_name, "[red]✗ Failed[/red]", "N/A")

        console.print(table)
        console.print("[bold green]✓[/bold green] Data collection completed")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if ctx.obj["verbose"]:
            console.print_exception()
        sys.exit(1)


@data.command()
@click.option(
    "--type",
    "-t",
    "data_type",
    type=click.Choice(["race-results", "qualifying", "schedules", "all"]),
    default="all",
    help="Data type to inspect",
)
@click.option("--season", "-s", type=int, help="Filter by season")
@click.option("--limit", "-l", type=int, default=10, help="Number of records to show")
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data",
    help="Data directory",
)
def show(data_type: str, season: Optional[int], limit: int, data_dir: str) -> None:
    """Display collected F1 data.

    \b
    Examples:
        f1-predict data show --type race-results --season 2024
        f1-predict data show --type qualifying --limit 20
    """
    try:
        data_path = Path(data_dir) / "raw"

        if data_type == "race-results":
            file_path = data_path / "race_results.json"
        elif data_type == "qualifying":
            file_path = data_path / "qualifying_results.json"
        elif data_type == "schedules":
            file_path = data_path / "race_schedules.json"
        else:
            console.print(f"[red]Unknown data type: {data_type}[/red]")
            sys.exit(1)

        if not file_path.exists():
            console.print(f"[red]Data file not found:[/red] {file_path}")
            console.print(
                "[yellow]Tip:[/yellow] Run [cyan]f1-predict data update[/cyan] first"
            )
            sys.exit(1)

        # Load and display data
        with open(file_path) as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Filter by season if specified
        if season and "season" in df.columns:
            df = df[df["season"] == season]

        # Display summary
        console.print(
            Panel(
                f"[bold]Dataset:[/bold] {data_type}\n"
                f"[bold]Total Records:[/bold] {len(df)}\n"
                + (f"[bold]Season:[/bold] {season}" if season else ""),
                title="Data Summary",
                border_style="blue",
            )
        )

        # Display sample data
        table = Table(title=f"Sample Data (first {limit} records)")

        # Add columns (limit to first 8 for readability)
        columns_to_show = list(df.columns)[:8]
        for col in columns_to_show:
            table.add_column(col, style="cyan", overflow="fold")

        # Add rows
        for _, row in df.head(limit).iterrows():
            table.add_row(*[str(row[col])[:30] for col in columns_to_show])

        console.print(table)

        if len(df.columns) > 8:
            console.print(
                f"\n[dim]Showing {len(columns_to_show)} of {len(df.columns)} columns[/dim]"
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@data.command()
@click.option(
    "--data-dir",
    type=click.Path(),
    default="data",
    help="Data directory",
)
def stats(data_dir: str) -> None:
    """Show data collection statistics and quality metrics.

    \b
    Example:
        f1-predict data stats
    """
    try:
        data_path = Path(data_dir) / "raw"

        datasets = {
            "Race Results": data_path / "race_results.json",
            "Qualifying Results": data_path / "qualifying_results.json",
            "Race Schedules": data_path / "race_schedules.json",
        }

        console.print("\n[bold underline]Data Collection Summary[/bold underline]\n")

        for name, file_path in datasets.items():
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                # Calculate basic stats
                seasons = (
                    sorted(df["season"].unique()) if "season" in df.columns else []
                )
                season_range = f"{min(seasons)}-{max(seasons)}" if seasons else "N/A"

                # Get file modification time
                import datetime

                mod_time = datetime.datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")

                panel = Panel(
                    f"[cyan]Records:[/cyan] {len(df)}\n"
                    f"[cyan]Seasons:[/cyan] {season_range}\n"
                    f"[cyan]Last Updated:[/cyan] {mod_time}\n"
                    f"[cyan]Columns:[/cyan] {len(df.columns)}",
                    title=f"[green]{name}[/green]",
                    border_style="green",
                )
                console.print(panel)
            else:
                panel = Panel(
                    "[red]No data collected[/red]\n"
                    "Run [cyan]f1-predict data update[/cyan] to collect data",
                    title=f"[yellow]{name}[/yellow]",
                    border_style="yellow",
                )
                console.print(panel)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


# ============================================================================
# PREDICTION COMMANDS
# ============================================================================


@cli.command()
@click.option(
    "--race",
    "-r",
    help="Race name or circuit (e.g., 'Monaco', 'British GP')",
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["random_forest", "xgboost", "lightgbm", "ensemble", "baseline"]),
    default="ensemble",
    help="Model to use for prediction",
)
@click.option(
    "--top-n",
    "-n",
    type=int,
    default=10,
    help="Number of predictions to show",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.pass_context
def predict(
    ctx: click.Context,
    race: Optional[str],
    model: str,
    top_n: int,
    output_format: str,
) -> None:
    """Predict race results for upcoming or hypothetical race.

    \b
    Examples:
        f1-predict predict --race Monaco --model ensemble
        f1-predict predict --race "British GP" --top-n 5 --format json
        f1-predict predict --model xgboost --format csv > predictions.csv

    Note: This is a simplified implementation. Full prediction
    requires trained models (see Issue #9).
    """
    # Only show progress messages for table output
    if output_format == "table":
        console.print(
            f"[bold blue]Generating predictions for {race or 'next race'}...[/bold blue]"
        )
        console.print(f"[dim]Using model:[/dim] [cyan]{model}[/cyan]\n")

    # For MVP, show example predictions structure
    # In production, this would load actual model and generate predictions
    example_predictions = [
        {
            "position": 1,
            "driver": "Max Verstappen",
            "team": "Red Bull Racing",
            "confidence": 85.2,
        },
        {
            "position": 2,
            "driver": "Sergio Perez",
            "team": "Red Bull Racing",
            "confidence": 72.8,
        },
        {
            "position": 3,
            "driver": "Charles Leclerc",
            "team": "Ferrari",
            "confidence": 68.5,
        },
        {
            "position": 4,
            "driver": "Carlos Sainz",
            "team": "Ferrari",
            "confidence": 65.1,
        },
        {
            "position": 5,
            "driver": "Lewis Hamilton",
            "team": "Mercedes",
            "confidence": 61.3,
        },
    ]

    predictions = example_predictions[:top_n]

    if output_format == "table":
        table = Table(title="Race Predictions", show_header=True)
        table.add_column("Pos", style="cyan", width=4)
        table.add_column("Driver", style="green")
        table.add_column("Team", style="yellow")
        table.add_column("Confidence", style="magenta")

        for pred in predictions:
            table.add_row(
                str(pred["position"]),
                pred["driver"],
                pred["team"],
                f"{pred['confidence']:.1f}%",
            )

        console.print(table)

        console.print(
            "\n[yellow]Note:[/yellow] This is example output. Full prediction "
            "requires trained models."
        )
        console.print("[dim]See Issue #9 for ML model implementation status.[/dim]")

    elif output_format == "json":
        output = {"race": race, "model": model, "predictions": predictions}
        console.print(json.dumps(output, indent=2))

    elif output_format == "csv":
        df = pd.DataFrame(predictions)
        console.print(df.to_csv(index=False))


# ============================================================================
# DOCUMENTATION COMMAND
# ============================================================================


@cli.command()
@click.argument("topic", required=False)
def docs(topic: Optional[str]) -> None:
    """Show documentation and command examples.

    \b
    Available topics:
        predict - Prediction commands and options
        data    - Data management commands
        analyze - Analysis commands
        model   - Model training and management

    \b
    Examples:
        f1-predict docs
        f1-predict docs predict
        f1-predict docs data
    """
    if not topic:
        console.print(
            Panel(
                "[bold]F1 Prediction System - Documentation[/bold]\n\n"
                "Available command topics:\n\n"
                "  [cyan]predict[/cyan] - Generate race predictions\n"
                "  [cyan]data[/cyan]    - Manage F1 data collection\n"
                "  [cyan]analyze[/cyan] - Analyze driver, team, circuit performance\n"
                "  [cyan]model[/cyan]   - Train and manage prediction models\n\n"
                "[dim]Use[/dim] [yellow]f1-predict docs <topic>[/yellow] [dim]for topic-specific help[/dim]\n"
                "[dim]Use[/dim] [yellow]f1-predict <command> --help[/yellow] [dim]for command help[/dim]",
                title="Documentation",
                border_style="blue",
            )
        )
    else:
        _show_topic_examples(topic)


def _show_topic_examples(topic: str) -> None:
    """Display examples for specific command topic."""
    examples = {
        "predict": [
            (
                "Predict next race with ensemble model",
                "f1-predict predict --race Monaco --model ensemble",
            ),
            (
                "Show top 5 predictions in JSON format",
                "f1-predict predict --race Silverstone --top-n 5 --format json",
            ),
            (
                "Export predictions to CSV",
                "f1-predict predict --race Spa --format csv > predictions.csv",
            ),
        ],
        "data": [
            (
                "Update all data",
                "f1-predict data update --type all",
            ),
            (
                "Show race results for 2024",
                "f1-predict data show --type race-results --season 2024",
            ),
            (
                "View data statistics",
                "f1-predict data stats",
            ),
            (
                "Force refresh race results",
                "f1-predict data update --type race-results --force",
            ),
        ],
        "analyze": [
            (
                "Coming soon in future update",
                "f1-predict analyze driver Hamilton",
            ),
        ],
        "model": [
            (
                "Coming soon in future update",
                "f1-predict model train --type xgboost",
            ),
        ],
    }

    if topic in examples:
        console.print(f"\n[bold]{topic.title()} Command Examples[/bold]\n")

        for description, command in examples[topic]:
            console.print(f"[cyan]{description}:[/cyan]")
            console.print(f"  $ [yellow]{command}[/yellow]\n")
    else:
        console.print(f"[red]Unknown topic:[/red] {topic}")
        console.print("\n[dim]Available topics:[/dim] predict, data, analyze, model")


def main() -> None:
    """Main entry point for enhanced CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
