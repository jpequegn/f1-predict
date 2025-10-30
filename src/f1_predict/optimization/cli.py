"""CLI commands for hyperparameter optimization."""

import logging

import click

from f1_predict.optimization.hyperparameter_optimizer import (
    HyperparameterOptimizer,
)

logger = logging.getLogger(__name__)


@click.group()
def optimize() -> None:
    """Commands for hyperparameter optimization."""
    pass


@optimize.command()
@click.option(
    "--model-type",
    type=click.Choice(["xgboost", "lightgbm", "random_forest"]),
    required=True,
    help="Type of model to optimize",
)
@click.option(
    "--n-trials",
    type=int,
    default=100,
    help="Maximum number of trials",
)
@click.option(
    "--timeout",
    type=int,
    default=3600,
    help="Timeout in seconds",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/optimized_params",
    help="Output directory for optimized params",
)
def run(
    model_type: str,
    n_trials: int,
    timeout: int,
    output_dir: str,
) -> None:
    """Run hyperparameter optimization for a model.

    Example:
        f1-predict optimize run --model-type xgboost --n-trials 100
    """
    click.echo(f"Starting optimization for {model_type}...")
    click.echo(f"  Trials: {n_trials}")
    click.echo(f"  Timeout: {timeout}s")
    click.echo(f"  Output: {output_dir}")

    try:
        # Create optimizer
        _ = HyperparameterOptimizer(
            model_type=model_type,
            study_name=f"{model_type}_optimization",
            n_trials=n_trials,
            timeout_seconds=timeout,
        )

        click.echo("\n✓ Optimizer created")
        click.echo("NOTE: Full data loading not implemented in CLI stub")
        click.echo("      Use Python API for actual optimization with real data")
        click.echo("\nFor usage example, see docs/HYPERPARAMETER_OPTIMIZATION.md")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        raise click.Abort() from e
