"""Command line interface for F1 prediction project."""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

from f1_predict import logging_config
from f1_predict.data.cleaning import DataCleaner, DataQualityValidator
from f1_predict.data.collector import F1DataCollector


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging_config.setup_logging(level=level)


def collect_data(args: argparse.Namespace) -> None:
    """Collect historical F1 data.

    Args:
        args: Command line arguments
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        collector = F1DataCollector(data_dir=args.data_dir)

        if args.type == "all":
            logger.info("Collecting all data types")
            results = collector.collect_all_data()
        elif args.type == "race-results":
            logger.info("Collecting race results")
            results = {"race_results": collector.collect_race_results()}
        elif args.type == "qualifying":
            logger.info("Collecting qualifying results")
            results = {"qualifying_results": collector.collect_qualifying_results()}
        elif args.type == "schedules":
            logger.info("Collecting race schedules")
            results = {"race_schedules": collector.collect_race_schedules()}
        else:
            logger.error(f"Unknown data type: {args.type}")
            sys.exit(1)

        # Print summary
        for data_type, file_path in results.items():
            if file_path:
                logger.info(f"✓ {data_type}: {file_path}")
            else:
                logger.warning(f"✗ {data_type}: Collection failed")

        logger.info("Data collection completed")

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def clean_data(args: argparse.Namespace) -> None:
    """Clean F1 data using the cleaning pipeline.

    Args:
        args: Command line arguments
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize cleaner
        cleaner = DataCleaner(enable_logging=True)
        data_dir = Path(args.data_dir)

        # Process each data type
        results = {}

        if args.type == "all" or args.type == "race-results":
            results.update(_clean_race_results(cleaner, data_dir, args))

        if args.type == "all" or args.type == "qualifying":
            results.update(_clean_qualifying_results(cleaner, data_dir, args))

        if args.type == "all" or args.type == "schedules":
            results.update(_clean_schedules(cleaner, data_dir, args))

        # Generate summary report
        _generate_cleaning_summary(results, args.output_dir)

        logger.info("Data cleaning completed")

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _clean_race_results(cleaner: DataCleaner, data_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Clean race results data."""
    logger = logging.getLogger(__name__)

    # Load race results
    race_results_file = data_dir / "raw" / "race_results.json"
    if not race_results_file.exists():
        logger.warning(f"Race results file not found: {race_results_file}")
        return {}

    with open(race_results_file) as f:
        race_data = json.load(f)

    logger.info(f"Cleaning {len(race_data)} race results")

    # Clean data
    cleaned_data, report = cleaner.clean_race_results(race_data)

    # Save cleaned data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = output_dir / "race_results_cleaned.json"
    with open(cleaned_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2, default=str)

    # Validate quality if requested
    quality_passed = cleaner.validate_data_quality(report)

    logger.info(f"Race results cleaned: {len(cleaned_data)}/{len(race_data)} records")
    logger.info(f"Quality score: {report.quality_score:.1f}%")
    logger.info(f"Quality validation: {'PASSED' if quality_passed else 'FAILED'}")

    if args.strict and not quality_passed:
        raise RuntimeError("Data quality validation failed in strict mode")

    return {
        "race_results": {
            "input_file": str(race_results_file),
            "output_file": str(cleaned_file),
            "input_count": len(race_data),
            "output_count": len(cleaned_data),
            "quality_report": report,
            "quality_passed": quality_passed
        }
    }


def _clean_qualifying_results(cleaner: DataCleaner, data_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Clean qualifying results data."""
    logger = logging.getLogger(__name__)

    # Load qualifying results
    qualifying_file = data_dir / "raw" / "qualifying_results.json"
    if not qualifying_file.exists():
        logger.warning(f"Qualifying results file not found: {qualifying_file}")
        return {}

    with open(qualifying_file) as f:
        qualifying_data = json.load(f)

    logger.info(f"Cleaning {len(qualifying_data)} qualifying results")

    # Clean data
    cleaned_data, report = cleaner.clean_qualifying_results(qualifying_data)

    # Save cleaned data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = output_dir / "qualifying_results_cleaned.json"
    with open(cleaned_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2, default=str)

    # Validate quality
    quality_passed = cleaner.validate_data_quality(report)

    logger.info(f"Qualifying results cleaned: {len(cleaned_data)}/{len(qualifying_data)} records")
    logger.info(f"Quality score: {report.quality_score:.1f}%")
    logger.info(f"Quality validation: {'PASSED' if quality_passed else 'FAILED'}")

    if args.strict and not quality_passed:
        raise RuntimeError("Data quality validation failed in strict mode")

    return {
        "qualifying_results": {
            "input_file": str(qualifying_file),
            "output_file": str(cleaned_file),
            "input_count": len(qualifying_data),
            "output_count": len(cleaned_data),
            "quality_report": report,
            "quality_passed": quality_passed
        }
    }


def _clean_schedules(cleaner: DataCleaner, data_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Clean race schedules data."""
    logger = logging.getLogger(__name__)

    # Load race schedules
    schedules_file = data_dir / "raw" / "race_schedules.json"
    if not schedules_file.exists():
        logger.warning(f"Race schedules file not found: {schedules_file}")
        return {}

    with open(schedules_file) as f:
        schedules_data = json.load(f)

    logger.info(f"Cleaning {len(schedules_data)} race schedules")

    # Clean data
    cleaned_data, report = cleaner.clean_race_schedules(schedules_data)

    # Save cleaned data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = output_dir / "race_schedules_cleaned.json"
    with open(cleaned_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2, default=str)

    # Validate quality
    quality_passed = cleaner.validate_data_quality(report)

    logger.info(f"Race schedules cleaned: {len(cleaned_data)}/{len(schedules_data)} records")
    logger.info(f"Quality score: {report.quality_score:.1f}%")
    logger.info(f"Quality validation: {'PASSED' if quality_passed else 'FAILED'}")

    if args.strict and not quality_passed:
        raise RuntimeError("Data quality validation failed in strict mode")

    return {
        "race_schedules": {
            "input_file": str(schedules_file),
            "output_file": str(cleaned_file),
            "input_count": len(schedules_data),
            "output_count": len(cleaned_data),
            "quality_report": report,
            "quality_passed": quality_passed
        }
    }


def _generate_cleaning_summary(results: Dict[str, Any], output_dir: str) -> None:
    """Generate cleaning summary report."""
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    summary_file = output_path / "cleaning_summary.json"

    # Prepare summary data
    summary = {
        "timestamp": str(logger.handlers[0].formatter.converter(None)),
        "total_datasets": len(results),
        "datasets": {}
    }

    for dataset, info in results.items():
        summary["datasets"][dataset] = {
            "input_count": info["input_count"],
            "output_count": info["output_count"],
            "quality_score": info["quality_report"].quality_score,
            "quality_passed": info["quality_passed"],
            "missing_values": len(info["quality_report"].missing_values),
            "validation_errors": len(info["quality_report"].validation_errors),
            "data_type_issues": len(info["quality_report"].data_type_issues),
            "standardization_changes": sum(
                len(changes) for changes in info["quality_report"].standardization_changes.values()
            )
        }

    # Save summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Cleaning summary saved to: {summary_file}")


def validate_data(args: argparse.Namespace) -> None:
    """Validate data quality.

    Args:
        args: Command line arguments
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        validator = DataQualityValidator(strict_mode=args.strict)
        data_dir = Path(args.data_dir)

        # Process each data type
        validation_results = {}

        if args.type == "all" or args.type == "race-results":
            validation_results.update(_validate_race_results(validator, data_dir))

        if args.type == "all" or args.type == "qualifying":
            validation_results.update(_validate_qualifying_results(validator, data_dir))

        if args.type == "all" or args.type == "schedules":
            validation_results.update(_validate_schedules(validator, data_dir))

        # Generate validation report
        _generate_validation_report(validation_results, args.output_dir)

        # Check if any validation failed
        failed_validations = [
            name for name, result in validation_results.items()
            if not result["passed"]
        ]

        if failed_validations:
            logger.error(f"Validation failed for: {', '.join(failed_validations)}")
            if args.strict:
                sys.exit(1)
        else:
            logger.info("All validations passed")

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def _validate_race_results(validator: DataQualityValidator, data_dir: Path) -> Dict[str, Any]:
    """Validate race results data."""
    logger = logging.getLogger(__name__)

    race_results_file = data_dir / "raw" / "race_results.json"
    if not race_results_file.exists():
        logger.warning(f"Race results file not found: {race_results_file}")
        return {}

    with open(race_results_file) as f:
        race_data = json.load(f)

    report = validator.validate_dataset(race_data, "race_results")
    passed = report.quality_score >= 85.0 and len(report.validation_errors) == 0

    logger.info(f"Race results validation: {'PASSED' if passed else 'FAILED'}")
    logger.info(f"Quality score: {report.quality_score:.1f}%")

    return {
        "race_results": {
            "passed": passed,
            "quality_score": report.quality_score,
            "report": report
        }
    }


def _validate_qualifying_results(validator: DataQualityValidator, data_dir: Path) -> Dict[str, Any]:
    """Validate qualifying results data."""
    logger = logging.getLogger(__name__)

    qualifying_file = data_dir / "raw" / "qualifying_results.json"
    if not qualifying_file.exists():
        logger.warning(f"Qualifying results file not found: {qualifying_file}")
        return {}

    with open(qualifying_file) as f:
        qualifying_data = json.load(f)

    report = validator.validate_dataset(qualifying_data, "qualifying_results")
    passed = report.quality_score >= 85.0 and len(report.validation_errors) == 0

    logger.info(f"Qualifying results validation: {'PASSED' if passed else 'FAILED'}")
    logger.info(f"Quality score: {report.quality_score:.1f}%")

    return {
        "qualifying_results": {
            "passed": passed,
            "quality_score": report.quality_score,
            "report": report
        }
    }


def _validate_schedules(validator: DataQualityValidator, data_dir: Path) -> Dict[str, Any]:
    """Validate race schedules data."""
    logger = logging.getLogger(__name__)

    schedules_file = data_dir / "raw" / "race_schedules.json"
    if not schedules_file.exists():
        logger.warning(f"Race schedules file not found: {schedules_file}")
        return {}

    with open(schedules_file) as f:
        schedules_data = json.load(f)

    report = validator.validate_dataset(schedules_data, "race_schedules")
    passed = report.quality_score >= 85.0 and len(report.validation_errors) == 0

    logger.info(f"Race schedules validation: {'PASSED' if passed else 'FAILED'}")
    logger.info(f"Quality score: {report.quality_score:.1f}%")

    return {
        "race_schedules": {
            "passed": passed,
            "quality_score": report.quality_score,
            "report": report
        }
    }


def _generate_validation_report(results: Dict[str, Any], output_dir: str) -> None:
    """Generate validation report."""
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "validation_report.json"

    # Prepare report data
    report_data = {
        "timestamp": str(logger.handlers[0].formatter.converter(None)),
        "overall_passed": all(result["passed"] for result in results.values()),
        "datasets": {}
    }

    for dataset, info in results.items():
        report_data["datasets"][dataset] = {
            "passed": info["passed"],
            "quality_score": info["quality_score"],
            "validation_errors": len(info["report"].validation_errors),
            "missing_values": len(info["report"].missing_values),
            "total_records": info["report"].total_records
        }

    # Save report
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Validation report saved to: {report_file}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="F1 Prediction Project CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect all historical data
  f1-predict collect --type all --data-dir data

  # Clean race results data
  f1-predict clean --type race-results --data-dir data --output-dir data/processed

  # Validate all data with strict mode
  f1-predict validate --type all --data-dir data --strict

  # Collect and clean pipeline
  f1-predict collect --type all --data-dir data
  f1-predict clean --type all --data-dir data --output-dir data/processed
        """
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect historical F1 data")
    collect_parser.add_argument(
        "--type",
        choices=["all", "race-results", "qualifying", "schedules"],
        default="all",
        help="Type of data to collect (default: all)"
    )
    collect_parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)"
    )
    collect_parser.set_defaults(func=collect_data)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean F1 data")
    clean_parser.add_argument(
        "--type",
        choices=["all", "race-results", "qualifying", "schedules"],
        default="all",
        help="Type of data to clean (default: all)"
    )
    clean_parser.add_argument(
        "--data-dir",
        default="data",
        help="Input data directory (default: data)"
    )
    clean_parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory for cleaned data (default: data/processed)"
    )
    clean_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if data quality validation fails"
    )
    clean_parser.set_defaults(func=clean_data)

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data quality")
    validate_parser.add_argument(
        "--type",
        choices=["all", "race-results", "qualifying", "schedules"],
        default="all",
        help="Type of data to validate (default: all)"
    )
    validate_parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory to validate (default: data)"
    )
    validate_parser.add_argument(
        "--output-dir",
        default="data/reports",
        help="Output directory for validation reports (default: data/reports)"
    )
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if validation fails"
    )
    validate_parser.set_defaults(func=validate_data)

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
