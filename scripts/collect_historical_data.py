#!/usr/bin/env python3
"""Command-line script for collecting historical F1 data.

This script provides a convenient command-line interface for collecting
historical Formula 1 data from the Ergast API for the years 2020-2024.

Usage:
    python scripts/collect_historical_data.py [options]

Examples:
    # Collect all data
    python scripts/collect_historical_data.py

    # Force refresh all data
    python scripts/collect_historical_data.py --refresh

    # Collect only race results
    python scripts/collect_historical_data.py --race-results

    # Get data summary
    python scripts/collect_historical_data.py --summary
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path to import F1 prediction modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1_predict.data.collector import F1DataCollector


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration.

    Args:
        verbose: If True, enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("data_collection.log")],
    )


def main():
    """Main function for the data collection script."""
    parser = argparse.ArgumentParser(
        description="Collect historical F1 data from Ergast API (2020-2024)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data collection options
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Collect all types of data (default)",
    )
    parser.add_argument(
        "--race-results", action="store_true", help="Collect race results only"
    )
    parser.add_argument(
        "--qualifying", action="store_true", help="Collect qualifying results only"
    )
    parser.add_argument(
        "--schedules", action="store_true", help="Collect race schedules only"
    )

    # Operation options
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh data (re-download even if files exist)",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show summary of existing data files"
    )

    # Configuration options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for storing data files (default: data)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Initialize data collector
        collector = F1DataCollector(data_dir=args.data_dir)

        if args.summary:
            # Show data summary
            summary = collector.get_data_summary()
            print("\n=== F1 Data Collection Summary ===")
            print(f"Data Directory: {summary['data_directory']}")
            print(f"Last Updated: {summary['last_updated']}")

            print(f"\nRaw Files ({len(summary['raw_files'])} files):")
            for filename, info in summary["raw_files"].items():
                size_mb = info["size_bytes"] / (1024 * 1024)
                print(f"  {filename}: {size_mb:.2f} MB (modified: {info['modified']})")

            print(f"\nProcessed Files ({len(summary['processed_files'])} files):")
            for filename, info in summary["processed_files"].items():
                size_mb = info["size_bytes"] / (1024 * 1024)
                print(f"  {filename}: {size_mb:.2f} MB (modified: {info['modified']})")

            return

        # Determine which data to collect
        if args.race_results or args.qualifying or args.schedules:
            # Collect specific data types
            results = {}

            if args.race_results:
                logger.info("Collecting race results...")
                try:
                    race_file = collector.collect_race_results(
                        force_refresh=args.refresh
                    )
                    results["race_results"] = f"Success: {race_file}"
                    print(f"✓ Race results saved to: {race_file}")
                except Exception as e:
                    results["race_results"] = f"Error: {str(e)}"
                    print(f"✗ Error collecting race results: {e}")

            if args.qualifying:
                logger.info("Collecting qualifying results...")
                try:
                    qualifying_file = collector.collect_qualifying_results(
                        force_refresh=args.refresh
                    )
                    results["qualifying_results"] = f"Success: {qualifying_file}"
                    print(f"✓ Qualifying results saved to: {qualifying_file}")
                except Exception as e:
                    results["qualifying_results"] = f"Error: {str(e)}"
                    print(f"✗ Error collecting qualifying results: {e}")

            if args.schedules:
                logger.info("Collecting race schedules...")
                try:
                    schedule_file = collector.collect_race_schedules(
                        force_refresh=args.refresh
                    )
                    results["race_schedules"] = f"Success: {schedule_file}"
                    print(f"✓ Race schedules saved to: {schedule_file}")
                except Exception as e:
                    results["race_schedules"] = f"Error: {str(e)}"
                    print(f"✗ Error collecting race schedules: {e}")

        else:
            # Collect all data
            if args.refresh:
                logger.info("Refreshing all F1 data...")
                results = collector.refresh_data()
            else:
                logger.info("Collecting all F1 data...")
                results = collector.collect_all_data()

            print("\n=== Data Collection Results ===")
            for data_type, result in results.items():
                status = "✓" if result.startswith("Success") else "✗"
                print(f"{status} {data_type}: {result}")

        print(f"\nData collection completed. Files saved to: {args.data_dir}")
        print("Use --summary to see detailed information about collected data.")

    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        print("\nData collection interrupted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
