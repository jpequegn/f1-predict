"""Speed trace generator for creating synthetic race visualizations."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


class SpeedTraceGenerator:
    """Generates synthetic speed-trace visualizations from lap data.

    Creates PNG plots showing speed vs lap number for each driver,
    providing visual patterns for multi-modal learning.
    """

    def __init__(self, output_dir: str = 'data/multimodal/speed_traces') -> None:
        """Initialize speed trace generator.

        Args:
            output_dir: Directory where speed trace PNGs are saved
        """
        self.output_dir = Path(output_dir)

    def generate_trace(
        self,
        race_id: str,
        driver_id: str,
        lap_data: List[float]
    ) -> str:
        """Generate speed trace PNG for a single driver.

        Args:
            race_id: Unique race identifier
            driver_id: Unique driver identifier
            lap_data: List of lap speeds (numeric values)

        Returns:
            Path to generated PNG file

        Raises:
            ValueError: If lap_data is empty or invalid
            TypeError: If lap_data contains non-numeric values
        """
        if not lap_data:
            raise ValueError("lap_data cannot be empty")

        # Validate data is numeric
        try:
            numeric_data = [float(x) for x in lap_data]
        except (TypeError, ValueError) as e:
            raise TypeError(f"lap_data must contain numeric values: {e}")

        # Create race-specific directory
        race_dir = self.output_dir / race_id
        race_dir.mkdir(parents=True, exist_ok=True)

        # Generate figure
        fig, ax = plt.subplots(figsize=(10, 7), dpi=80)

        # Plot speed vs lap number
        lap_numbers = list(range(1, len(numeric_data) + 1))
        ax.plot(lap_numbers, numeric_data, 'b-', linewidth=2, label='Speed')
        ax.scatter(lap_numbers, numeric_data, s=50, color='darkblue', alpha=0.6)

        # Formatting
        ax.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
        ax.set_title(f'Speed Trace: {race_id} - {driver_id}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)

        # Add statistics to plot
        min_speed = min(numeric_data)
        max_speed = max(numeric_data)
        avg_speed = sum(numeric_data) / len(numeric_data)

        stats_text = f'Min: {min_speed:.1f}\nMax: {max_speed:.1f}\nAvg: {avg_speed:.1f}'
        ax.text(
            0.98, 0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # Save file
        output_path = race_dir / f'{driver_id}.png'
        fig.tight_layout()
        fig.savefig(output_path, dpi=80, bbox_inches='tight')
        plt.close(fig)

        return str(output_path)

    def generate_batch(self, races: List[Dict]) -> Dict[str, str]:
        """Generate speed traces for multiple races.

        Args:
            races: List of race dictionaries with structure:
                {
                    'race_id': str,
                    'drivers': [
                        {'driver_id': str, 'lap_data': List[float]},
                        ...
                    ]
                }

        Returns:
            Dictionary mapping 'race_id_driver_id' to generated PNG paths

        Raises:
            KeyError: If race dict missing required fields
        """
        results: Dict[str, str] = {}

        for race in races:
            race_id = race['race_id']
            drivers = race['drivers']

            for driver in drivers:
                driver_id = driver['driver_id']
                lap_data = driver['lap_data']

                try:
                    path = self.generate_trace(race_id, driver_id, lap_data)
                    key = f"{race_id}_{driver_id}"
                    results[key] = path
                except (ValueError, TypeError) as e:
                    # Log and continue
                    print(f"Warning: Failed to generate trace for {race_id}/{driver_id}: {e}")
                    continue

        return results
