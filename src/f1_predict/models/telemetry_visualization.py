"""Extended telemetry visualization for multi-modal learning.

Generates various telemetry visualizations including tire degradation,
fuel load estimation, sector times, race strategy patterns, and weather
condition overlays.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of telemetry visualizations."""

    SPEED_TRACE = "speed_trace"
    TIRE_DEGRADATION = "tire_degradation"
    FUEL_LOAD = "fuel_load"
    SECTOR_TIMES = "sector_times"
    STRATEGY_PATTERN = "strategy_pattern"
    LAP_COMPARISON = "lap_comparison"
    POSITION_TRACE = "position_trace"


@dataclass
class TireData:
    """Tire-related telemetry data."""

    compound: str  # soft, medium, hard
    laps_on_tire: list[int]
    lap_times: list[float]  # in seconds
    degradation_rate: Optional[float] = None  # seconds per lap


@dataclass
class FuelData:
    """Fuel-related telemetry data."""

    starting_fuel_kg: float
    fuel_consumption_per_lap: float
    lap_weights: list[float]  # estimated weight per lap


@dataclass
class SectorData:
    """Sector time data."""

    sector1_times: list[float]
    sector2_times: list[float]
    sector3_times: list[float]
    lap_numbers: list[int]


@dataclass
class StrategyData:
    """Race strategy data."""

    pit_stop_laps: list[int]
    tire_compounds: list[str]  # compound for each stint
    stint_lengths: list[int]
    total_laps: int


@dataclass
class WeatherData:
    """Weather condition data."""

    lap_numbers: list[int]
    temperatures: list[float]
    humidity: list[float]
    rain_intensity: list[float]  # 0.0 to 1.0


class TelemetryVisualizer:
    """Generate telemetry visualizations for multi-modal learning.

    Creates various plot types from race telemetry data, optimized
    for CNN feature extraction.
    """

    # Standard figure settings for CNN input
    FIGURE_SIZE = (10, 7)
    DPI = 80
    TARGET_SIZE = (224, 224)

    # Color schemes
    COMPOUND_COLORS = {
        "soft": "#FF0000",
        "medium": "#FFFF00",
        "hard": "#FFFFFF",
        "intermediate": "#00FF00",
        "wet": "#0000FF",
    }

    def __init__(
        self,
        output_dir: str = "data/multimodal/telemetry",
        figure_size: tuple[int, int] = FIGURE_SIZE,
        dpi: int = DPI,
    ):
        """Initialize telemetry visualizer.

        Args:
            output_dir: Directory for saving visualizations
            figure_size: Figure size in inches
            dpi: Resolution for saved images
        """
        self.output_dir = Path(output_dir)
        self.figure_size = figure_size
        self.dpi = dpi

    def _setup_figure(self) -> tuple[Any, Any]:
        """Create standardized figure."""
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        return fig, ax

    def _save_figure(
        self,
        fig: Any,
        race_id: str,
        driver_id: str,
        viz_type: VisualizationType,
    ) -> str:
        """Save figure to file.

        Args:
            fig: Matplotlib figure
            race_id: Race identifier
            driver_id: Driver identifier
            viz_type: Visualization type

        Returns:
            Path to saved file
        """
        # Create directory structure
        race_dir = self.output_dir / race_id / viz_type.value
        race_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        output_path = race_dir / f"{driver_id}.png"
        fig.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return str(output_path)

    def generate_tire_degradation(
        self,
        race_id: str,
        driver_id: str,
        tire_data: TireData,
    ) -> str:
        """Generate tire degradation visualization.

        Shows lap time progression indicating tire wear patterns.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            tire_data: Tire telemetry data

        Returns:
            Path to generated image
        """
        fig, ax = self._setup_figure()

        # Plot lap times
        laps = tire_data.laps_on_tire
        times = tire_data.lap_times
        compound_color = self.COMPOUND_COLORS.get(tire_data.compound.lower(), "#808080")

        ax.plot(laps, times, "o-", color=compound_color, linewidth=2, markersize=6)

        # Add degradation trend line if available
        if len(laps) > 2:
            z = np.polyfit(laps, times, 1)
            p = np.poly1d(z)
            ax.plot(laps, p(laps), "--", color="gray", alpha=0.7, label="Trend")
            degradation = z[0]  # Slope indicates degradation

            # Add degradation info
            ax.text(
                0.02,
                0.98,
                f"Deg: {degradation:.3f}s/lap",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

        # Formatting
        ax.set_xlabel("Lap on Tire", fontsize=12, fontweight="bold")
        ax.set_ylabel("Lap Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Tire Degradation: {race_id} - {driver_id}\n({tire_data.compound})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")

        # Add compound indicator
        ax.fill_between(
            laps, min(times) - 1, max(times) + 1, alpha=0.1, color=compound_color
        )

        return self._save_figure(
            fig, race_id, driver_id, VisualizationType.TIRE_DEGRADATION
        )

    def generate_fuel_load(
        self,
        race_id: str,
        driver_id: str,
        fuel_data: FuelData,
    ) -> str:
        """Generate fuel load visualization.

        Shows estimated car weight over race distance.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            fuel_data: Fuel telemetry data

        Returns:
            Path to generated image
        """
        fig, ax = self._setup_figure()

        laps = list(range(1, len(fuel_data.lap_weights) + 1))
        weights = fuel_data.lap_weights

        # Create area plot for fuel load
        ax.fill_between(laps, weights, alpha=0.3, color="orange")
        ax.plot(laps, weights, "o-", color="darkorange", linewidth=2, markersize=4)

        # Add starting fuel and consumption info
        info_text = (
            f"Start: {fuel_data.starting_fuel_kg:.1f}kg\n"
            f"Rate: {fuel_data.fuel_consumption_per_lap:.2f}kg/lap"
        )
        ax.text(
            0.98,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        # Formatting
        ax.set_xlabel("Lap", fontsize=12, fontweight="bold")
        ax.set_ylabel("Estimated Car Weight (kg)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Fuel Load: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(min(laps) - 0.5, max(laps) + 0.5)

        return self._save_figure(fig, race_id, driver_id, VisualizationType.FUEL_LOAD)

    def generate_sector_times(
        self,
        race_id: str,
        driver_id: str,
        sector_data: SectorData,
    ) -> str:
        """Generate sector times visualization.

        Shows stacked bar chart of sector times per lap.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            sector_data: Sector time data

        Returns:
            Path to generated image
        """
        fig, ax = self._setup_figure()

        laps = sector_data.lap_numbers
        s1 = sector_data.sector1_times
        s2 = sector_data.sector2_times
        s3 = sector_data.sector3_times

        # Stacked bar chart
        width = 0.8
        ax.bar(laps, s1, width, label="Sector 1", color="#FF6B6B")
        ax.bar(laps, s2, width, bottom=s1, label="Sector 2", color="#4ECDC4")
        ax.bar(
            laps,
            s3,
            width,
            bottom=[a + b for a, b in zip(s1, s2)],
            label="Sector 3",
            color="#45B7D1",
        )

        # Add total lap time line
        total_times = [a + b + c for a, b, c in zip(s1, s2, s3)]
        ax2 = ax.twinx()
        ax2.plot(laps, total_times, "k-", linewidth=2, marker="o", markersize=4)
        ax2.set_ylabel("Total Lap Time (s)", fontsize=10)

        # Formatting
        ax.set_xlabel("Lap", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sector Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Sector Times: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        return self._save_figure(
            fig, race_id, driver_id, VisualizationType.SECTOR_TIMES
        )

    def generate_strategy_pattern(
        self,
        race_id: str,
        driver_id: str,
        strategy_data: StrategyData,
    ) -> str:
        """Generate race strategy visualization.

        Shows tire stints and pit stops as a horizontal bar pattern.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            strategy_data: Strategy data

        Returns:
            Path to generated image
        """
        fig, ax = self._setup_figure()

        # Create stint visualization
        current_lap = 0
        for stint_len, compound in zip(
            strategy_data.stint_lengths, strategy_data.tire_compounds
        ):
            color = self.COMPOUND_COLORS.get(compound.lower(), "#808080")
            ax.barh(
                y=0,
                width=stint_len,
                left=current_lap,
                height=0.5,
                color=color,
                edgecolor="black",
                linewidth=1,
            )
            # Add stint label
            ax.text(
                current_lap + stint_len / 2,
                0,
                f"{compound[0].upper()}\n{stint_len}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
            current_lap += stint_len

        # Mark pit stops
        for pit_lap in strategy_data.pit_stop_laps:
            ax.axvline(x=pit_lap, color="red", linestyle="--", linewidth=2)
            ax.text(
                pit_lap,
                0.35,
                "PIT",
                ha="center",
                fontsize=8,
                color="red",
                fontweight="bold",
            )

        # Formatting
        ax.set_xlim(-1, strategy_data.total_laps + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("Lap", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Race Strategy: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_yticks([])

        # Add legend
        for compound, color in self.COMPOUND_COLORS.items():
            ax.plot([], [], "s", color=color, markersize=10, label=compound.capitalize())
        ax.legend(loc="upper right", ncol=3)

        return self._save_figure(
            fig, race_id, driver_id, VisualizationType.STRATEGY_PATTERN
        )

    def generate_position_trace(
        self,
        race_id: str,
        driver_id: str,
        positions: list[int],
    ) -> str:
        """Generate position trace over race.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            positions: List of positions per lap

        Returns:
            Path to generated image
        """
        fig, ax = self._setup_figure()

        laps = list(range(1, len(positions) + 1))

        # Plot position trace (inverted y-axis)
        ax.plot(laps, positions, "b-", linewidth=2, marker="o", markersize=4)
        ax.fill_between(laps, positions, max(positions), alpha=0.3)

        # Invert y-axis (P1 at top)
        ax.invert_yaxis()

        # Add position zones
        ax.axhspan(0.5, 3.5, alpha=0.1, color="gold", label="Podium")
        ax.axhspan(3.5, 10.5, alpha=0.1, color="green", label="Points")

        # Formatting
        ax.set_xlabel("Lap", fontsize=12, fontweight="bold")
        ax.set_ylabel("Position", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Position Trace: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(max(positions) + 0.5, 0.5)

        # Stats
        start_pos = positions[0]
        end_pos = positions[-1]
        best_pos = min(positions)
        stats_text = f"Start: P{start_pos}\nFinish: P{end_pos}\nBest: P{best_pos}"
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
        )

        return self._save_figure(
            fig, race_id, driver_id, VisualizationType.POSITION_TRACE
        )

    def generate_lap_comparison(
        self,
        race_id: str,
        driver_id: str,
        lap_times: list[float],
        reference_times: Optional[list[float]] = None,
        reference_label: str = "Reference",
    ) -> str:
        """Generate lap time comparison visualization.

        Compares driver's lap times to a reference (e.g., average, leader).

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            lap_times: Driver's lap times
            reference_times: Optional reference lap times
            reference_label: Label for reference line

        Returns:
            Path to generated image
        """
        fig, ax = self._setup_figure()

        laps = list(range(1, len(lap_times) + 1))

        # Plot driver lap times
        ax.plot(
            laps,
            lap_times,
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
            label=driver_id,
        )

        # Plot reference if provided
        if reference_times:
            ax.plot(
                laps[: len(reference_times)],
                reference_times,
                "r--",
                linewidth=2,
                marker="s",
                markersize=4,
                label=reference_label,
            )

            # Calculate delta
            deltas = [d - r for d, r in zip(lap_times, reference_times)]
            avg_delta = np.mean(deltas)

            ax2 = ax.twinx()
            ax2.bar(
                laps[: len(deltas)],
                deltas,
                alpha=0.3,
                color=["green" if d < 0 else "red" for d in deltas],
            )
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_ylabel(f"Delta to {reference_label} (s)", fontsize=10)

            # Add average delta
            ax.text(
                0.98,
                0.02,
                f"Avg Delta: {avg_delta:+.3f}s",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

        # Formatting
        ax.set_xlabel("Lap", fontsize=12, fontweight="bold")
        ax.set_ylabel("Lap Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Lap Comparison: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")

        return self._save_figure(
            fig, race_id, driver_id, VisualizationType.LAP_COMPARISON
        )

    def generate_weather_overlay(
        self,
        race_id: str,
        driver_id: str,
        lap_times: list[float],
        weather_data: WeatherData,
    ) -> str:
        """Generate lap times with weather overlay.

        Shows how weather conditions correlate with lap times.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            lap_times: Driver's lap times
            weather_data: Weather data

        Returns:
            Path to generated image
        """
        fig, ax1 = self._setup_figure()

        laps = list(range(1, len(lap_times) + 1))

        # Plot lap times
        ln1 = ax1.plot(
            laps,
            lap_times,
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Lap Time",
        )
        ax1.set_xlabel("Lap", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Lap Time (s)", fontsize=12, color="blue", fontweight="bold")

        # Create second y-axis for temperature
        ax2 = ax1.twinx()
        ln2 = ax2.plot(
            weather_data.lap_numbers,
            weather_data.temperatures,
            "r--",
            linewidth=1.5,
            label="Temperature",
        )
        ax2.set_ylabel("Temperature (°C)", fontsize=10, color="red")

        # Add rain intensity as background shading
        for lap, rain in zip(
            weather_data.lap_numbers[:-1], weather_data.rain_intensity[:-1]
        ):
            if rain > 0:
                ax1.axvspan(lap, lap + 1, alpha=rain * 0.3, color="blue")

        # Formatting
        ax1.set_title(
            f"Weather Overlay: {race_id} - {driver_id}",
            fontsize=14,
            fontweight="bold",
        )

        # Combined legend
        lns = ln1 + ln2
        labs = [str(ll.get_label()) for ll in lns]
        ax1.legend(lns, labs, loc="upper right")
        ax1.grid(True, alpha=0.3, linestyle="--")

        return self._save_figure(
            fig, race_id, driver_id, VisualizationType.SPEED_TRACE  # Reuse type
        )

    def generate_all_visualizations(
        self,
        race_id: str,
        driver_id: str,
        lap_times: list[float],
        positions: list[int],
        tire_data: Optional[TireData] = None,
        fuel_data: Optional[FuelData] = None,
        sector_data: Optional[SectorData] = None,
        strategy_data: Optional[StrategyData] = None,
        weather_data: Optional[WeatherData] = None,
    ) -> dict[str, str]:
        """Generate all available visualizations for a driver.

        Args:
            race_id: Race identifier
            driver_id: Driver identifier
            lap_times: Lap times
            positions: Position per lap
            tire_data: Optional tire data
            fuel_data: Optional fuel data
            sector_data: Optional sector data
            strategy_data: Optional strategy data
            weather_data: Optional weather data

        Returns:
            Dictionary mapping visualization type to file path
        """
        results: dict[str, str] = {}

        # Always generate basic visualizations
        results["position_trace"] = self.generate_position_trace(
            race_id, driver_id, positions
        )
        results["lap_comparison"] = self.generate_lap_comparison(
            race_id, driver_id, lap_times
        )

        # Conditional visualizations
        if tire_data:
            results["tire_degradation"] = self.generate_tire_degradation(
                race_id, driver_id, tire_data
            )

        if fuel_data:
            results["fuel_load"] = self.generate_fuel_load(
                race_id, driver_id, fuel_data
            )

        if sector_data:
            results["sector_times"] = self.generate_sector_times(
                race_id, driver_id, sector_data
            )

        if strategy_data:
            results["strategy_pattern"] = self.generate_strategy_pattern(
                race_id, driver_id, strategy_data
            )

        if weather_data:
            results["weather_overlay"] = self.generate_weather_overlay(
                race_id, driver_id, lap_times, weather_data
            )

        logger.info(
            f"Generated {len(results)} visualizations for {race_id}/{driver_id}"
        )

        return results


def create_synthetic_telemetry(
    race_id: str,
    driver_id: str,
    num_laps: int = 50,
    base_lap_time: float = 90.0,
    seed: Optional[int] = None,
) -> dict:
    """Create synthetic telemetry data for testing.

    Args:
        race_id: Race identifier
        driver_id: Driver identifier
        num_laps: Number of laps
        base_lap_time: Base lap time in seconds
        seed: Random seed

    Returns:
        Dictionary with all telemetry data
    """
    rng = np.random.default_rng(seed)

    # Generate lap times with realistic variation
    lap_times = []
    for i in range(num_laps):
        # Add tire degradation effect
        degradation = 0.02 * (i % 20)  # Reset every 20 laps (pit stop)
        # Add random variation
        variation = rng.normal(0, 0.3)
        # Add traffic effect randomly
        traffic = rng.choice([0, 0.5, 1.0], p=[0.8, 0.15, 0.05])
        lap_times.append(base_lap_time + degradation + variation + traffic)

    # Generate positions
    positions: list[int] = [int(rng.integers(1, 21))]
    for _ in range(1, num_laps):
        change = rng.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
        new_pos = max(1, min(20, positions[-1] + change))
        positions.append(int(new_pos))

    # Generate tire data
    tire_data = TireData(
        compound=rng.choice(["soft", "medium", "hard"]),
        laps_on_tire=list(range(1, min(20, num_laps) + 1)),
        lap_times=lap_times[:20],
    )

    # Generate fuel data
    starting_fuel = rng.uniform(100, 110)
    consumption = rng.uniform(1.5, 2.0)
    fuel_data = FuelData(
        starting_fuel_kg=starting_fuel,
        fuel_consumption_per_lap=consumption,
        lap_weights=[
            starting_fuel - consumption * i + 750  # Car weight
            for i in range(num_laps)
        ],
    )

    # Generate sector data
    total_time = base_lap_time
    s1_frac = rng.uniform(0.28, 0.32)
    s2_frac = rng.uniform(0.33, 0.37)
    s3_frac = 1 - s1_frac - s2_frac

    sector_data = SectorData(
        sector1_times=[
            (total_time + rng.normal(0, 0.2)) * s1_frac for _ in range(num_laps)
        ],
        sector2_times=[
            (total_time + rng.normal(0, 0.2)) * s2_frac for _ in range(num_laps)
        ],
        sector3_times=[
            (total_time + rng.normal(0, 0.2)) * s3_frac for _ in range(num_laps)
        ],
        lap_numbers=list(range(1, num_laps + 1)),
    )

    # Generate strategy data
    num_stops = rng.integers(1, 4)
    pit_laps = sorted(rng.choice(range(10, num_laps - 5), num_stops, replace=False))
    compounds = rng.choice(["soft", "medium", "hard"], num_stops + 1)
    stint_lengths = []
    prev_lap = 0
    for pit_lap in pit_laps:
        stint_lengths.append(pit_lap - prev_lap)
        prev_lap = pit_lap
    stint_lengths.append(num_laps - prev_lap)

    strategy_data = StrategyData(
        pit_stop_laps=list(pit_laps),
        tire_compounds=list(compounds),
        stint_lengths=stint_lengths,
        total_laps=num_laps,
    )

    # Generate weather data
    weather_data = WeatherData(
        lap_numbers=list(range(1, num_laps + 1)),
        temperatures=[20 + rng.normal(0, 2) for _ in range(num_laps)],
        humidity=[60 + rng.normal(0, 10) for _ in range(num_laps)],
        rain_intensity=[
            max(0, rng.normal(-0.3, 0.2)) for _ in range(num_laps)
        ],
    )

    return {
        "race_id": race_id,
        "driver_id": driver_id,
        "lap_times": lap_times,
        "positions": positions,
        "tire_data": tire_data,
        "fuel_data": fuel_data,
        "sector_data": sector_data,
        "strategy_data": strategy_data,
        "weather_data": weather_data,
    }
