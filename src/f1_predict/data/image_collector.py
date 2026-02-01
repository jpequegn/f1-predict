"""Image data collection for multi-modal F1 prediction.

Provides collectors for generating and managing F1 image data including
track layouts, weather condition images, and telemetry visualizations.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImageCategory(Enum):
    """Categories of F1 images."""

    TRACK_LAYOUT = "track_layout"
    WEATHER = "weather"
    TELEMETRY = "telemetry"
    SYNTHETIC = "synthetic"


@dataclass
class ImageMetadata:
    """Metadata for a collected image."""

    image_id: str
    category: str
    race_id: str
    driver_id: Optional[str]
    created_at: str
    file_path: str
    width: int
    height: int
    file_size_bytes: int
    checksum: str
    labels: dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic"


@dataclass
class TrackFeatures:
    """Features of an F1 circuit."""

    circuit_id: str
    name: str
    length_km: float
    num_turns: int
    turn_directions: list[str]  # 'L' or 'R' for each turn
    drs_zones: int
    elevation_change_m: float
    track_type: str  # street, permanent, hybrid


@dataclass
class WeatherConditions:
    """Weather conditions for image generation."""

    condition: str  # clear, cloudy, rain, fog
    temperature_c: float
    humidity_percent: float
    wind_speed_kmh: float
    visibility_km: float


class TrackLayoutGenerator:
    """Generate synthetic track layout images.

    Creates schematic track representations that capture key circuit
    characteristics for CNN feature extraction.
    """

    # Track layout colors
    COLORS = {
        "track": "#333333",
        "kerb": "#FF0000",
        "start_finish": "#FFFFFF",
        "drs": "#00FF00",
        "background": "#1a1a1a",
    }

    def __init__(
        self,
        output_dir: str = "data/multimodal/track_layouts",
        image_size: tuple[int, int] = (224, 224),
    ):
        """Initialize track layout generator.

        Args:
            output_dir: Directory for saving track images
            image_size: Output image dimensions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size

    def generate_track_image(
        self,
        track_features: TrackFeatures,
        seed: Optional[int] = None,
    ) -> tuple[str, ImageMetadata]:
        """Generate a track layout image.

        Args:
            track_features: Circuit features
            seed: Random seed for reproducibility

        Returns:
            Tuple of (file path, metadata)
        """
        rng = np.random.default_rng(seed)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        ax.set_facecolor(self.COLORS["background"])

        # Generate track points based on characteristics
        points = self._generate_track_points(track_features, rng)

        # Draw track
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Main track outline
        ax.plot(
            x_coords + [x_coords[0]],
            y_coords + [y_coords[0]],
            color=self.COLORS["track"],
            linewidth=15,
            solid_capstyle="round",
        )

        # Track surface
        ax.plot(
            x_coords + [x_coords[0]],
            y_coords + [y_coords[0]],
            color="#4a4a4a",
            linewidth=12,
            solid_capstyle="round",
        )

        # Start/finish line
        sf_idx = 0
        ax.plot(
            [x_coords[sf_idx], x_coords[sf_idx]],
            [y_coords[sf_idx] - 0.03, y_coords[sf_idx] + 0.03],
            color=self.COLORS["start_finish"],
            linewidth=3,
        )

        # DRS zones
        num_points = len(points)
        for zone_idx in range(track_features.drs_zones):
            drs_start = (zone_idx + 1) * num_points // (track_features.drs_zones + 2)
            drs_end = drs_start + num_points // 10
            ax.plot(
                x_coords[drs_start:drs_end],
                y_coords[drs_start:drs_end],
                color=self.COLORS["drs"],
                linewidth=14,
                alpha=0.5,
            )

        # Turn numbers
        turn_indices = np.linspace(0, num_points - 1, track_features.num_turns + 1)[
            1:
        ].astype(int)
        for turn_num, idx in enumerate(turn_indices, 1):
            ax.annotate(
                str(turn_num),
                (x_coords[idx], y_coords[idx]),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
            )

        # Clean up axes
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add track name
        ax.text(
            0.5,
            0.02,
            track_features.name,
            transform=ax.transAxes,
            fontsize=10,
            ha="center",
            color="white",
        )

        # Save image
        file_path = self.output_dir / f"{track_features.circuit_id}.png"
        fig.savefig(
            file_path,
            dpi=80,
            bbox_inches="tight",
            facecolor=self.COLORS["background"],
        )
        plt.close(fig)

        # Create metadata
        img = Image.open(file_path)
        checksum = self._compute_checksum(file_path)
        metadata = ImageMetadata(
            image_id=f"track_{track_features.circuit_id}",
            category=ImageCategory.TRACK_LAYOUT.value,
            race_id=track_features.circuit_id,
            driver_id=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            file_path=str(file_path),
            width=img.width,
            height=img.height,
            file_size_bytes=file_path.stat().st_size,
            checksum=checksum,
            labels={
                "circuit_id": track_features.circuit_id,
                "num_turns": track_features.num_turns,
                "drs_zones": track_features.drs_zones,
                "track_type": track_features.track_type,
                "length_km": track_features.length_km,
            },
        )

        return str(file_path), metadata

    def _generate_track_points(
        self,
        features: TrackFeatures,
        rng: np.random.Generator,
    ) -> list[tuple[float, float]]:
        """Generate track outline points.

        Creates a track shape based on circuit characteristics.

        Args:
            features: Track features
            rng: Random number generator

        Returns:
            List of (x, y) coordinates
        """
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points)

        # Base shape with perturbations
        base_radius = 0.35
        points = []

        for angle in angles:
            # Add variation based on turns
            turn_influence = np.sin(angle * features.num_turns / 2) * 0.1

            # Add street circuit "boxiness" if applicable
            if features.track_type == "street":
                box_factor = np.abs(np.sin(angle * 2)) * 0.05
            else:
                box_factor = 0

            # Random variation
            noise = rng.normal(0, 0.02)

            radius = base_radius + turn_influence + box_factor + noise
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)

            # Apply elevation scaling
            elevation_factor = 1 + features.elevation_change_m / 500 * np.sin(angle * 3)
            y = 0.5 + (y - 0.5) * elevation_factor

            points.append((x, y))

        return points

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of file."""
        hasher = hashlib.md5(usedforsecurity=False)  # noqa: S324
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()


class WeatherImageGenerator:
    """Generate synthetic weather condition images.

    Creates visual representations of weather conditions for
    multi-modal feature learning.
    """

    # Weather color palettes
    PALETTES = {
        "clear": {"sky": "#87CEEB", "sun": "#FFD700", "overlay": None},
        "cloudy": {
            "sky": "#A9A9A9",
            "clouds": "#D3D3D3",
            "overlay": (200, 200, 200, 50),
        },
        "rain": {"sky": "#4B5563", "drops": "#6B7280", "overlay": (70, 70, 100, 80)},
        "fog": {"sky": "#D1D5DB", "fog": "#E5E7EB", "overlay": (220, 220, 220, 150)},
    }

    def __init__(
        self,
        output_dir: str = "data/multimodal/weather",
        image_size: tuple[int, int] = (224, 224),
    ):
        """Initialize weather image generator.

        Args:
            output_dir: Directory for saving weather images
            image_size: Output image dimensions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size

    def generate_weather_image(
        self,
        race_id: str,
        conditions: WeatherConditions,
        seed: Optional[int] = None,
    ) -> tuple[str, ImageMetadata]:
        """Generate a weather condition image.

        Args:
            race_id: Race identifier
            conditions: Weather conditions
            seed: Random seed

        Returns:
            Tuple of (file path, metadata)
        """
        rng = np.random.default_rng(seed)

        # Create base image
        width, height = self.image_size
        img_array = np.zeros((height, width, 3), dtype=np.uint8)

        palette = self.PALETTES.get(conditions.condition, self.PALETTES["clear"])

        # Fill background with sky color
        sky_color = self._hex_to_rgb(palette["sky"])
        img_array[:, :] = sky_color

        # Add condition-specific elements
        if conditions.condition == "clear":
            self._add_sun(img_array, rng)
        elif conditions.condition == "cloudy":
            self._add_clouds(img_array, rng)
        elif conditions.condition == "rain":
            self._add_rain(img_array, rng, intensity=conditions.humidity_percent / 100)
        elif conditions.condition == "fog":
            self._add_fog(img_array, rng, density=1 - conditions.visibility_km / 10)

        # Add temperature indicator
        self._add_temperature_bar(img_array, conditions.temperature_c)

        # Add wind indicator
        self._add_wind_indicator(img_array, conditions.wind_speed_kmh, rng)

        # Convert to PIL and save
        img = Image.fromarray(img_array)
        file_path = self.output_dir / f"{race_id}_weather.png"
        img.save(file_path)

        # Create metadata
        checksum = self._compute_checksum(file_path)
        metadata = ImageMetadata(
            image_id=f"weather_{race_id}",
            category=ImageCategory.WEATHER.value,
            race_id=race_id,
            driver_id=None,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            file_path=str(file_path),
            width=width,
            height=height,
            file_size_bytes=file_path.stat().st_size,
            checksum=checksum,
            labels={
                "condition": conditions.condition,
                "temperature_c": conditions.temperature_c,
                "humidity_percent": conditions.humidity_percent,
                "wind_speed_kmh": conditions.wind_speed_kmh,
                "visibility_km": conditions.visibility_km,
            },
        )

        return str(file_path), metadata

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]

    def _add_sun(
        self,
        img: np.ndarray,
        rng: np.random.Generator,  # noqa: ARG002 - reserved for variation
    ) -> None:
        """Add sun to image."""
        h, w = img.shape[:2]
        cx, cy = int(w * 0.8), int(h * 0.2)
        radius = int(min(w, h) * 0.08)

        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        img[mask] = [255, 215, 0]  # Gold color

    def _add_clouds(self, img: np.ndarray, rng: np.random.Generator) -> None:
        """Add clouds to image."""
        h, w = img.shape[:2]
        num_clouds = rng.integers(3, 7)

        for _ in range(num_clouds):
            cx = rng.integers(int(w * 0.1), int(w * 0.9))
            cy = rng.integers(int(h * 0.1), int(h * 0.4))
            cloud_w = rng.integers(int(w * 0.1), int(w * 0.2))
            cloud_h = rng.integers(int(h * 0.05), int(h * 0.1))

            y, x = np.ogrid[:h, :w]
            mask = ((x - cx) ** 2 / cloud_w**2 + (y - cy) ** 2 / cloud_h**2) <= 1
            img[mask] = [211, 211, 211]  # Light gray

    def _add_rain(
        self,
        img: np.ndarray,
        rng: np.random.Generator,
        intensity: float = 0.5,
    ) -> None:
        """Add rain drops to image."""
        h, w = img.shape[:2]
        num_drops = int(100 * intensity)

        for _ in range(num_drops):
            x = rng.integers(0, w)
            y = rng.integers(0, h)
            length = rng.integers(5, 15)

            y_end = min(h - 1, y + length)
            if y < y_end:
                img[y:y_end, x] = [100, 100, 150]  # Rain color

    def _add_fog(
        self,
        img: np.ndarray,
        rng: np.random.Generator,  # noqa: ARG002 - reserved for variation
        density: float = 0.3,
    ) -> None:
        """Add fog overlay to image."""
        fog_layer = np.full_like(img, 220, dtype=np.float32)

        # Blend with original
        alpha = min(0.7, density)
        img[:] = (1 - alpha) * img + alpha * fog_layer

    def _add_temperature_bar(self, img: np.ndarray, temp_c: float) -> None:
        """Add temperature indicator bar."""
        h, _w = img.shape[:2]

        # Temperature bar on left edge
        bar_width = 10
        bar_x = 5

        # Map temperature to color (blue = cold, red = hot)
        # Range: 0-40°C
        temp_normalized = np.clip((temp_c - 0) / 40, 0, 1)
        r = int(255 * temp_normalized)
        b = int(255 * (1 - temp_normalized))
        color = [r, 0, b]

        # Fill bar height based on temperature
        bar_height = int(h * 0.6 * temp_normalized) + int(h * 0.2)
        bar_top = h - bar_height - int(h * 0.1)

        img[bar_top : bar_top + bar_height, bar_x : bar_x + bar_width] = color

    def _add_wind_indicator(
        self,
        img: np.ndarray,
        wind_speed: float,
        rng: np.random.Generator,
    ) -> None:
        """Add wind speed indicator (lines suggesting wind)."""
        h, w = img.shape[:2]

        # Number of wind lines based on speed
        num_lines = int(wind_speed / 10)

        for _ in range(min(num_lines, 10)):
            y = rng.integers(int(h * 0.4), int(h * 0.8))
            x_start = rng.integers(int(w * 0.2), int(w * 0.5))
            x_end = x_start + rng.integers(20, 50)
            x_end = min(x_end, w - 1)

            img[y, x_start:x_end] = [200, 200, 200]

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute MD5 checksum of file."""
        hasher = hashlib.md5(usedforsecurity=False)  # noqa: S324
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()


class ImageMetadataManager:
    """Manage metadata for collected images.

    Provides storage, retrieval, and querying of image metadata
    for the multi-modal dataset.
    """

    def __init__(
        self,
        metadata_dir: str = "data/multimodal/metadata",
    ):
        """Initialize metadata manager.

        Args:
            metadata_dir: Directory for metadata storage
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.metadata_dir / "image_metadata.json"
        self._metadata: dict[str, dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load existing metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self._metadata = json.load(f)
            logger.info(f"Loaded {len(self._metadata)} image metadata records")

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2)

    def add_metadata(self, metadata: ImageMetadata) -> None:
        """Add or update image metadata.

        Args:
            metadata: Image metadata to store
        """
        self._metadata[metadata.image_id] = asdict(metadata)
        self._save_metadata()

    def get_metadata(self, image_id: str) -> Optional[ImageMetadata]:
        """Retrieve metadata for an image.

        Args:
            image_id: Image identifier

        Returns:
            ImageMetadata if found, None otherwise
        """
        data = self._metadata.get(image_id)
        if data:
            return ImageMetadata(**data)
        return None

    def query_by_category(self, category: ImageCategory) -> list[ImageMetadata]:
        """Query images by category.

        Args:
            category: Image category

        Returns:
            List of matching ImageMetadata
        """
        results = []
        for data in self._metadata.values():
            if data["category"] == category.value:
                results.append(ImageMetadata(**data))
        return results

    def query_by_race(self, race_id: str) -> list[ImageMetadata]:
        """Query images by race.

        Args:
            race_id: Race identifier

        Returns:
            List of matching ImageMetadata
        """
        results = []
        for data in self._metadata.values():
            if data["race_id"] == race_id:
                results.append(ImageMetadata(**data))
        return results

    def get_all_metadata(self) -> list[ImageMetadata]:
        """Get all image metadata.

        Returns:
            List of all ImageMetadata records
        """
        return [ImageMetadata(**data) for data in self._metadata.values()]

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of collected images.

        Returns:
            Summary dictionary with counts and categories
        """
        categories: dict[str, int] = {}
        total_size = 0

        for data in self._metadata.values():
            cat = data["category"]
            categories[cat] = categories.get(cat, 0) + 1
            total_size += data.get("file_size_bytes", 0)

        return {
            "total_images": len(self._metadata),
            "by_category": categories,
            "total_size_mb": total_size / (1024 * 1024),
        }


class ImageDataCollector:
    """Unified image data collection pipeline.

    Orchestrates generation of all image types and manages
    the resulting dataset.
    """

    def __init__(
        self,
        output_dir: str = "data/multimodal",
    ):
        """Initialize image data collector.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize generators
        self.track_generator = TrackLayoutGenerator(
            output_dir=str(self.output_dir / "track_layouts")
        )
        self.weather_generator = WeatherImageGenerator(
            output_dir=str(self.output_dir / "weather")
        )

        # Metadata manager
        self.metadata_manager = ImageMetadataManager(
            metadata_dir=str(self.output_dir / "metadata")
        )

    def collect_track_layouts(
        self,
        circuits: list[TrackFeatures],
    ) -> dict[str, str]:
        """Generate track layout images for circuits.

        Args:
            circuits: List of circuit features

        Returns:
            Dictionary mapping circuit_id to file path
        """
        results: dict[str, str] = {}

        for circuit in circuits:
            try:
                file_path, metadata = self.track_generator.generate_track_image(
                    circuit,
                    seed=hash(circuit.circuit_id) % (2**31),
                )
                self.metadata_manager.add_metadata(metadata)
                results[circuit.circuit_id] = file_path
                logger.info(f"Generated track layout for {circuit.circuit_id}")
            except Exception as e:
                logger.error(f"Failed to generate track for {circuit.circuit_id}: {e}")

        return results

    def collect_weather_images(
        self,
        race_conditions: list[tuple[str, WeatherConditions]],
    ) -> dict[str, str]:
        """Generate weather condition images.

        Args:
            race_conditions: List of (race_id, conditions) tuples

        Returns:
            Dictionary mapping race_id to file path
        """
        results: dict[str, str] = {}

        for race_id, conditions in race_conditions:
            try:
                file_path, metadata = self.weather_generator.generate_weather_image(
                    race_id,
                    conditions,
                    seed=hash(race_id) % (2**31),
                )
                self.metadata_manager.add_metadata(metadata)
                results[race_id] = file_path
                logger.info(f"Generated weather image for {race_id}")
            except Exception as e:
                logger.error(f"Failed to generate weather for {race_id}: {e}")

        return results

    def get_dataset_summary(self) -> dict[str, Any]:
        """Get summary of collected image data.

        Returns:
            Summary dictionary
        """
        return self.metadata_manager.get_summary()


# Example circuits for testing
SAMPLE_CIRCUITS = [
    TrackFeatures(
        circuit_id="monaco",
        name="Monaco",
        length_km=3.337,
        num_turns=19,
        turn_directions=["R", "L"] * 10,
        drs_zones=1,
        elevation_change_m=42,
        track_type="street",
    ),
    TrackFeatures(
        circuit_id="silverstone",
        name="Silverstone",
        length_km=5.891,
        num_turns=18,
        turn_directions=["R", "L", "R", "L"] * 5,
        drs_zones=2,
        elevation_change_m=19,
        track_type="permanent",
    ),
    TrackFeatures(
        circuit_id="spa",
        name="Spa-Francorchamps",
        length_km=7.004,
        num_turns=19,
        turn_directions=["L", "R"] * 10,
        drs_zones=2,
        elevation_change_m=102,
        track_type="permanent",
    ),
    TrackFeatures(
        circuit_id="singapore",
        name="Singapore",
        length_km=5.063,
        num_turns=23,
        turn_directions=["L", "R", "L"] * 8,
        drs_zones=3,
        elevation_change_m=10,
        track_type="street",
    ),
]
