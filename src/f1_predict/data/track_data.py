"""Track characteristics data loader and manager."""

import json
import logging
from pathlib import Path
from typing import Optional

from f1_predict.data.external_models import TrackCharacteristics


class TrackDataManager:
    """Manages track characteristics data."""

    def __init__(self, data_file: Optional[Path] = None):
        """Initialize track data manager.

        Args:
            data_file: Path to track characteristics JSON file
        """
        self.logger = logging.getLogger(__name__)

        if data_file is None:
            # Default to bundled track data
            data_file = Path(__file__).parent.parent.parent.parent / "data" / "external" / "track_characteristics.json"

        self.data_file = Path(data_file)
        self._tracks: dict[str, TrackCharacteristics] = {}
        self._load_tracks()

    def _load_tracks(self) -> None:
        """Load track characteristics from JSON file."""
        if not self.data_file.exists():
            self.logger.warning(f"Track data file not found: {self.data_file}")
            return

        try:
            with open(self.data_file) as f:
                tracks_data = json.load(f)

            for track_dict in tracks_data:
                track = TrackCharacteristics(**track_dict)
                self._tracks[track.circuit_id] = track

            self.logger.info(f"Loaded {len(self._tracks)} track characteristics")

        except Exception as e:
            self.logger.error(f"Failed to load track data: {e}")
            self._tracks = {}

    def get_track(self, circuit_id: str) -> Optional[TrackCharacteristics]:
        """Get track characteristics by circuit ID.

        Args:
            circuit_id: Circuit identifier

        Returns:
            Track characteristics or None if not found
        """
        track = self._tracks.get(circuit_id)

        if track is None:
            self.logger.warning(f"Track characteristics not found for: {circuit_id}")

        return track

    def list_circuits(self) -> list[str]:
        """Get list of all circuit IDs.

        Returns:
            List of circuit identifiers
        """
        return list(self._tracks.keys())

    def get_all_tracks(self) -> dict[str, TrackCharacteristics]:
        """Get all track characteristics.

        Returns:
            Dictionary mapping circuit IDs to track characteristics
        """
        return self._tracks.copy()

    def add_track(self, track: TrackCharacteristics) -> None:
        """Add or update track characteristics.

        Args:
            track: Track characteristics to add/update
        """
        self._tracks[track.circuit_id] = track
        self.logger.info(f"Added/updated track: {track.circuit_id}")

    def save_tracks(self, output_file: Optional[Path] = None) -> None:
        """Save track characteristics to JSON file.

        Args:
            output_file: Output file path (defaults to original file)
        """
        output_path = output_file or self.data_file

        tracks_data = [track.model_dump() for track in self._tracks.values()]

        with open(output_path, "w") as f:
            json.dump(tracks_data, f, indent=2)

        self.logger.info(f"Saved {len(tracks_data)} tracks to {output_path}")

    def get_tracks_by_type(self, track_type: str) -> list[TrackCharacteristics]:
        """Get tracks filtered by type.

        Args:
            track_type: Track type (street, permanent, semi_permanent)

        Returns:
            List of matching tracks
        """
        return [
            track
            for track in self._tracks.values()
            if track.track_type.value == track_type
        ]

    def get_high_downforce_tracks(self) -> list[TrackCharacteristics]:
        """Get tracks requiring high or very high downforce.

        Returns:
            List of high downforce tracks
        """
        return [
            track
            for track in self._tracks.values()
            if track.downforce_level.value in ["high", "very_high"]
        ]

    def get_difficult_overtaking_tracks(self, threshold: int = 7) -> list[TrackCharacteristics]:
        """Get tracks where overtaking is difficult.

        Args:
            threshold: Minimum difficulty rating (1-10)

        Returns:
            List of difficult overtaking tracks
        """
        return [
            track
            for track in self._tracks.values()
            if track.overtaking_difficulty >= threshold
        ]
