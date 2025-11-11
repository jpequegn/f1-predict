"""Real-time F1 data API client for live race data integration.

Provides async interfaces to fetch and stream live F1 data from external APIs:
- OpenF1 API for session data
- F1 Live Timing for high-frequency updates
- Weather and circuit data

This module enables live predictions during race weekends.
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any

import aiohttp
from pydantic import BaseModel, Field


# ============================================================================
# Data Models
# ============================================================================

class SessionType(str, Enum):
    """F1 session types."""
    PRACTICE_1 = "FP1"
    PRACTICE_2 = "FP2"
    PRACTICE_3 = "FP3"
    QUALIFYING = "Q"
    RACE = "R"
    SPRINT = "S"


class SessionStatus(str, Enum):
    """Session execution status."""
    SCHEDULED = "scheduled"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TireCompound(str, Enum):
    """F1 tire compounds."""
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class SessionData(BaseModel):
    """Current F1 session status and information."""
    session_type: SessionType
    status: SessionStatus
    timestamp: datetime
    circuit: str
    year: int
    round: Optional[int] = None
    lap_count: Optional[int] = None  # For races
    total_laps: Optional[int] = None  # For races
    weather_condition: Optional[str] = None
    track_temperature: Optional[float] = None  # Celsius
    air_temperature: Optional[float] = None  # Celsius

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "session_type": "R",
                "status": "ongoing",
                "timestamp": "2024-03-17T14:30:00Z",
                "circuit": "Albert Park",
                "year": 2024,
                "round": 2,
                "lap_count": 25,
                "total_laps": 58,
                "weather_condition": "Clear",
                "track_temperature": 28.5,
                "air_temperature": 22.3,
            }
        }


class DriverPosition(BaseModel):
    """Current position and status of a driver in session."""
    driver_id: str
    driver_name: str
    position: int
    gap_to_leader: Optional[float] = None  # seconds
    gap_to_previous: Optional[float] = None  # seconds
    latest_lap_time: Optional[float] = None  # seconds
    lap_count: Optional[int] = None
    tire_compound: Optional[TireCompound] = None
    laps_on_tire: Optional[int] = None
    pit_stop_count: Optional[int] = None
    dnf: bool = False
    dnf_reason: Optional[str] = None

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "driver_id": "VER",
                "driver_name": "Max Verstappen",
                "position": 1,
                "gap_to_leader": None,
                "gap_to_previous": 1.234,
                "latest_lap_time": 82.456,
                "lap_count": 25,
                "tire_compound": "HARD",
                "laps_on_tire": 12,
                "pit_stop_count": 2,
                "dnf": False,
            }
        }


class WeatherData(BaseModel):
    """Current and forecast weather data."""
    timestamp: datetime
    track_temperature: float  # Celsius
    air_temperature: float  # Celsius
    humidity: Optional[float] = None  # Percentage
    wind_speed: Optional[float] = None  # m/s
    wind_direction: Optional[str] = None
    precipitation_chance: Optional[float] = None  # Percentage
    condition: str  # 'Clear', 'Rainy', 'Cloudy', etc.

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "timestamp": "2024-03-17T14:30:00Z",
                "track_temperature": 28.5,
                "air_temperature": 22.3,
                "humidity": 45.0,
                "wind_speed": 12.5,
                "wind_direction": "NW",
                "precipitation_chance": 5.0,
                "condition": "Clear",
            }
        }


class SessionUpdate(BaseModel):
    """Incremental update from live data stream."""
    timestamp: datetime
    session: SessionData
    positions: List[DriverPosition]
    event_type: Optional[str] = None  # 'lap_complete', 'pit_stop', 'red_flag', etc.
    event_data: Optional[Dict[str, Any]] = None
    weather: Optional[WeatherData] = None

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "timestamp": "2024-03-17T14:30:15Z",
                "session": {
                    "session_type": "R",
                    "status": "ongoing",
                    "timestamp": "2024-03-17T14:30:15Z",
                    "circuit": "Albert Park",
                    "year": 2024,
                    "round": 2,
                    "lap_count": 26,
                    "total_laps": 58,
                },
                "positions": [
                    {
                        "driver_id": "VER",
                        "driver_name": "Max Verstappen",
                        "position": 1,
                    }
                ],
                "event_type": "lap_complete",
                "event_data": {"driver": "VER", "lap": 26},
            }
        }


# ============================================================================
# Real-Time API Client
# ============================================================================

class RealtimeF1APIClient:
    """Client for fetching real-time F1 data.

    Supports multiple data sources:
    - OpenF1 API (primary)
    - F1 Live Timing (high-frequency)
    - Weather APIs (supplementary)

    All methods are async to support non-blocking I/O.
    """

    def __init__(
        self,
        openf1_base_url: str = "https://openf1.org/v1",
        timeout: int = 10,
        rate_limit_delay: float = 0.1,
    ):
        """Initialize real-time API client.

        Args:
            openf1_base_url: Base URL for OpenF1 API
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
        """
        self.openf1_base_url = openf1_base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure session is created."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = asyncio.get_event_loop().time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

    async def _get(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated GET request with rate limiting.

        Args:
            url: URL to request
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            aiohttp.ClientError: If request fails
            asyncio.TimeoutError: If request times out
        """
        await self._apply_rate_limit()
        await self._ensure_session()

        assert self.session is not None
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_current_session(self, circuit: str, year: int = 2024) -> Optional[SessionData]:
        """Get current session information for a circuit.

        Args:
            circuit: Circuit name or code
            year: Season year

        Returns:
            SessionData if session is found, None otherwise
        """
        try:
            url = f"{self.openf1_base_url}/sessions"
            params = {
                "circuit_short_name": circuit,
                "year": year,
            }
            data = await self._get(url, params)

            if not isinstance(data, list) or not data:
                return None

            # Find the most recent/current session
            sessions = sorted(
                data,
                key=lambda x: x.get("date_start", ""),
                reverse=True,
            )

            if not sessions:
                return None

            session_data = sessions[0]
            return SessionData(
                session_type=SessionType(session_data.get("session_type", "FP1")),
                status=SessionStatus.ONGOING,
                timestamp=datetime.now(),
                circuit=circuit,
                year=year,
                round=session_data.get("round_number"),
                weather_condition=session_data.get("weather", "Unknown"),
            )
        except Exception as e:
            print(f"Error fetching current session: {e}")
            return None

    async def get_live_positions(
        self,
        circuit: str,
        session_type: SessionType = SessionType.RACE,
    ) -> List[DriverPosition]:
        """Get current driver positions in session.

        Args:
            circuit: Circuit name or code
            session_type: Type of session (FP, Q, R, etc.)

        Returns:
            List of driver positions
        """
        try:
            url = f"{self.openf1_base_url}/drivers"
            params = {
                "circuit_short_name": circuit,
                "session_type": session_type.value,
            }
            data = await self._get(url, params)

            if not isinstance(data, list):
                return []

            positions = []
            for idx, driver_data in enumerate(data, 1):
                positions.append(
                    DriverPosition(
                        driver_id=driver_data.get("driver_number", str(idx)),
                        driver_name=driver_data.get("full_name", "Unknown"),
                        position=idx,
                        lap_count=driver_data.get("lap_count"),
                        pit_stop_count=driver_data.get("pit_stop_count", 0),
                    )
                )

            return positions
        except Exception as e:
            print(f"Error fetching live positions: {e}")
            return []

    async def get_session_weather(
        self,
        circuit: str,
        session_type: SessionType = SessionType.RACE,
    ) -> Optional[WeatherData]:
        """Get current weather data for session.

        Args:
            circuit: Circuit name or code
            session_type: Type of session

        Returns:
            WeatherData if available, None otherwise
        """
        try:
            # Placeholder - would integrate with actual weather API
            # For now, return mock data
            return WeatherData(
                timestamp=datetime.now(),
                track_temperature=28.5,
                air_temperature=22.3,
                humidity=45.0,
                wind_speed=12.5,
                wind_direction="NW",
                precipitation_chance=5.0,
                condition="Clear",
            )
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return None

    async def stream_session_data(
        self,
        circuit: str,
        session_type: SessionType = SessionType.RACE,
        update_interval: int = 5,
    ) -> AsyncIterator[SessionUpdate]:
        """Stream live session data at regular intervals.

        Args:
            circuit: Circuit name or code
            session_type: Type of session
            update_interval: Seconds between updates

        Yields:
            SessionUpdate objects as they become available

        Example:
            async with RealtimeF1APIClient() as client:
                async for update in client.stream_session_data("Albert Park"):
                    print(f"Update at {update.timestamp}")
                    for pos in update.positions:
                        print(f"{pos.position}. {pos.driver_name}")
        """
        while True:
            try:
                session = await self.get_current_session(circuit)
                positions = await self.get_live_positions(circuit, session_type)
                weather = await self.get_session_weather(circuit, session_type)

                if session and positions:
                    yield SessionUpdate(
                        timestamp=datetime.now(),
                        session=session,
                        positions=positions,
                        weather=weather,
                    )

                await asyncio.sleep(update_interval)

            except Exception as e:
                print(f"Error in stream_session_data: {e}")
                await asyncio.sleep(update_interval)

    async def close(self):
        """Close API client session."""
        if self.session:
            await self.session.close()
            self.session = None


# ============================================================================
# Convenience Functions
# ============================================================================

async def get_current_session(circuit: str) -> Optional[SessionData]:
    """Fetch current session data (convenience function).

    Args:
        circuit: Circuit name or code

    Returns:
        SessionData or None if not found
    """
    async with RealtimeF1APIClient() as client:
        return await client.get_current_session(circuit)


async def get_live_positions(
    circuit: str,
    session_type: SessionType = SessionType.RACE,
) -> List[DriverPosition]:
    """Fetch current driver positions (convenience function).

    Args:
        circuit: Circuit name or code
        session_type: Type of session

    Returns:
        List of driver positions
    """
    async with RealtimeF1APIClient() as client:
        return await client.get_live_positions(circuit, session_type)
