"""Pydantic models for F1 data structures from the Ergast API.

This module defines the data models that correspond to the Ergast API
response structures, providing type safety and data validation for
all F1-related data.
"""

from datetime import date, datetime
from datetime import time as dt_time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Circuit(BaseModel):
    """F1 circuit information."""

    circuit_id: str = Field(..., alias="circuitId")
    url: str
    circuit_name: str = Field(..., alias="circuitName")
    location: "Location"

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class Location(BaseModel):
    """Geographic location information."""

    lat: float
    long: float
    locality: str
    country: str


class Constructor(BaseModel):
    """F1 constructor (team) information."""

    constructor_id: str = Field(..., alias="constructorId")
    url: str
    name: str
    nationality: str

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class Driver(BaseModel):
    """F1 driver information."""

    driver_id: str = Field(..., alias="driverId")
    permanent_number: Optional[int] = Field(None, alias="permanentNumber")
    code: Optional[str] = None
    url: str
    given_name: str = Field(..., alias="givenName")
    family_name: str = Field(..., alias="familyName")
    date_of_birth: date = Field(..., alias="dateOfBirth")
    nationality: str

    class Config:
        """Pydantic configuration."""

        populate_by_name = True

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def parse_date(cls, v) -> date:
        """Parse date string to date object."""
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v


class Race(BaseModel):
    """F1 race information."""

    season: str
    round: str
    url: str
    race_name: str = Field(..., alias="raceName")
    circuit: Circuit
    date: date
    time: Optional[dt_time] = None
    first_practice: Optional["Session"] = Field(None, alias="FirstPractice")
    second_practice: Optional["Session"] = Field(None, alias="SecondPractice")
    third_practice: Optional["Session"] = Field(None, alias="ThirdPractice")
    qualifying: Optional["Session"] = Field(None, alias="Qualifying")
    sprint: Optional["Session"] = Field(None, alias="Sprint")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v) -> date:
        """Parse date string to date object."""
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @field_validator("time", mode="before")
    @classmethod
    def parse_time(cls, v) -> Optional[time]:
        """Parse time string to time object."""
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%H:%M:%SZ").time()
            except ValueError:
                return None
        return v


class Session(BaseModel):
    """F1 session (practice, qualifying, etc.) information."""

    model_config = ConfigDict(populate_by_name=True)

    date: date
    time: Optional[dt_time] = None

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v) -> date:
        """Parse date string to date object."""
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d").date()
        return v

    @field_validator("time", mode="before")
    @classmethod
    def parse_time(cls, v) -> Optional[dt_time]:
        """Parse time string to time object."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return datetime.strptime(v, "%H:%M:%SZ").time()
            except ValueError:
                return None
        if isinstance(v, dt_time):
            return v
        return v


class Result(BaseModel):
    """F1 race result."""

    number: int
    position: Optional[int] = None
    position_text: str = Field(..., alias="positionText")
    points: float
    driver: Driver
    constructor: Constructor
    grid: int
    laps: int
    status: str
    time: Optional["ResultTime"] = None
    fastest_lap: Optional["FastestLap"] = Field(None, alias="FastestLap")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class ResultTime(BaseModel):
    """Race result time information."""

    millis: Optional[int] = None
    time: Optional[str] = None


class FastestLap(BaseModel):
    """Fastest lap information."""

    rank: int
    lap: int
    time: "LapTime" = Field(..., alias="Time")
    average_speed: "AverageSpeed" = Field(..., alias="AverageSpeed")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class LapTime(BaseModel):
    """Lap time information."""

    time: str


class AverageSpeed(BaseModel):
    """Average speed information."""

    units: str
    speed: float


class QualifyingResult(BaseModel):
    """Qualifying result."""

    number: int
    position: int
    driver: Driver
    constructor: Constructor
    q1: Optional[str] = Field(None, alias="Q1")
    q2: Optional[str] = Field(None, alias="Q2")
    q3: Optional[str] = Field(None, alias="Q3")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class Standing(BaseModel):
    """Base class for standings."""

    position: int
    position_text: str = Field(..., alias="positionText")
    points: float
    wins: int

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class DriverStanding(Standing):
    """Driver championship standing."""

    driver: Driver
    constructors: list[Constructor] = Field(..., alias="Constructors")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class ConstructorStanding(Standing):
    """Constructor championship standing."""

    constructor: Constructor

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class Season(BaseModel):
    """F1 season information."""

    season: str
    url: str


# API Response Wrappers
class MRData(BaseModel):
    """Main response wrapper for Ergast API."""

    xmlns: str
    series: str
    url: str
    limit: str
    offset: str
    total: str


class RaceTable(BaseModel):
    """Race table response."""

    season: Optional[str] = None
    round: Optional[str] = None
    races: list[Race] = Field(default_factory=list, alias="Races")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class ResultsTable(BaseModel):
    """Results table response."""

    season: Optional[str] = None
    round: Optional[str] = None
    race: Optional[Race] = Field(None, alias="Race")
    results: list[Result] = Field(default_factory=list, alias="Results")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class QualifyingTable(BaseModel):
    """Qualifying table response."""

    season: Optional[str] = None
    round: Optional[str] = None
    qualifying_results: list[QualifyingResult] = Field(
        default_factory=list, alias="QualifyingResults"
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class StandingsTable(BaseModel):
    """Standings table response."""

    season: Optional[str] = None
    round: Optional[str] = None
    standings_lists: list["StandingsList"] = Field(
        default_factory=list, alias="StandingsLists"
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class StandingsList(BaseModel):
    """Standings list."""

    season: str
    round: str
    driver_standings: Optional[list[DriverStanding]] = Field(
        None, alias="DriverStandings"
    )
    constructor_standings: Optional[list[ConstructorStanding]] = Field(
        None, alias="ConstructorStandings"
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class DriverTable(BaseModel):
    """Driver table response."""

    season: Optional[str] = None
    drivers: list[Driver] = Field(default_factory=list, alias="Drivers")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class ConstructorTable(BaseModel):
    """Constructor table response."""

    season: Optional[str] = None
    constructors: list[Constructor] = Field(default_factory=list, alias="Constructors")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class CircuitTable(BaseModel):
    """Circuit table response."""

    circuits: list[Circuit] = Field(default_factory=list, alias="Circuits")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class SeasonTable(BaseModel):
    """Season table response."""

    seasons: list[Season] = Field(default_factory=list, alias="Seasons")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


# Main API Response Models
class ErgastResponse(BaseModel):
    """Base Ergast API response."""

    mr_data: MRData = Field(..., alias="MRData")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class RaceResponse(ErgastResponse):
    """Race data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def race_table(self) -> Optional[RaceTable]:
        """Get race table from response."""
        if hasattr(self.mr_data, "RaceTable"):
            return self.mr_data.RaceTable
        return None


class ResultResponse(ErgastResponse):
    """Result data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def results_table(self) -> Optional[ResultsTable]:
        """Get results table from response."""
        if hasattr(self.mr_data, "RaceTable"):
            return self.mr_data.RaceTable
        return None


class QualifyingResponse(ErgastResponse):
    """Qualifying data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def qualifying_table(self) -> Optional[QualifyingTable]:
        """Get qualifying table from response."""
        if hasattr(self.mr_data, "QualifyingTable"):
            return self.mr_data.QualifyingTable
        return None


class StandingsResponse(ErgastResponse):
    """Standings data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def standings_table(self) -> Optional[StandingsTable]:
        """Get standings table from response."""
        if hasattr(self.mr_data, "StandingsTable"):
            return self.mr_data.StandingsTable
        return None


class DriverResponse(ErgastResponse):
    """Driver data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def driver_table(self) -> Optional[DriverTable]:
        """Get driver table from response."""
        if hasattr(self.mr_data, "DriverTable"):
            return self.mr_data.DriverTable
        return None


class ConstructorResponse(ErgastResponse):
    """Constructor data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def constructor_table(self) -> Optional[ConstructorTable]:
        """Get constructor table from response."""
        if hasattr(self.mr_data, "ConstructorTable"):
            return self.mr_data.ConstructorTable
        return None


class CircuitResponse(ErgastResponse):
    """Circuit data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def circuit_table(self) -> Optional[CircuitTable]:
        """Get circuit table from response."""
        if hasattr(self.mr_data, "CircuitTable"):
            return self.mr_data.CircuitTable
        return None


class SeasonResponse(ErgastResponse):
    """Season data response."""

    mr_data: MRData = Field(..., alias="MRData")

    @property
    def season_table(self) -> Optional[SeasonTable]:
        """Get season table from response."""
        if hasattr(self.mr_data, "SeasonTable"):
            return self.mr_data.SeasonTable
        return None
