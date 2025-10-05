# Data Models Reference

This document provides comprehensive documentation of all Pydantic models, database schemas, and API contracts used in the F1 Prediction System.

## Core Pydantic Models

All models are defined in `src/f1_predict/data/models.py` and use Pydantic v2 for validation.

### Race Model

Represents a Formula 1 race event.

```python
from pydantic import BaseModel, Field
from datetime import datetime

class Race(BaseModel):
    """Formula 1 race information."""

    season: int = Field(..., ge=1950, le=2100, description="Championship year")
    round: int = Field(..., ge=1, le=25, description="Race number in season")
    race_name: str = Field(..., alias="raceName", description="Official race name")
    circuit: Circuit = Field(..., description="Circuit information")
    date: datetime = Field(..., description="Race date")
    time: datetime | None = Field(None, description="Race start time (optional)")
    url: str = Field(..., description="Wikipedia URL")

    class Config:
        populate_by_name = True  # Allow both alias and field name
```

**Field Descriptions**:
- `season`: Championship year (1950-2100 validation range)
- `round`: Race number within season (1-25, max races per season)
- `race_name`: Official race name (e.g., "Monaco Grand Prix")
- `circuit`: Nested Circuit model with track details
- `date`: Race date (YYYY-MM-DD format from API)
- `time`: Race start time (optional, may be TBD for future races)
- `url`: Wikipedia page URL for the race

**Example**:
```json
{
    "season": 2024,
    "round": 6,
    "raceName": "Monaco Grand Prix",
    "circuit": {...},
    "date": "2024-05-26",
    "time": "13:00:00",
    "url": "https://en.wikipedia.org/wiki/2024_Monaco_Grand_Prix"
}
```

---

### Driver Model

Represents an F1 driver.

```python
class Driver(BaseModel):
    """F1 driver information."""

    driver_id: str = Field(..., alias="driverId", description="Unique driver ID")
    permanent_number: int | None = Field(
        None, alias="permanentNumber", description="Permanent race number"
    )
    code: str | None = Field(None, description="3-letter driver code")
    given_name: str = Field(..., alias="givenName", description="First name")
    family_name: str = Field(..., alias="familyName", description="Last name")
    date_of_birth: datetime = Field(..., alias="dateOfBirth", description="Birth date")
    nationality: str = Field(..., description="Driver nationality")
    url: str = Field(..., description="Wikipedia URL")
```

**Field Descriptions**:
- `driver_id`: Unique identifier (e.g., "verstappen", "hamilton")
- `permanent_number`: Driver's permanent race number (e.g., 1, 44, 33)
- `code`: Three-letter code (e.g., "VER", "HAM", "LEC")
- `given_name`: First/given name
- `family_name`: Last/family name
- `date_of_birth`: Birth date for age calculations
- `nationality`: Driver's nationality (e.g., "Dutch", "British")
- `url`: Wikipedia page URL

**Example**:
```json
{
    "driverId": "verstappen",
    "permanentNumber": 1,
    "code": "VER",
    "givenName": "Max",
    "familyName": "Verstappen",
    "dateOfBirth": "1997-09-30",
    "nationality": "Dutch",
    "url": "http://en.wikipedia.org/wiki/Max_Verstappen"
}
```

---

### Constructor Model

Represents an F1 team/constructor.

```python
class Constructor(BaseModel):
    """F1 team/constructor information."""

    constructor_id: str = Field(..., alias="constructorId", description="Unique team ID")
    name: str = Field(..., description="Constructor name")
    nationality: str = Field(..., description="Constructor nationality")
    url: str = Field(..., description="Wikipedia URL")
```

**Field Descriptions**:
- `constructor_id`: Unique identifier (e.g., "red_bull", "ferrari")
- `name`: Official constructor name (e.g., "Red Bull Racing")
- `nationality`: Constructor's nationality (e.g., "Austrian", "Italian")
- `url`: Wikipedia page URL

**Example**:
```json
{
    "constructorId": "red_bull",
    "name": "Red Bull Racing",
    "nationality": "Austrian",
    "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing"
}
```

---

### Circuit Model

Represents an F1 racing circuit.

```python
class Circuit(BaseModel):
    """F1 circuit information."""

    circuit_id: str = Field(..., alias="circuitId", description="Unique circuit ID")
    circuit_name: str = Field(..., alias="circuitName", description="Circuit name")
    location: Location = Field(..., description="Circuit location")
    url: str = Field(..., description="Wikipedia URL")
```

**Nested Location Model**:
```python
class Location(BaseModel):
    """Geographic location."""

    lat: float = Field(..., description="Latitude")
    long: float = Field(..., description="Longitude")
    locality: str = Field(..., description="City/locality")
    country: str = Field(..., description="Country")
```

**Example**:
```json
{
    "circuitId": "monaco",
    "circuitName": "Circuit de Monaco",
    "location": {
        "lat": 43.7347,
        "long": 7.4206,
        "locality": "Monte-Carlo",
        "country": "Monaco"
    },
    "url": "http://en.wikipedia.org/wiki/Circuit_de_Monaco"
}
```

---

### Result Model

Represents a driver's race result.

```python
class Result(BaseModel):
    """Race result for a single driver."""

    number: int = Field(..., description="Car number")
    position: int | None = Field(None, description="Finishing position")
    position_text: str = Field(..., alias="positionText", description="Position as text")
    points: float = Field(..., description="Points scored")
    driver: Driver = Field(..., description="Driver information")
    constructor: Constructor = Field(..., description="Team information")
    grid: int = Field(..., description="Starting grid position")
    laps: int = Field(..., description="Laps completed")
    status: str = Field(..., description="Finish status")
    time: Time | None = Field(None, alias="Time", description="Race time")
    fastest_lap: FastestLap | None = Field(
        None, alias="FastestLap", description="Fastest lap info"
    )
```

**Field Descriptions**:
- `number`: Race car number
- `position`: Final position (None if DNF)
- `position_text`: Position as string ("1", "2", "R" for retired)
- `points`: Championship points awarded
- `grid`: Starting grid position (1-20)
- `laps`: Total laps completed
- `status`: Finish status ("Finished", "+1 Lap", "Collision", etc.)
- `time`: Total race time (winner only)
- `fastest_lap`: Fastest lap information (if achieved)

---

## External Data Models

Defined in `src/f1_predict/data/external_models.py`.

### WeatherData Model

Weather conditions for a race session.

```python
from enum import Enum

class WeatherCondition(str, Enum):
    """Weather condition types."""
    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    DRIZZLE = "drizzle"
    FOG = "fog"
    MIST = "mist"
    SNOW = "snow"

class WeatherData(BaseModel):
    """Weather data for a race session."""

    session_date: datetime = Field(..., description="Session date and time")
    circuit_id: str = Field(..., description="Circuit identifier")
    season: str = Field(..., description="Season year")
    round: str = Field(..., description="Round number")
    air_temperature: float = Field(..., description="Air temp (°C)")
    track_temperature: float | None = Field(None, description="Track temp (°C)")
    condition: WeatherCondition = Field(..., description="Weather condition")
    precipitation_mm: float = Field(0.0, description="Precipitation (mm)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity %")
    wind_speed: float = Field(..., description="Wind speed (km/h)")
    wind_direction: int | None = Field(None, ge=0, le=360, description="Wind direction")
    pressure: float | None = Field(None, description="Atmospheric pressure (hPa)")
```

---

### TrackCharacteristics Model

Track-specific characteristics affecting race outcomes.

```python
class TrackType(str, Enum):
    """Type of racing circuit."""
    STREET = "street"
    PERMANENT = "permanent"
    HYBRID = "hybrid"

class DownforceLevel(str, Enum):
    """Aerodynamic downforce requirement."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class TrackCharacteristics(BaseModel):
    """Physical and strategic characteristics of an F1 circuit."""

    circuit_id: str = Field(..., description="Circuit identifier")
    circuit_name: str = Field(..., description="Official circuit name")
    length_km: float = Field(..., description="Lap length (km)")
    number_of_corners: int = Field(..., description="Total corners")
    number_of_drs_zones: int | None = Field(None, description="DRS zones")
    track_type: TrackType = Field(..., description="Circuit type")
    downforce_level: DownforceLevel = Field(..., description="Downforce requirement")
    overtaking_difficulty: int = Field(..., ge=1, le=10, description="Overtaking (1-10)")
    surface_roughness: float | None = Field(None, description="Surface roughness")
    asphalt_age: int | None = Field(None, description="Asphalt age (years)")
    grip_level: int | None = Field(None, ge=1, le=10, description="Grip level (1-10)")
    average_safety_car_probability: float = Field(
        ..., description="Safety car probability"
    )
    track_limits_severity: int | None = Field(
        None, ge=1, le=10, description="Track limits (1-10)"
    )
    average_lap_time_seconds: float | None = Field(None, description="Avg lap time")
    top_speed_km_h: float | None = Field(None, description="Top speed (km/h)")
    power_unit_stress: int | None = Field(None, ge=1, le=10, description="PU stress")
    brake_stress: int | None = Field(None, ge=1, le=10, description="Brake stress")
    tire_stress: int | None = Field(None, ge=1, le=10, description="Tire stress")
```

**Example**:
```json
{
    "circuit_id": "monaco",
    "circuit_name": "Circuit de Monaco",
    "length_km": 3.337,
    "number_of_corners": 19,
    "number_of_drs_zones": 1,
    "track_type": "street",
    "downforce_level": "very_high",
    "overtaking_difficulty": 10,
    "average_safety_car_probability": 0.75,
    "top_speed_km_h": 290.0,
    "power_unit_stress": 4,
    "brake_stress": 6,
    "tire_stress": 4
}
```

---

### TireStintData Model

Tire compound usage and degradation data.

```python
class TireCompound(str, Enum):
    """F1 tire compound types."""
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"

class TireStintData(BaseModel):
    """Tire usage data for a race stint."""

    session_type: str = Field(..., description="Session type (race, qualifying)")
    season: str = Field(..., description="Season year")
    round: str = Field(..., description="Round number")
    driver_id: str = Field(..., description="Driver identifier")
    compound: TireCompound = Field(..., description="Tire compound")
    stint_number: int = Field(..., description="Stint number in race")
    starting_lap: int = Field(..., description="First lap of stint")
    ending_lap: int = Field(..., description="Last lap of stint")
    laps_completed: int = Field(..., description="Laps on this compound")
    average_lap_time: float | None = Field(None, description="Average lap time")
    fastest_lap_time: float | None = Field(None, description="Fastest lap time")
    degradation_rate: float | None = Field(None, description="Degradation rate")
    stint_end_reason: str | None = Field(None, description="Why stint ended")
```

---

## Database Schemas

### SQLite Schema (Future Enhancement)

Planned database schema for persistent storage:

```sql
-- Races table
CREATE TABLE races (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    race_name TEXT NOT NULL,
    circuit_id TEXT NOT NULL,
    race_date DATE NOT NULL,
    race_time TIME,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, round)
);

-- Drivers table
CREATE TABLE drivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    driver_id TEXT NOT NULL UNIQUE,
    permanent_number INTEGER,
    code TEXT,
    given_name TEXT NOT NULL,
    family_name TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    nationality TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Constructors table
CREATE TABLE constructors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    constructor_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    nationality TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Race results table
CREATE TABLE race_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver_id TEXT NOT NULL,
    constructor_id TEXT NOT NULL,
    grid_position INTEGER,
    final_position INTEGER,
    position_text TEXT,
    points REAL,
    laps INTEGER,
    status TEXT,
    fastest_lap_rank INTEGER,
    fastest_lap_time TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
    FOREIGN KEY (constructor_id) REFERENCES constructors(constructor_id),
    UNIQUE(race_id, driver_id)
);

-- Qualifying results table
CREATE TABLE qualifying_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    driver_id TEXT NOT NULL,
    constructor_id TEXT NOT NULL,
    position INTEGER,
    q1_time TEXT,
    q2_time TEXT,
    q3_time TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (driver_id) REFERENCES drivers(driver_id),
    FOREIGN KEY (constructor_id) REFERENCES constructors(constructor_id),
    UNIQUE(race_id, driver_id)
);

-- Weather data table
CREATE TABLE weather_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    session_type TEXT NOT NULL,
    air_temperature REAL,
    track_temperature REAL,
    condition TEXT,
    precipitation_mm REAL,
    humidity REAL,
    wind_speed REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES races(id)
);

-- Indexes for performance
CREATE INDEX idx_race_results_season ON race_results(race_id);
CREATE INDEX idx_race_results_driver ON race_results(driver_id);
CREATE INDEX idx_qualifying_season ON qualifying_results(race_id);
```

---

## API Contracts

### Ergast F1 API Response Format

**Base Response Structure**:
```json
{
    "MRData": {
        "xmlns": "http://ergast.com/mrd/1.5",
        "series": "f1",
        "url": "http://ergast.com/api/f1/2024/results",
        "limit": "30",
        "offset": "0",
        "total": "440",
        "RaceTable": {
            "season": "2024",
            "Races": [...]
        }
    }
}
```

**Race Results Response**:
```json
{
    "MRData": {
        "RaceTable": {
            "Races": [
                {
                    "season": "2024",
                    "round": "1",
                    "raceName": "Bahrain Grand Prix",
                    "Circuit": {
                        "circuitId": "bahrain",
                        "circuitName": "Bahrain International Circuit",
                        "Location": {
                            "lat": "26.0325",
                            "long": "50.5106",
                            "locality": "Sakhir",
                            "country": "Bahrain"
                        }
                    },
                    "date": "2024-03-02",
                    "time": "15:00:00Z",
                    "Results": [
                        {
                            "number": "1",
                            "position": "1",
                            "positionText": "1",
                            "points": "25",
                            "Driver": {
                                "driverId": "verstappen",
                                "permanentNumber": "1",
                                "code": "VER",
                                "givenName": "Max",
                                "familyName": "Verstappen"
                            },
                            "Constructor": {
                                "constructorId": "red_bull",
                                "name": "Red Bull Racing"
                            },
                            "grid": "1",
                            "laps": "57",
                            "status": "Finished"
                        }
                    ]
                }
            ]
        }
    }
}
```

### OpenWeatherMap API Response

**Current Weather Response**:
```json
{
    "coord": {"lon": 7.4206, "lat": 43.7347},
    "weather": [
        {
            "id": 800,
            "main": "Clear",
            "description": "clear sky",
            "icon": "01d"
        }
    ],
    "main": {
        "temp": 22.5,
        "feels_like": 22.3,
        "pressure": 1013,
        "humidity": 65
    },
    "wind": {
        "speed": 3.5,
        "deg": 180
    },
    "dt": 1684339200
}
```

## Validation Rules

### Data Quality Checks

Implemented in `src/f1_predict/data/cleaning.py`:

```python
# Position validation
assert 1 <= result.position <= 20, "Position must be 1-20"

# Points validation
assert 0 <= result.points <= 26, "Points must be 0-26"

# Laps validation
assert result.laps >= 0, "Laps cannot be negative"
assert result.laps <= 100, "Laps exceeds maximum"

# Grid position validation
assert 1 <= result.grid <= 20, "Grid position must be 1-20"

# Date validation
assert 1950 <= result.season <= 2100, "Invalid season year"

# Coordinate validation
assert -90 <= circuit.location.lat <= 90, "Invalid latitude"
assert -180 <= circuit.location.long <= 180, "Invalid longitude"
```

---

## Model Versioning

Models follow semantic versioning based on breaking changes:

**Version Format**: `v{major}.{minor}.{patch}`

**Breaking Changes** (increment major):
- Field removal
- Field type change
- Required field addition

**Non-Breaking Changes** (increment minor):
- Optional field addition
- Enum value addition
- Validation rule relaxation

**Patch Changes**:
- Documentation updates
- Bug fixes in validation logic

Current versions tracked in `src/f1_predict/data/models.py`:

```python
__version__ = "1.0.0"
```

---

## References

- [Ergast F1 API Documentation](http://ergast.com/mrd/)
- [OpenWeatherMap API Documentation](https://openweathermap.org/api)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Architecture Overview](../architecture/overview.md)
