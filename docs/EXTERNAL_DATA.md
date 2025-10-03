# External Data Sources Integration

This document describes the external data sources integrated into the F1 prediction system to enhance model accuracy by incorporating weather conditions, track characteristics, and tire strategy data.

## Overview

The external data integration enriches race predictions with critical factors that significantly impact race outcomes but are not captured in basic race results data.

**Data Sources Integrated:**
- ✅ Weather Data (OpenWeatherMap API)
- ✅ Track Characteristics (Manual Database)
- ✅ Tire Strategy Data (Parser/Collector)
- ⏳ Real-time Weather Forecasts (Requires API subscription)
- ⏳ Historical Weather Patterns (Requires API subscription)

## Weather Data Integration

### OpenWeatherMap API

**Purpose**: Provides historical weather conditions and forecasts for race weekends.

**Setup**:
```bash
# Set API key environment variable
export OPENWEATHER_API_KEY="your_api_key_here"

# Or add to .env file
echo "OPENWEATHER_API_KEY=your_api_key_here" >> .env
```

**API Tier Information**:
- **Free Tier**: Current weather, 5-day forecast (60 calls/minute)
- **Paid Tier**: Historical weather data, extended forecasts
- **Cost**: Historical data requires subscription ($40+/month)

**Data Collected**:
- Air temperature (°C)
- Track temperature (°C) - when available
- Weather condition (clear, rain, cloudy, etc.)
- Precipitation amount (mm)
- Humidity (%)
- Wind speed (km/h) and direction
- Atmospheric pressure (hPa)

**Weather Conditions Mapped**:
- `CLEAR`: Clear sky
- `PARTLY_CLOUDY`: Few clouds
- `CLOUDY`: Scattered/broken clouds
- `OVERCAST`: Overcast sky
- `LIGHT_RAIN`: Drizzle or light rain
- `RAIN`: Moderate rain
- `HEAVY_RAIN`: Heavy rain
- `THUNDERSTORM`: Thunderstorm conditions
- `FOG`: Mist, fog, or haze

### Usage Example

```python
from f1_predict.data.weather_collector import WeatherDataCollector

# Initialize collector (requires API key in environment)
collector = WeatherDataCollector(data_dir="data")

# Collect race weather
weather = collector.collect_race_weather(
    circuit_id="bahrain",
    season="2024",
    round_num="1",
    lat=26.0325,
    lon=50.5106,
    race_date=datetime(2024, 3, 2, 15, 0)
)

print(f"Temperature: {weather.air_temperature}°C")
print(f"Condition: {weather.condition.value}")
print(f"Humidity: {weather.humidity}%")
```

### CLI Integration

```bash
# Collect data with weather enrichment (requires API key)
f1-predict collect --type all --enrich

# Check enrichment results
cat data/processed/enriched_race_data.json
```

## Track Characteristics Database

### Manual Database

**Purpose**: Provides physical and technical characteristics of F1 circuits.

**Data File**: `data/external/track_characteristics.json`

**Circuits Covered**: 20 circuits from 2020-2024 calendar
- Bahrain International Circuit
- Jeddah Corniche Circuit
- Albert Park Circuit
- Autodromo Enzo e Dino Ferrari (Imola)
- Miami International Autodrome
- Circuit de Monaco
- Circuit de Barcelona-Catalunya
- Red Bull Ring
- Silverstone Circuit
- Hungaroring
- Circuit de Spa-Francorchamps
- Circuit Zandvoort
- Autodromo Nazionale di Monza
- Marina Bay Street Circuit
- Suzuka International Racing Course
- Losail International Circuit
- Circuit of the Americas
- Autódromo Hermanos Rodríguez
- Autódromo José Carlos Pace (Interlagos)
- Yas Marina Circuit

**Track Data Attributes**:
- **Layout**: Length (km), corners, DRS zones
- **Type**: Street, permanent, semi-permanent
- **Downforce Level**: Low, medium, high, very high
- **Overtaking Difficulty**: 1-10 scale
- **Surface**: Roughness, asphalt age, grip level
- **Safety**: Average safety car probability
- **Performance**: Average lap time, top speed
- **Stress Levels**: Power unit, brakes, tires (1-10 scale)

### Usage Example

```python
from f1_predict.data.track_data import TrackDataManager

# Load track database
track_manager = TrackDataManager()

# Get specific track
monaco = track_manager.get_track("monaco")
print(f"Circuit: {monaco.circuit_name}")
print(f"Downforce: {monaco.downforce_level.value}")
print(f"Overtaking Difficulty: {monaco.overtaking_difficulty}/10")

# Filter tracks by characteristics
high_downforce_tracks = track_manager.get_high_downforce_tracks()
difficult_overtaking = track_manager.get_difficult_overtaking_tracks(threshold=7)
```

## Tire Strategy Data

### Tire Data Parser

**Purpose**: Extracts tire compound usage, pit stop strategies, and degradation patterns from race data.

**Data Collected**:
- **Tire Stints**: Compound, lap range, performance, degradation
- **Pit Strategies**: Stop count, timing, durations, compound sequence
- **Degradation Rates**: Lap time increase per lap on each compound

**Tire Compounds**:
- Dry: Soft (Red), Medium (Yellow), Hard (White)
- Wet: Intermediate (Green), Wet (Blue)
- Historical: Hypersoft, Ultrasoft, Supersoft, Superhard

### Usage Example

```python
from f1_predict.data.tire_data import TireDataCollector

# Initialize collector
tire_collector = TireDataCollector(data_dir="data")

# Parse tire compound
compound = tire_collector.parse_tire_compound("soft")  # TireCompound.SOFT

# Create stint data
stint = tire_collector.create_tire_stint(
    season="2024",
    round_num="1",
    driver_id="max_verstappen",
    compound=TireCompound.SOFT,
    stint_number=1,
    starting_lap=1,
    ending_lap=15,
    lap_times=[92.1, 92.3, 92.5, 92.6, 92.8],  # seconds
    stint_end_reason="pit_stop"
)

print(f"Degradation: {stint.degradation_rate:.3f} sec/lap")
```

## Data Enrichment Pipeline

### Integration with Data Collector

The enrichment pipeline automatically integrates external data sources with collected race data.

**Architecture**:
```
Race Data Collection
    ↓
External Data Enrichment
    ├─ Track Characteristics (automatic)
    ├─ Weather Data (if API key available)
    └─ Tire Strategy (when pit stop data available)
    ↓
Enriched Race Data
```

### Enrichment Coverage

**Data Completeness Score**: 0.0 to 1.0 indicating % of external data available

```python
from f1_predict.data.collector import F1DataCollector

# Collect and enrich all data
collector = F1DataCollector(data_dir="data")
results = collector.collect_and_clean_all_data(
    force_refresh=False,
    enable_cleaning=True,
    enable_enrichment=True
)

# Check enrichment results
enrichment = results["enrichment"]
print(f"Enriched {enrichment['enriched_count']} races")
print(f"Coverage: {enrichment['coverage']:.1%}")
```

### Output Format

Enriched data is saved to `data/processed/enriched_race_data.json`:

```json
{
  "season": "2024",
  "round": "1",
  "circuit_id": "bahrain",
  "track_characteristics": {
    "circuit_name": "Bahrain International Circuit",
    "length_km": 5.412,
    "downforce_level": "medium",
    "overtaking_difficulty": 4,
    "average_safety_car_probability": 0.35
  },
  "race_weather": {
    "air_temperature": 28.5,
    "condition": "clear",
    "humidity": 35.0,
    "wind_speed": 12.5
  },
  "data_completeness_score": 0.85
}
```

## Model Integration

### Feature Engineering

External data can be incorporated into prediction models as features:

**Weather Features**:
- Temperature differential (air vs track)
- Rain probability flag
- Wind speed categories
- Humidity levels

**Track Features**:
- Downforce level encoding
- Overtaking difficulty
- Safety car probability
- Power unit/brake/tire stress

**Tire Features**:
- Compound hardness encoding
- Historical degradation rates
- Optimal strategy patterns
- Pit window timing

### Example Usage

```python
from f1_predict.data.external_models import EnrichedRaceData
import pandas as pd

# Load enriched data
with open("data/processed/enriched_race_data.json") as f:
    enriched_races = json.load(f)

# Convert to features for ML model
features = []
for race in enriched_races:
    track = race["track_characteristics"]

    features.append({
        "circuit_id": race["circuit_id"],
        "downforce": track["downforce_level"],
        "overtaking_difficulty": track["overtaking_difficulty"],
        "safety_car_prob": track["average_safety_car_probability"],
        "power_stress": track["power_unit_stress"],
        # Add more features...
    })

df = pd.DataFrame(features)
```

## Data Quality and Validation

### Quality Thresholds

- **Weather Data**: >95% coverage for race weekends
- **Track Characteristics**: 100% coverage for circuits in database
- **Tire Data**: >90% coverage where pit stop data available
- **Overall Enrichment**: Target ≥85% data completeness

### Fallback Strategies

**Weather API Unavailable**:
- Uses historical weather patterns for circuit
- Default reasonable weather conditions
- Logs missing data for analysis

**Track Data Missing**:
- Uses average values from similar circuit types
- Logs circuits needing database updates

**Tire Data Incomplete**:
- Estimates based on compound and circuit
- Uses historical averages for degradation

## Limitations and Future Work

### Current Limitations

1. **Historical Weather**: Requires paid OpenWeatherMap subscription
2. **Track Database**: Manual updates needed for new circuits or changes
3. **Tire Data**: Limited by availability of detailed pit stop data
4. **Real-time Integration**: Not yet implemented for live predictions

### Planned Enhancements

- [ ] Real-time race weekend weather monitoring
- [ ] Automated track database updates from FIA sources
- [ ] Pirelli official tire performance data integration
- [ ] Circuit evolution tracking (grip levels over sessions)
- [ ] Driver contract status and team dynamics
- [ ] Power unit penalty grid tracking

## Cost Considerations

**Free Tier (Current)**:
- OpenWeatherMap current weather: Free
- Track characteristics: No cost (manual database)
- Tire data: No cost (parsed from existing data)

**Paid Tier (Optional)**:
- Historical weather data: $40-200/month (OpenWeatherMap)
- Extended forecasts: Included in historical plan
- Alternative providers: Weather Underground, ClimaCell

**Recommendation**: Start with free tier and track characteristics. Evaluate paid weather API based on model performance improvement.

## Support and Troubleshooting

### Common Issues

**"Weather API not configured"**:
```bash
# Set API key
export OPENWEATHER_API_KEY="your_key"

# Verify it's set
echo $OPENWEATHER_API_KEY
```

**"Track characteristics not found"**:
- Ensure `data/external/track_characteristics.json` exists
- Verify circuit_id matches database entries
- Check track database for new circuit additions

**"Enrichment failed"**:
- Check logs for specific error messages
- Verify all required files exist
- Ensure data has been collected first (schedules required)

### Support Channels

- **Issues**: https://github.com/your-repo/f1-predict/issues
- **Documentation**: https://github.com/your-repo/f1-predict/docs
- **Discussions**: https://github.com/your-repo/f1-predict/discussions
