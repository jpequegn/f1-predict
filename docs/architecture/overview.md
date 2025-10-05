# F1 Prediction System Architecture

## High-Level Architecture

The F1 Prediction System follows a layered architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Web UI       │  │ CLI          │  │ REST API     │       │
│  │ (Streamlit)  │  │ (Click)      │  │ (Future)     │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────┐
│         │        Application Layer            │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐      │
│  │ Prediction   │  │ Analysis     │  │ LLM Chat     │      │
│  │ Service      │  │ Service      │  │ Service      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────┐
│         │         Business Logic Layer        │              │
│  ┌──────▼───────────────────────────────────────────┐        │
│  │           Model Ensemble Orchestrator            │        │
│  └───┬──────────┬──────────┬──────────┬───────────┬─┘        │
│      │          │          │          │           │          │
│  ┌───▼───┐  ┌──▼───┐  ┌───▼───┐  ┌──▼────┐  ┌───▼───┐      │
│  │Rule-  │  │Logic │  │Random │  │XGBoost│  │Light  │      │
│  │Based  │  │Reg   │  │Forest │  │       │  │GBM    │      │
│  └───────┘  └──────┘  └───────┘  └───────┘  └───────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │          Feature Engineering Engine              │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘
          │
┌─────────┼──────────────────────────────────────────────────┐
│         │              Data Layer                          │
│  ┌──────▼───────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Data         │  │ Data         │  │ External     │     │
│  │ Collector    │  │ Cleaner      │  │ Data         │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │            │
│  ┌──────▼──────────────────▼──────────────────▼───────┐   │
│  │         API Client Layer (Ergast, Weather)          │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules

```
src/f1_predict/
├── api/                    # External API clients
│   ├── base.py            # Base API client with rate limiting
│   ├── ergast.py          # Ergast F1 API client
│   └── weather.py         # OpenWeatherMap API client
│
├── data/                   # Data layer
│   ├── models.py          # Pydantic data models
│   ├── collector.py       # Data collection orchestrator
│   ├── cleaning.py        # Data cleaning pipeline
│   ├── external_models.py # External data models (weather, tracks)
│   ├── weather_collector.py  # Weather data collection
│   ├── track_data.py      # Track characteristics
│   └── tire_data.py       # Tire strategy data
│
├── features/               # Feature engineering
│   └── engineering.py     # Feature creation pipeline
│
├── models/                 # ML models
│   ├── baseline.py        # Rule-based baseline
│   ├── logistic.py        # Logistic regression
│   ├── random_forest.py   # Random forest
│   ├── xgboost_model.py   # XGBoost
│   ├── lightgbm_model.py  # LightGBM
│   ├── ensemble.py        # Ensemble predictor
│   └── evaluation.py      # Model evaluation
│
├── metrics/                # Performance metrics
│   └── performance.py     # Metric calculations
│
├── strategy/               # Race strategy
│   └── predictor.py       # Strategy predictions
│
├── llm/                    # LLM integration
│   ├── base.py            # Base LLM client
│   ├── providers.py       # LLM provider implementations
│   └── chat_service.py    # Chat orchestration
│
├── analysis/               # Race analysis
│   └── generator.py       # Analysis generation
│
├── web/                    # Web interface
│   └── app.py             # Streamlit application
│
├── cli.py                  # Command-line interface
└── logging_config.py       # Logging configuration
```

## Data Flow

### Prediction Workflow

```
1. User Request (CLI/Web/API)
   ↓
2. Prediction Service
   ↓
3. Feature Engineering
   │  ├─ Load historical data
   │  ├─ Calculate rolling statistics
   │  ├─ Encode categorical features
   │  └─ Create interaction features
   ↓
4. Model Ensemble
   │  ├─ Rule-Based Predictor (weight: 0.1)
   │  ├─ Logistic Regression (weight: 0.15)
   │  ├─ Random Forest (weight: 0.2)
   │  ├─ XGBoost (weight: 0.25)
   │  └─ LightGBM (weight: 0.3)
   ↓
5. Weighted Voting
   ↓
6. Result Formatting
   ↓
7. Response to User
```

### Data Collection Workflow

```
1. CLI Command: f1-predict collect --type all
   ↓
2. F1DataCollector
   ↓
3. API Client Layer
   │  ├─ Ergast API (race results, qualifying)
   │  ├─ OpenWeatherMap API (weather data)
   │  └─ Track characteristics (JSON database)
   ↓
4. Data Validation (Pydantic models)
   ↓
5. Data Cleaning Pipeline
   │  ├─ Handle missing values
   │  ├─ Standardize names
   │  ├─ Validate data types
   │  └─ Quality scoring
   ↓
6. Storage (CSV + JSON)
   ↓
7. Success/Failure Report
```

## Key Design Patterns

### 1. Repository Pattern
Data access abstracted through collector classes:
```python
collector = F1DataCollector(data_dir="data")
results = collector.collect_race_results(seasons=[2024])
```

### 2. Strategy Pattern
Interchangeable prediction models:
```python
model: PredictorProtocol = RandomForestRacePredictor()
predictions = model.predict(features)
```

### 3. Factory Pattern
Model creation based on configuration:
```python
def create_model(model_type: str) -> PredictorProtocol:
    if model_type == "random_forest":
        return RandomForestRacePredictor()
    elif model_type == "xgboost":
        return XGBoostRacePredictor()
    # ...
```

### 4. Observer Pattern
Event-driven data updates:
```python
collector.on_data_collected(callback=process_new_data)
```

## Technology Stack

### Core Technologies
- **Python 3.9+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Pydantic**: Data validation and serialization
- **Scikit-learn**: Machine learning framework

### ML Libraries
- **XGBoost**: Gradient boosting
- **LightGBM**: Gradient boosting with categorical support
- **Scikit-learn**: Random Forest, Logistic Regression

### External APIs
- **Ergast F1 API**: Historical F1 data (free, rate-limited)
- **OpenWeatherMap API**: Weather data (free tier: 60 req/min)

### Development Tools
- **uv**: Fast Python package manager
- **Ruff**: Linting and formatting
- **MyPy**: Static type checking
- **Pytest**: Testing framework
- **Pre-commit**: Git hook management

### Web/CLI
- **Streamlit**: Web interface
- **Click**: CLI framework
- **Rich**: CLI formatting (planned)

## Performance Considerations

### Caching Strategy
```python
# Feature cache to avoid recomputation
@lru_cache(maxsize=128)
def calculate_driver_form(driver_id: str, date: datetime) -> float:
    # Expensive computation cached
```

### Rate Limiting
```python
# API client with built-in rate limiting
client = ErgastAPIClient(
    rate_limit_requests=4,  # 4 requests per second
    rate_limit_window=1.0
)
```

### Batch Processing
```python
# Process multiple predictions in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    predictions = executor.map(model.predict, feature_batches)
```

## Security Considerations

### API Key Management
```python
# Never commit API keys to version control
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not set")
```

### Input Validation
```python
# Pydantic models validate all external input
class RacePredictionRequest(BaseModel):
    season: int = Field(..., ge=2020, le=2030)
    circuit_id: str = Field(..., regex=r"^[a-z_]+$")
```

### Data Sanitization
```python
# Clean user input before SQL queries
driver_id = re.sub(r"[^a-z0-9_]", "", driver_id.lower())
```

## Scalability

### Current Limitations
- **Single-threaded prediction**: One prediction at a time
- **In-memory models**: All models loaded in RAM
- **File-based storage**: CSV files for data

### Future Improvements
- **Model serving**: Deploy models as microservices
- **Database**: PostgreSQL for persistent storage
- **Caching layer**: Redis for feature caching
- **Message queue**: Celery for async predictions
- **Containerization**: Docker for deployment

## Monitoring and Observability

### Logging
```python
# Structured logging with context
logger.info(
    "prediction_completed",
    extra={
        "model": "ensemble",
        "circuit": "monaco",
        "accuracy": 0.87,
        "duration_ms": 250
    }
)
```

### Metrics
- Prediction latency (p50, p95, p99)
- Model accuracy over time
- API error rates
- Data collection success rate

### Error Tracking
- Sentry integration (planned)
- Error aggregation and alerting
- Performance regression detection

## Testing Strategy

### Unit Tests
```python
# Test individual components in isolation
def test_feature_engineering():
    engineer = FeatureEngineer()
    features = engineer.create_basic_features(sample_data)
    assert "grid_position" in features.columns
```

### Integration Tests
```python
# Test end-to-end workflows
@pytest.mark.integration
def test_prediction_pipeline():
    collector = F1DataCollector()
    data = collector.collect_race_results([2024])
    model = RandomForestRacePredictor()
    predictions = model.fit(data).predict(data)
    assert len(predictions) > 0
```

### Test Coverage
- **Target**: 80% minimum coverage
- **Current**: 80%+ on core modules
- **Tools**: pytest-cov for coverage reporting

## Deployment

### Local Development
```bash
uv sync --dev
uv run f1-predict --help
```

### Production (Planned)
```bash
docker build -t f1-predict:latest .
docker run -p 8000:8000 f1-predict:latest
```

## References

- [Architecture Decision Records](decisions.md)
- [Data Schemas](../schemas/data_models.md)
- [API Reference](../api-reference.md)
- [Contributing Guide](../../CONTRIBUTING.md)
