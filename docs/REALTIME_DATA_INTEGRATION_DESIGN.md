# Real-Time Data Integration & Live Predictions (Issue #31)

## Executive Summary

Implement real-time F1 data integration and live prediction capabilities to enable dynamic predictions during race weekends. This enables the system to provide predictions pre-race, during qualifying, and during the race itself.

## Architecture Overview

```
┌─────────────────────┐
│   External APIs     │
├─────────────────────┤
│ • OpenF1 API        │
│ • F1 Live Timing    │
│ • Weather APIs      │
│ • Ergast (schedule) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────┐
│  Real-Time Data Client      │
├─────────────────────────────┤
│ • Session monitoring        │
│ • Rate limiting            │
│ • Data validation          │
│ • Streaming handler        │
└──────────┬──────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│   Real-Time Data Pipeline            │
├──────────────────────────────────────┤
│ • Data ingestion & buffering         │
│ • Session state tracking             │
│ • Data enrichment (weather, gaps)    │
│ • Caching layer (Redis optional)     │
│ • Time-series storage (optional)     │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│   Live Prediction Engine             │
├──────────────────────────────────────┤
│ • Pre-race predictions               │
│ • Qualifying predictions             │
│ • Mid-race updates                   │
│ • Confidence interval calculation    │
│ • Accuracy tracking                  │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│   Output/Display Layer               │
├──────────────────────────────────────┤
│ • Web interface updates              │
│ • WebSocket push updates             │
│ • Historical logging                 │
│ • Accuracy metrics                   │
└──────────────────────────────────────┘
```

## Detailed Component Design

### 1. Real-Time API Client

**File**: `src/f1_predict/api/realtime.py`

```python
class RealtimeF1APIClient:
    """Handles real-time F1 data from OpenF1 or live timing APIs."""

    async def get_current_session(self) -> SessionData:
        """Get current session status (practice, qualifying, race)."""
        pass

    async def get_live_positions(self) -> List[DriverPosition]:
        """Get current race/session positions."""
        pass

    async def stream_session_data(self) -> AsyncIterator[SessionUpdate]:
        """Stream live data as it becomes available."""
        pass

    async def get_session_weather(self) -> WeatherData:
        """Get current weather at circuit."""
        pass

    async def get_tire_strategy(self) -> Dict[str, TireInfo]:
        """Get current tire information for drivers."""
        pass

class SessionData:
    """Current session status."""
    session_type: str  # 'FP1', 'FP2', 'FP3', 'Qualifying', 'Race'
    status: str  # 'scheduled', 'ongoing', 'completed'
    timestamp: datetime
    lap_count: Optional[int]  # for race
    circuit: str

class DriverPosition:
    """Current position of a driver."""
    driver_id: str
    position: int
    gap_to_leader: float  # seconds
    latest_lap_time: Optional[float]  # seconds
    tires: Optional[str]  # compound
    laps_on_tires: Optional[int]
    pit_stop_count: Optional[int]
```

### 2. Real-Time Data Pipeline

**File**: `src/f1_predict/data/realtime_pipeline.py`

```python
class RealtimeDataPipeline:
    """Manages ingestion and processing of real-time F1 data."""

    async def start_session_monitoring(self, circuit: str) -> SessionMonitor:
        """Start monitoring a specific session."""
        pass

    async def process_session_update(self, update: SessionUpdate) -> EnrichedUpdate:
        """Process and enrich incoming data."""
        pass

    async def get_current_state(self) -> RaceState:
        """Get current race state for predictions."""
        pass

    async def stop_monitoring(self):
        """Stop monitoring and cleanup."""
        pass

class RaceState:
    """Complete state for live prediction."""
    session_type: str
    lap_count: int
    leader: DriverInfo
    positions: List[DriverPosition]
    weather: WeatherData
    tire_strategies: Dict[str, TireStrategy]
    red_flags: int
    safety_car_active: bool
    weather_changes: List[WeatherChange]
    timestamp: datetime

class SessionMonitor:
    """Monitors a single race session."""
    circuit: str
    session_type: str
    start_time: datetime
    updates_received: int

    async def wait_for_update(self, timeout: int = 10) -> SessionUpdate:
        """Wait for next data update."""
        pass
```

### 3. Live Prediction Engine

**File**: `src/f1_predict/predictions/live_predictor.py`

```python
class LivePredictor:
    """Generates live predictions during race sessions."""

    def __init__(self, model: EnsemblePredictor, analyzer: PerformanceAnalyzer):
        self.model = model
        self.analyzer = analyzer
        self.prediction_history = []

    async def predict_pre_race(self, qualifying_results: List[QualifyingResult]) -> RacePrediction:
        """Generate prediction before race starts."""
        # Use qualifying positions, historical data, weather
        pass

    async def predict_during_race(self, race_state: RaceState) -> LiveRacePrediction:
        """Update prediction based on live race data."""
        # Account for current positions, tire strategy, weather changes
        pass

    async def calculate_confidence_intervals(self, prediction: RacePrediction) -> ConfidenceInterval:
        """Calculate confidence bounds around prediction."""
        pass

    def track_accuracy(self, predicted: RacePrediction, actual: RaceResult):
        """Log prediction vs actual for accuracy analysis."""
        pass

class RacePrediction:
    """Prediction for entire race outcome."""
    predicted_winner: str
    top_3: List[str]
    points_distribution: Dict[str, float]
    confidence: float
    methodology: str
    timestamp: datetime

class LiveRacePrediction(RacePrediction):
    """Live prediction with additional context."""
    current_leader: str
    next_likely_overtake: Optional[OvertakeChance]
    dnf_risk: Dict[str, float]
    strategy_impact: Dict[str, StrategyImpact]
    confidence_trend: str  # 'improving', 'stable', 'declining'
```

### 4. WebSocket/Streaming Interface

**File**: `src/f1_predict/web/utils/realtime_websocket.py`

```python
class RealtimeWebSocketHandler:
    """Manages WebSocket connections for live data push."""

    async def handle_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        pass

    async def broadcast_prediction_update(self, prediction: LiveRacePrediction):
        """Send prediction update to all connected clients."""
        pass

    async def broadcast_race_event(self, event: RaceEvent):
        """Notify clients of significant race events."""
        pass

class RealtimeStreamingServer:
    """Server that streams live predictions."""

    async def start_streaming(self, session: SessionData):
        """Start streaming predictions for session."""
        pass

    async def stream_predictions(self) -> AsyncIterator[PredictionUpdate]:
        """Stream updates as they become available."""
        pass
```

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Design and implement `RealtimeF1APIClient`
- [ ] Create session data models
- [ ] Implement basic session monitoring
- [ ] Add rate limiting and error handling
- [ ] Write unit tests (20+ tests)

### Phase 2: Data Pipeline (Week 2)
- [ ] Implement `RealtimeDataPipeline`
- [ ] Create data enrichment logic
- [ ] Build state management system
- [ ] Add data validation and cleaning
- [ ] Write integration tests (15+ tests)

### Phase 3: Live Predictions (Week 2-3)
- [ ] Implement `LivePredictor`
- [ ] Create pre-race prediction logic
- [ ] Build mid-race prediction updates
- [ ] Implement confidence interval calculation
- [ ] Add accuracy tracking
- [ ] Write prediction tests (20+ tests)

### Phase 4: Streaming & Integration (Week 3-4)
- [ ] Build WebSocket server
- [ ] Create streaming endpoints
- [ ] Integrate with web interface
- [ ] Add dashboard updates
- [ ] Performance optimization
- [ ] End-to-end testing (10+ tests)

## Data Source Options

### Option A: OpenF1 API (Recommended)
- **Pros**: Free, comprehensive, well-documented
- **Cons**: May have rate limits
- **Status**: Available and active
- **Data**: Session data, positions, timings, weather

### Option B: F1 Live Timing API
- **Pros**: Official F1 data
- **Cons**: Rate limited, may require authentication
- **Status**: Official source
- **Data**: Real-time positioning, lap times, pit stops

### Option C: Hybrid Approach
- Combine OpenF1 for general data
- F1 Live Timing for high-frequency updates
- Ergast for historical context

## API Endpoints

### Real-Time Endpoints
```
GET  /api/v1/session/current         - Current session status
GET  /api/v1/session/{circuit}/live  - Live session data
WS   /ws/live-predictions             - WebSocket for predictions
GET  /api/v1/prediction/live          - Current prediction
POST /api/v1/prediction/subscribe     - Subscribe to updates
```

## Performance Requirements

| Component | Target | Acceptable |
|-----------|--------|-----------|
| API response | <1s | <2s |
| Data pipeline | <3s | <5s |
| Prediction update | <5s | <10s |
| WebSocket push | <100ms | <500ms |
| Database query | <500ms | <2s |

## Error Handling & Resilience

```python
class RealtimeErrorHandler:
    """Handles errors in real-time pipeline."""

    async def handle_api_timeout(self):
        """Fallback when API is slow."""
        # Use cached data or historical averages
        pass

    async def handle_data_gap(self, duration: float):
        """Handle missing data."""
        # Interpolate or use last known state
        pass

    async def handle_prediction_failure(self):
        """Handle prediction calculation error."""
        # Return previous prediction with confidence reduced
        pass

    async def handle_connection_loss(self):
        """Handle lost API connection."""
        # Switch to alternative source or queue updates
        pass
```

## Testing Strategy

### Unit Tests
- Individual component functionality
- Data model validation
- Error handling paths
- Rate limiting logic

### Integration Tests
- API client integration
- Pipeline end-to-end flow
- Prediction generation
- WebSocket communication

### Performance Tests
- Latency under load
- Throughput capacity
- Memory usage
- Database performance

### Simulation Tests
- Mock race scenarios
- Edge cases (red flags, DNF, etc.)
- Weather changes
- Tire degradation scenarios

## Success Criteria

1. **Data Timeliness**: <5 second update latency
2. **Prediction Accuracy**: >75% accuracy on top 3 predictions
3. **System Reliability**: >99% uptime during race sessions
4. **User Experience**: Sub-second WebSocket updates
5. **Scalability**: Support 100+ concurrent connections
6. **Coverage**: 80%+ code coverage with tests

## Dependencies

### New Libraries Required
- `aiohttp` - Async HTTP client (already have)
- `websockets` - WebSocket support
- `aioredis` - Optional caching
- `openf1` - F1 API client (if available)

### Infrastructure
- Real-time database (optional): InfluxDB, TimescaleDB
- Caching: Redis (optional)
- Message queue: RabbitMQ/Kafka (optional, for scaling)

## Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| API rate limiting | Can't get updates | High | Implement queue, caching, multiple sources |
| Network latency | Delays updates | Medium | WebSocket pooling, compression |
| Model stale during race | Bad predictions | Medium | Incremental updates, ensemble averaging |
| High concurrent users | Server overload | Medium | Horizontal scaling, connection limits |
| Data inconsistency | Wrong predictions | Low | Validation, checksums, audit logs |

## Future Enhancements

1. **Machine Learning**: Train models on live race data
2. **Predictive Analytics**: Forecast pit stops, DNF probability
3. **Comparative Analysis**: Compare predictions across models
4. **Historical Analysis**: Track prediction accuracy over seasons
5. **Mobile Push**: Send critical updates via mobile app
6. **Social Integration**: Share predictions on social media
7. **Betting Integration**: Link with fantasy F1 platforms
