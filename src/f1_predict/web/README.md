# F1 Race Predictor Web Interface

Modern, interactive web application for F1 race predictions built with Streamlit and Nebula UI design system.

## Features

### 🏠 Home Dashboard
- Quick stats and KPIs
- Upcoming race schedule
- Recent predictions history
- System status monitoring
- Quick action buttons

### 🏁 Race Prediction
- Select upcoming races
- Choose ML models (Ensemble, XGBoost, LightGBM, Random Forest)
- Generate predictions with confidence scores
- View predicted podium and full race order
- Explore feature importance and explanations
- *Coming soon: Full implementation with real predictions*

### 📊 Driver & Team Comparison
- Head-to-head driver comparisons
- Team performance analysis
- Filter by circuit or season
- Interactive charts (race results, qualifying, points trends)
- Circuit-specific statistics
- *Coming soon: Full implementation with real data*

### 📈 Analytics Dashboard
- Championship standings (drivers and constructors)
- Performance analysis charts
- Win rate by team visualization
- Reliability analysis
- Circuit performance heatmaps
- Auto-refresh capability
- *Coming soon: Full implementation with real analytics*

### 💬 Chat Interface
- Natural language F1 queries
- AI-powered conversational predictions
- Statistical analysis on demand
- Race insights and explanations
- Interactive charts and tables
- Suggested queries
- *Coming soon: Integration with LLM backend from Issue #11*

### ⚙️ Settings
- Theme configuration
- Timezone selection
- Unit preferences (metric/imperial)
- Default ML model selection
- Confidence threshold adjustment
- API configuration
- Cache settings
- *Coming soon: Full implementation with persistent settings*

## Installation

### Install Dependencies

```bash
# Install web interface dependencies
uv sync --extra web

# Or with pip
pip install -e ".[web]"
```

### Required Packages

- `streamlit>=1.31.0` - Web framework
- `streamlit-option-menu>=0.3.12` - Navigation menu
- `streamlit-autorefresh>=1.0.1` - Auto-refresh functionality

## Running the App

### Start the Web Interface

```bash
# From project root
streamlit run src/f1_predict/web/app.py

# Or navigate to web directory
cd src/f1_predict/web
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Configuration

Create a `.streamlit/config.toml` file in the project root for custom configuration:

```toml
[theme]
primaryColor = "#1F4E8C"
backgroundColor = "#121317"
secondaryBackgroundColor = "#1E2130"
textColor = "#E0E6F0"
font = "sans serif"

[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Architecture

### Directory Structure

```
src/f1_predict/web/
├── app.py                  # Main Streamlit application
├── pages/                  # Page components
│   ├── __init__.py        # Page exports
│   ├── home.py            # Home/dashboard
│   ├── predict.py         # Prediction interface
│   ├── compare.py         # Comparison tools
│   ├── analytics.py       # Analytics dashboard
│   ├── chat.py            # Chat interface
│   └── settings.py        # Settings page
├── components/            # Reusable UI components
│   ├── __init__.py
│   ├── charts.py         # Chart components
│   ├── tables.py         # Table components
│   ├── metrics.py        # Metric displays
│   └── forms.py          # Form components
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── theme.py          # Nebula UI theme
│   ├── state.py          # Session state
│   └── cache.py          # Caching utilities
└── README.md             # This file
```

### Design System: Nebula UI

The web interface follows the Nebula UI design system:

**Colors:**
- Background: `#121317` (very dark)
- Surface: `#1E2130` (dark charcoal)
- Primary Accent: `#1F4E8C` (dark blue)
- Text Primary: `#E0E6F0` (light gray)
- Text Secondary: `#A3A9BF` (muted gray)

**Principles:**
- Modern, futuristic, elegant
- Dark theme with moody aesthetics
- Compact layouts with minimal whitespace
- Sharp, angular components (2-4px border radius)
- Minimal, snappy interactions (100-200ms animations)

## Development

### Adding New Pages

1. Create a new file in `src/f1_predict/web/pages/`
2. Implement `show_<page>_page()` function
3. Add import to `pages/__init__.py`
4. Add navigation option in `app.py`

Example:

```python
# src/f1_predict/web/pages/my_page.py
import streamlit as st

def show_my_page() -> None:
    """Display my custom page."""
    st.title("My Page")
    st.write("Content goes here")
```

### Creating Reusable Components

Place shared components in `src/f1_predict/web/components/`:

```python
# src/f1_predict/web/components/charts.py
import plotly.graph_objects as go

def create_performance_chart(data):
    """Create a performance comparison chart."""
    fig = go.Figure()
    # Add chart logic
    return fig
```

### State Management

Use Streamlit's session state for persistent data:

```python
import streamlit as st

# Initialize state
if "my_data" not in st.session_state:
    st.session_state.my_data = {}

# Access state
data = st.session_state.my_data

# Update state
st.session_state.my_data = new_data
```

## Testing

### Run Web Component Tests

```bash
# Run all web tests
uv run pytest tests/web/ -v

# Run specific test file
uv run pytest tests/web/test_pages.py -v
```

### Manual Testing Checklist

- [ ] All pages load without errors
- [ ] Navigation works correctly
- [ ] Theme applies consistently
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Interactive elements respond within 200ms
- [ ] Loading states display correctly
- [ ] Error messages are clear and actionable

## Performance

### Optimization Tips

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Lazy Loading**: Load data only when needed
3. **Batch Operations**: Minimize API calls
4. **Session State**: Avoid unnecessary recomputations

Example:

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_race_data(race_id):
    """Load and cache race data."""
    # Expensive operation
    return data
```

### Performance Targets

- Initial page load: <3 seconds
- Prediction generation: <10 seconds
- Chart rendering: <2 seconds
- Chat response: <5 seconds
- Support: 50+ concurrent users

## Deployment

### Local Production Mode

```bash
# Run with production settings
streamlit run src/f1_predict/web/app.py --server.port=8080 --server.address=0.0.0.0
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[web]"

EXPOSE 8501

CMD ["streamlit", "run", "src/f1_predict/web/app.py"]
```

### Cloud Deployment

Streamlit Community Cloud:
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Configure secrets and settings
4. Deploy

## Roadmap

### Phase 1: Core Infrastructure (Current)
- ✅ Directory structure
- ✅ Nebula UI theme
- ✅ Navigation system
- ✅ Page stubs
- ⏳ Dependencies configuration

### Phase 2: Prediction Features
- ⏳ Race prediction integration
- ⏳ Model selection
- ⏳ Results visualization
- ⏳ Feature importance display

### Phase 3: Analytics & Comparisons
- ⏳ Driver comparison implementation
- ⏳ Team comparison implementation
- ⏳ Analytics dashboard
- ⏳ Interactive charts

### Phase 4: Advanced Features
- ⏳ Chat interface (requires Issue #11)
- ⏳ Settings persistence
- ⏳ User preferences
- ⏳ Real-time updates

### Phase 5: Polish & Optimization
- ⏳ Performance optimization
- ⏳ Mobile responsiveness
- ⏳ Accessibility (WCAG 2.1 AA)
- ⏳ Comprehensive testing

## Contributing

When contributing to the web interface:

1. Follow the Nebula UI design system
2. Maintain mobile responsiveness
3. Add unit tests for new components
4. Update this README with new features
5. Ensure accessibility compliance

## Dependencies & Integration

This web interface integrates with:

- **Issue #9**: ML models for predictions
- **Issue #11**: LLM API for chat
- **Issue #13**: Chat logic and conversation management
- **Issue #16**: Data visualization components

## License

MIT License - see LICENSE file for details
