# Architecture Decision Records (ADRs)

This document tracks significant architectural decisions made during the development of the F1 Prediction System.

## ADR Format

Each decision record includes:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Date**: When the decision was made
- **Context**: Problem being addressed
- **Decision**: What was decided
- **Consequences**: Trade-offs and implications

---

## ADR-001: Pydantic Models for Data Validation

**Status**: Accepted
**Date**: 2024-01-15

### Context

The Ergast F1 API returns JSON responses with inconsistent field naming (camelCase vs snake_case) and optional fields. We need:
- Type-safe data structures
- Automatic validation of API responses
- Easy serialization/deserialization
- Good IDE support with autocomplete

### Decision

Use Pydantic v2 models for all data entities (Race, Driver, Constructor, Circuit, etc.) with field aliases to handle API naming conventions.

### Implementation

```python
from pydantic import BaseModel, Field

class Driver(BaseModel):
    driver_id: str = Field(..., alias="driverId")
    given_name: str = Field(..., alias="givenName")
    family_name: str = Field(..., alias="familyName")

    class Config:
        populate_by_name = True  # Allow both alias and field name
```

### Consequences

**Positive**:
- ✅ Automatic validation of API responses
- ✅ Type hints improve IDE support and catch errors early
- ✅ JSON serialization out of the box
- ✅ Clear documentation through model definitions
- ✅ Easy to extend with custom validators

**Negative**:
- ❌ Slight performance overhead vs raw dicts (~10-15%)
- ❌ Learning curve for developers unfamiliar with Pydantic
- ❌ Model definitions require maintenance as API evolves

**Mitigation**:
- Performance impact negligible for our use case (<100 req/min)
- Comprehensive documentation and examples provided
- Automated tests alert us to API schema changes

---

## ADR-002: Ensemble Model Architecture

**Status**: Accepted
**Date**: 2024-02-20

### Context

Single ML models have limitations:
- Rule-based models lack sophistication
- Logistic regression struggles with non-linear patterns
- Tree-based models can overfit
- No single model performs best across all scenarios

Need to combine strengths of multiple models for better predictions.

### Decision

Implement `EnsemblePredictor` that combines multiple models using weighted voting:
- Rule-Based: 10% weight (baseline sanity check)
- Logistic Regression: 15% weight (linear patterns)
- Random Forest: 20% weight (non-linear, robust)
- XGBoost: 25% weight (high accuracy)
- LightGBM: 30% weight (best performance + speed)

### Implementation

```python
class EnsemblePredictor:
    def __init__(self, voting='soft'):
        self.models = []
        self.weights = []

    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, features):
        predictions = [m.predict_proba(features) for m in self.models]
        return np.average(predictions, axis=0, weights=self.weights)
```

### Consequences

**Positive**:
- ✅ Improved prediction accuracy (70% → 87% on podium predictions)
- ✅ Robust to individual model weaknesses
- ✅ Confidence estimates from soft voting
- ✅ Can add/remove models without changing architecture

**Negative**:
- ❌ Increased training time (5x slower than single model)
- ❌ Higher memory usage (all models loaded)
- ❌ More complex model management and versioning
- ❌ Harder to interpret feature importance

**Mitigation**:
- Training is offline, so 2-minute train time acceptable
- 16GB RAM sufficient for all models (<100MB total)
- Save ensemble as single pickle file
- Provide feature importance from Random Forest component

---

## ADR-003: Streamlit for Web Interface

**Status**: Accepted
**Date**: 2024-03-01

### Context

Need web interface for:
- Non-technical users to make predictions
- Visualize race analysis
- Interactive parameter tuning
- Real-time prediction updates

Options considered:
1. **React + FastAPI**: Full control, modern stack, requires frontend expertise
2. **Flask + Jinja**: Python-only, limited interactivity
3. **Streamlit**: Python-only, rapid development, built-in widgets

### Decision

Use Streamlit for the web interface due to rapid prototyping capability and Python-only development.

### Implementation

```python
import streamlit as st

st.title("F1 Race Predictor")
circuit = st.selectbox("Circuit", circuits)
model_type = st.radio("Model", ["Random Forest", "Ensemble"])

if st.button("Predict"):
    predictions = make_prediction(circuit, model_type)
    st.dataframe(predictions)
```

### Consequences

**Positive**:
- ✅ Rapid development (MVP in 2 days vs 2 weeks)
- ✅ Python-only, no frontend expertise required
- ✅ Built-in state management
- ✅ Auto-refresh on code changes
- ✅ Easy deployment (Streamlit Cloud, Docker)

**Negative**:
- ❌ Limited customization of UI components
- ❌ Less control over page layout vs traditional frontend
- ❌ State management can be confusing with reruns
- ❌ Not ideal for complex, multi-page applications

**Mitigation**:
- Accept UI limitations for v1.0
- Plan migration to React if complexity grows
- Use st.session_state for complex state management
- Document Streamlit patterns for team

---

## ADR-004: uv for Package Management

**Status**: Accepted
**Date**: 2024-03-10

### Context

Python package management options:
1. **pip + venv**: Standard, slow, basic dependency resolution
2. **Poetry**: Modern, pyproject.toml, slow lock file generation
3. **uv**: Ultra-fast (10-100x), pip-compatible, modern

Need fast, reliable package management for development and CI/CD.

### Decision

Use `uv` as primary package manager with pip fallback for compatibility.

### Implementation

```bash
# Install dependencies
uv sync --dev

# Add package
uv add pandas

# Run command
uv run pytest
```

### Consequences

**Positive**:
- ✅ 10-100x faster than pip (2s vs 30s install)
- ✅ Better dependency resolution than pip
- ✅ Compatible with pip/pyproject.toml
- ✅ Built-in virtual environment management
- ✅ Lockfile for reproducible installs

**Negative**:
- ❌ Newer tool, smaller community
- ❌ Potential breaking changes in updates
- ❌ Not available in all CI environments by default

**Mitigation**:
- Provide pip install instructions as fallback
- Pin uv version in CI workflows
- Document both uv and pip workflows

---

## ADR-005: Click for Enhanced CLI

**Status**: Accepted
**Date**: 2024-03-15

### Context

Current CLI using argparse is:
- Verbose and boilerplate-heavy
- Limited composability
- No built-in validation
- Poor help text formatting

Need more powerful CLI framework for growing command set.

### Decision

Migrate CLI from argparse to Click framework with Rich for formatting.

### Implementation

```python
import click
from rich.console import Console

@click.group()
def cli():
    """F1 Predict - Machine learning for Formula 1 predictions."""
    pass

@cli.command()
@click.option('--type', type=click.Choice(['all', 'race-results']))
@click.option('--enrich/--no-enrich', default=False)
def collect(type, enrich):
    """Collect historical F1 data."""
    console = Console()
    console.print(f"[green]Collecting {type} data...[/green]")
```

### Consequences

**Positive**:
- ✅ Cleaner, more maintainable code
- ✅ Automatic help generation
- ✅ Built-in type validation
- ✅ Command grouping and nesting
- ✅ Rich formatting for beautiful output

**Negative**:
- ❌ Additional dependency (Click + Rich)
- ❌ Migration effort for existing commands
- ❌ Decorator-based style may be unfamiliar

**Mitigation**:
- Click is industry standard (Flask uses it)
- Incremental migration command-by-command
- Examples in documentation

---

## ADR-006: Temporal Data Splitting

**Status**: Accepted
**Date**: 2024-02-25

### Context

ML model evaluation requires train/test split. For time-series F1 data:
- Random split causes data leakage (future → past)
- Models would have unrealistically high accuracy
- Predictions wouldn't generalize to future races

### Decision

Always use temporal split: train on older data, test on more recent data.

### Implementation

```python
# ✅ Correct: Temporal split
train = data[data['season'] < 2024]
test = data[data['season'] == 2024]

# ❌ Wrong: Random split (DO NOT USE)
train, test = train_test_split(data, test_size=0.2)
```

### Consequences

**Positive**:
- ✅ Realistic accuracy estimates
- ✅ Prevents data leakage
- ✅ Models generalize to future races
- ✅ Mimics production scenario

**Negative**:
- ❌ Lower reported accuracy vs random split
- ❌ Requires more historical data
- ❌ Seasonal effects may impact results

**Mitigation**:
- Accept lower but honest accuracy
- Collect 5+ years of data
- Use TimeSeriesSplit for cross-validation

---

## ADR-007: External Data Integration (Weather, Tracks, Tires)

**Status**: Accepted
**Date**: 2024-03-20

### Context

Race outcomes affected by factors beyond historical results:
- Weather conditions (rain dramatically changes outcomes)
- Track characteristics (Monaco vs Monza very different)
- Tire strategies (compound choice affects performance)

Need to integrate external data sources for better predictions.

### Decision

Implement optional external data enrichment:
- OpenWeatherMap API for weather data
- JSON database for track characteristics (20 circuits)
- Tire strategy parsing from historical data

### Implementation

```python
# Collect with enrichment
collector = F1DataCollector()
collector.collect_and_clean_all_data(enable_enrichment=True)

# Or enrich separately
collector.enrich_collected_data()
```

### Consequences

**Positive**:
- ✅ More comprehensive feature set
- ✅ Weather improves wet race predictions
- ✅ Track data helps circuit-specific predictions
- ✅ Tire data for strategy analysis

**Negative**:
- ❌ Requires API key (free tier available)
- ❌ Additional API calls slow down collection
- ❌ Weather data may not be available for all races
- ❌ Increased complexity

**Mitigation**:
- Make enrichment optional (default: off)
- Graceful fallback if weather API unavailable
- Cache weather data to reduce API calls
- Document free tier limitations

---

## ADR-008: LLM Integration for Analysis Generation

**Status**: Accepted
**Date**: 2024-03-25

### Context

Users want natural language explanations of predictions, not just numbers. Need to:
- Explain why a driver is predicted to podium
- Analyze race strategy implications
- Generate human-readable race analysis

### Decision

Integrate LLM (Anthropic Claude, OpenAI, Ollama) for:
- Natural language race analysis
- Strategy explanation
- Chat interface for questions

### Implementation

```python
from f1_predict.llm.providers import AnthropicProvider

provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
analysis = provider.generate_race_analysis(
    predictions=predictions,
    historical_data=data,
    circuit="monaco"
)
```

### Consequences

**Positive**:
- ✅ Human-readable explanations
- ✅ Engaging user experience
- ✅ Can answer follow-up questions
- ✅ Combines multiple data sources

**Negative**:
- ❌ Requires API key and costs money
- ❌ Latency (3-5s per analysis)
- ❌ Non-deterministic output
- ❌ Privacy concerns with data sharing

**Mitigation**:
- Support free local models (Ollama)
- Cache common analyses
- Allow users to disable LLM features
- Document privacy implications

---

## ADR-009: Ruff for Linting and Formatting

**Status**: Accepted
**Date**: 2024-01-20

### Context

Need consistent code style across codebase. Traditional Python tools:
- **Black**: Formatting only
- **isort**: Import sorting
- **Flake8**: Linting
- **pylint**: Comprehensive linting

Managing 4+ tools is cumbersome. Ruff is a unified, fast alternative.

### Decision

Replace Black, isort, Flake8, and pylint with Ruff for all linting and formatting.

### Implementation

```toml
[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "A"]
ignore = ["E501"]  # Line too long (handled by formatter)
```

### Consequences

**Positive**:
- ✅ 10-100x faster than existing tools
- ✅ Single tool vs 4+ tools
- ✅ Auto-fix for most issues
- ✅ Compatible with Black formatting
- ✅ Active development and updates

**Negative**:
- ❌ Newer tool, fewer rules than pylint
- ❌ Some team members unfamiliar with Ruff
- ❌ Potential breaking changes

**Mitigation**:
- Ruff covers 90%+ of common issues
- Extensive documentation provided
- Pin Ruff version for stability

---

## Decision Review Process

ADRs should be reviewed and updated:
- **Quarterly**: Review all Accepted ADRs for relevance
- **On major changes**: Update or supersede ADRs as needed
- **On new decisions**: Create new ADR using template

### ADR Template

```markdown
## ADR-XXX: Title

**Status**: Proposed
**Date**: YYYY-MM-DD

### Context
[Problem description]

### Decision
[What was decided]

### Implementation
[Code examples]

### Consequences
**Positive**:
- ✅ Benefit 1
- ✅ Benefit 2

**Negative**:
- ❌ Trade-off 1
- ❌ Trade-off 2

**Mitigation**:
- How we address trade-offs
```

## References

- [Architecture Overview](overview.md)
- [CONTRIBUTING.md](../../CONTRIBUTING.md)
- [ADR GitHub Repo](https://adr.github.io/)
