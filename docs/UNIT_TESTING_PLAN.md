# Unit Testing Implementation Plan (Issue #17)

**Date**: 2025-11-09
**Status**: 📋 **PLANNING PHASE**
**Priority**: High
**Estimated Duration**: 3-4 weeks
**Coverage Target**: ≥80% overall, ≥90% for core modules

---

## Executive Summary

Issue #17 requires establishing comprehensive unit test coverage (≥80%) for all core modules. This document outlines the phased implementation strategy, covering:

1. Test infrastructure setup (pytest configuration, fixtures)
2. API client tests (BaseAPIClient, ErgastAPIClient)
3. Data collection tests (F1DataCollector, data validation)
4. Feature engineering tests (feature calculations)
5. Model prediction tests (all 6 model types)
6. Performance and metrics tests

---

## Current Test Status

### Existing Tests
- ✅ **models/test_random_forest.py**: 29 tests, ~90% coverage
- ✅ **models/test_xgboost.py**: 20 tests, ~86% coverage
- ✅ **models/test_lightgbm.py**: 15 tests, ~80% coverage
- ✅ **models/test_api/**: Multiple API tests (need verification)
- ✅ **Hyperparameter optimization tests**: 12 new integration tests

**Existing Coverage**: ~6% (from full test run)
**Target**: ≥80%

### Key Gaps
- API client tests (incomplete)
- Data collector tests (missing)
- Feature engineering tests (missing)
- CLI tests (missing)
- Integration workflow tests (missing)
- Performance metric tests (incomplete)

---

## Phase 1: Test Infrastructure Setup (Week 1)

### 1.1 Pytest Configuration
**File**: `pyproject.toml` (update existing)

**Current Configuration Status**: ✅ Already configured
- Minimum version: 7.4.0
- Coverage requirement: 80% fail-under
- Markers defined: slow, integration, unit, ml, data, api

**Required Actions**:
- [x] Verify pytest configuration
- [x] Confirm coverage reports generate (HTML, XML)
- [x] Verify test discovery works

### 1.2 Test Fixtures (conftest.py)

**Location**: `tests/conftest.py`
**Status**: Needs creation/expansion

**Required Fixtures**:
```python
# Data fixtures
@pytest.fixture
def sample_features() -> pd.DataFrame
    """20-driver sample with typical race features"""

@pytest.fixture
def sample_race_results() -> pd.DataFrame
    """Corresponding race results for training"""

@pytest.fixture
def sample_historical_data() -> dict
    """Multi-season historical data structure"""

# API fixtures
@pytest.fixture
def mock_http_response()
    """Mock HTTP 200 response"""

@pytest.fixture
def mock_api_error()
    """Mock HTTP 500 error response"""

# Model fixtures
@pytest.fixture
def trained_random_forest()
    """Fitted RandomForestRacePredictor"""

@pytest.fixture
def trained_xgboost()
    """Fitted XGBoostRacePredictor"""

# File fixtures
@pytest.fixture
def temp_model_dir(tmp_path)
    """Temporary directory for model files"""

@pytest.fixture
def temp_data_dir(tmp_path)
    """Temporary directory for data files"""
```

**Effort**: ~4 hours

---

## Phase 2: API Client Tests (Week 1-2)

### 2.1 BaseAPIClient Tests

**Location**: `tests/api/test_base.py`
**Current Status**: Partial - needs completion

**Test Categories** (12-15 tests):

1. **Initialization & Configuration** (2 tests)
   - [ ] Initialize with valid parameters
   - [ ] Raise error on invalid base URL

2. **Rate Limiting** (3 tests)
   - [ ] Enforce minimum delay between requests
   - [ ] Allow burst requests within rate limit
   - [ ] Accept custom rate limit values

3. **Error Handling & Retries** (4 tests)
   - [ ] Retry on transient 5xx errors
   - [ ] Exponential backoff on retries
   - [ ] Give up after max retries
   - [ ] Propagate non-retryable errors

4. **Response Parsing** (3 tests)
   - [ ] Parse JSON response correctly
   - [ ] Validate Pydantic models from response
   - [ ] Handle empty response gracefully

5. **Timeout & Connection** (2 tests)
   - [ ] Respect timeout configuration
   - [ ] Handle connection timeout errors

**Effort**: ~8 hours

### 2.2 ErgastAPIClient Tests

**Location**: `tests/api/test_ergast.py`
**Current Status**: Partial

**Test Categories** (10-12 tests):

1. **Initialization** (1 test)
   - [ ] Initialize with correct endpoints

2. **Race Data Fetching** (3 tests)
   - [ ] Fetch race results for given season/round
   - [ ] Handle missing race data gracefully
   - [ ] Parse driver/constructor data correctly

3. **Qualifying Data** (2 tests)
   - [ ] Fetch qualifying session data
   - [ ] Handle missing qualifying data

4. **Schedule Fetching** (2 tests)
   - [ ] Fetch race schedule for season
   - [ ] Parse dates and circuit information

5. **Error Scenarios** (2 tests)
   - [ ] Handle API errors gracefully
   - [ ] Retry on rate limit (429) responses

**Effort**: ~6 hours

### 2.3 Total Effort (Phase 2): ~14 hours

---

## Phase 3: Data Collection Tests (Week 2)

### 3.1 F1DataCollector Tests

**Location**: `tests/data/test_collector.py`
**Current Status**: Missing

**Test Categories** (15-18 tests):

1. **Initialization** (2 tests)
   - [ ] Initialize with valid data directory
   - [ ] Create directory structure automatically

2. **Race Results Collection** (4 tests)
   - [ ] Collect race results for specific season
   - [ ] Save to CSV and JSON formats
   - [ ] Force refresh existing data
   - [ ] Handle collection errors gracefully

3. **Qualifying Collection** (3 tests)
   - [ ] Collect qualifying session data
   - [ ] Parse grid positions correctly
   - [ ] Handle missing data

4. **Schedule Collection** (2 tests)
   - [ ] Fetch and save race schedule
   - [ ] Parse dates in correct format

5. **Full Pipeline** (3 tests)
   - [ ] Collect all data types successfully
   - [ ] Generate collection summary
   - [ ] Handle partial failures gracefully

6. **Data Validation** (2 tests)
   - [ ] Validate collected data schema
   - [ ] Check data for completeness

**Effort**: ~12 hours

### 3.2 Data Models Tests

**Location**: `tests/data/test_models.py`
**Current Status**: Partial

**Test Categories** (8-10 tests):

1. **Model Validation** (4 tests)
   - [ ] Validate Race model with correct data
   - [ ] Validate Driver model with aliases
   - [ ] Validate Constructor model
   - [ ] Validate Circuit model

2. **Error Handling** (2 tests)
   - [ ] Reject invalid data types
   - [ ] Handle missing required fields

3. **Serialization** (2 tests)
   - [ ] Convert model to dict
   - [ ] Serialize to JSON

**Effort**: ~6 hours

### 3.3 Data Cleaning Tests

**Location**: `tests/data/test_cleaning.py`
**Current Status**: Partial

**Test Categories** (8-10 tests):

1. **DataCleaner** (5 tests)
   - [ ] Clean missing values with defaults
   - [ ] Standardize driver names across seasons
   - [ ] Validate data types
   - [ ] Generate quality reports
   - [ ] Handle various edge cases

2. **DataQualityValidator** (3 tests)
   - [ ] Validate data completeness
   - [ ] Check for duplicate records
   - [ ] Verify position uniqueness

**Effort**: ~8 hours

### 3.4 Total Effort (Phase 3): ~26 hours

---

## Phase 4: Feature Engineering Tests (Week 2-3)

### 4.1 Feature Engineering Tests

**Location**: `tests/features/test_engineering.py`
**Current Status**: Missing

**Test Categories** (15-20 tests):

1. **Driver Form Calculation** (3 tests)
   - [ ] Calculate form score from recent races
   - [ ] Handle insufficient history gracefully
   - [ ] Normalize scores to 0-100 range

2. **Team Reliability** (3 tests)
   - [ ] Calculate from DNF statistics
   - [ ] Weight recent seasons higher
   - [ ] Handle teams with no history

3. **Circuit Performance** (3 tests)
   - [ ] Calculate driver's circuit-specific performance
   - [ ] Handle first-time circuit visits
   - [ ] Aggregate historical results correctly

4. **Feature Normalization** (2 tests)
   - [ ] Normalize features to 0-1 range
   - [ ] Handle outliers appropriately
   - [ ] Preserve feature relationships

5. **Edge Cases** (4 tests)
   - [ ] Handle missing data gracefully
   - [ ] Validate feature value ranges
   - [ ] Test with different dataset sizes

**Effort**: ~16 hours

### 4.2 Total Effort (Phase 4): ~16 hours

---

## Phase 5: Model Prediction Tests (Week 3-4)

### 5.1 Rule-Based Model Tests

**Location**: `tests/models/test_rule_based.py`
**Current Status**: Partial

**Test Count**: ~8 tests
**Effort**: ~6 hours

### 5.2 Logistic Regression Tests

**Location**: `tests/models/test_logistic.py`
**Current Status**: Partial

**Test Count**: ~10 tests
**Effort**: ~8 hours

### 5.3 Random Forest Tests

**Location**: `tests/models/test_random_forest.py`
**Current Status**: ✅ Complete (29 tests)

**Coverage**: ~90%
**Status**: Already implemented and passing

### 5.4 XGBoost Tests

**Location**: `tests/models/test_xgboost.py`
**Current Status**: ✅ Complete (20 tests)

**Coverage**: ~86%
**Status**: Already implemented and passing

### 5.5 LightGBM Tests

**Location**: `tests/models/test_lightgbm.py`
**Current Status**: ✅ Complete (15 tests)

**Coverage**: ~80%
**Status**: Already implemented and passing

### 5.6 Ensemble Tests

**Location**: `tests/models/test_ensemble.py`
**Current Status**: Missing

**Test Categories** (10-12 tests):

1. **Soft Voting Ensemble** (2 tests)
   - [ ] Aggregate probabilities correctly
   - [ ] Weight models equally by default

2. **Weighted Ensemble** (3 tests)
   - [ ] Apply custom weights
   - [ ] Validate weight normalization
   - [ ] Compare to unweighted version

3. **Model Agreement** (2 tests)
   - [ ] Calculate agreement score
   - [ ] Identify disagreement patterns

4. **Serialization** (2 tests)
   - [ ] Save ensemble with weights
   - [ ] Load and verify consistency

5. **Edge Cases** (2 tests)
   - [ ] Handle single model ensemble
   - [ ] Validate with different model types

**Effort**: ~10 hours

### 5.7 Total Effort (Phase 5): ~32 hours

---

## Phase 6: Performance & Metrics Tests (Week 4)

### 6.1 Performance Metrics Tests

**Location**: `tests/metrics/test_performance.py`
**Current Status**: Partial

**Test Categories** (12-15 tests):

1. **Accuracy Metrics** (3 tests)
   - [ ] Calculate precision correctly
   - [ ] Calculate recall correctly
   - [ ] Calculate F1 score

2. **Regression Metrics** (3 tests)
   - [ ] Calculate MAE
   - [ ] Calculate RMSE
   - [ ] Calculate R² score

3. **Model Comparison** (2 tests)
   - [ ] Compare models fairly
   - [ ] Identify best performer

4. **Cross-Validation** (2 tests)
   - [ ] Perform k-fold cross-validation
   - [ ] Calculate mean scores

**Effort**: ~10 hours

### 6.2 CLI Tests

**Location**: `tests/test_cli.py`
**Current Status**: Partial

**Test Categories** (8-10 tests):

1. **CLI Commands** (5 tests)
   - [ ] Test predict command
   - [ ] Test analyze command
   - [ ] Test train command
   - [ ] Test explain command
   - [ ] Test export command

2. **Error Handling** (2 tests)
   - [ ] Handle invalid arguments
   - [ ] Display helpful error messages

3. **Output Formatting** (2 tests)
   - [ ] Format output correctly
   - [ ] Write to file when requested

**Effort**: ~8 hours

### 6.3 Total Effort (Phase 6): ~18 hours

---

## Summary of Testing Scope

| Module | Test File | Tests | Target Coverage | Status |
|--------|-----------|-------|-----------------|--------|
| **api/base.py** | test_base.py | 12-15 | ≥85% | To implement |
| **api/ergast.py** | test_ergast.py | 10-12 | ≥85% | To implement |
| **data/collector.py** | test_collector.py | 15-18 | ≥90% | To implement |
| **data/models.py** | test_models.py | 8-10 | ≥85% | Partial |
| **data/cleaning.py** | test_cleaning.py | 8-10 | ≥85% | Partial |
| **features/engineering.py** | test_engineering.py | 15-20 | ≥85% | To implement |
| **models/rule_based.py** | test_rule_based.py | 8 | ≥80% | Partial |
| **models/logistic.py** | test_logistic.py | 10 | ≥80% | Partial |
| **models/random_forest.py** | test_random_forest.py | 29 | ≥90% | ✅ Complete |
| **models/xgboost_model.py** | test_xgboost.py | 20 | ≥86% | ✅ Complete |
| **models/lightgbm_model.py** | test_lightgbm.py | 15 | ≥80% | ✅ Complete |
| **models/ensemble.py** | test_ensemble.py | 10-12 | ≥85% | To implement |
| **metrics/performance.py** | test_performance.py | 12-15 | ≥75% | Partial |
| **cli.py** | test_cli.py | 8-10 | ≥60% | Partial |
| **Total** | | **~180-200** | **≥80%** | **In Progress** |

---

## Implementation Timeline

### Week 1 (Days 1-5)
- **Mon-Tue**: Pytest setup & fixtures (4 hours)
- **Wed-Thu**: API client tests (12 hours)
- **Fri**: Code review & fixes (4 hours)

### Week 2 (Days 6-10)
- **Mon-Tue**: Data collection tests (12 hours)
- **Wed-Thu**: Data model tests (8 hours)
- **Fri**: Review & integration (4 hours)

### Week 3 (Days 11-15)
- **Mon-Tue**: Feature engineering tests (10 hours)
- **Wed-Thu**: Model tests - rule-based & logistic (12 hours)
- **Fri**: Review & optimization (4 hours)

### Week 4 (Days 16-20)
- **Mon-Tue**: Ensemble & additional model tests (8 hours)
- **Wed-Thu**: Performance & CLI tests (12 hours)
- **Fri**: Coverage verification & final cleanup (4 hours)

---

## Success Criteria

### Code Coverage
- ✅ Overall coverage ≥80%
- ✅ Core modules ≥90% (api, data, features, models)
- ✅ Optional modules ≥60% (cli, visualization)
- ✅ All critical paths covered

### Test Quality
- ✅ No flaky tests
- ✅ All tests deterministic
- ✅ Proper mocking of external dependencies
- ✅ Clear test documentation

### Performance
- ✅ Full test suite runs in <2 minutes
- ✅ Unit tests only (no external dependencies)
- ✅ Parallel execution possible

### CI/CD Integration
- ✅ Tests pass in GitHub Actions
- ✅ Coverage reports generated
- ✅ Automatic failure on coverage drop

---

## Tools & Dependencies

### Current Setup
- ✅ pytest 7.4.0+
- ✅ pytest-cov 4.1.0+
- ✅ pytest-mock 3.11.0+
- ✅ All data science libraries

### Additional Configuration Needed
- [ ] pytest-xdist (for parallel execution)
- [ ] pytest-timeout (for long-running tests)
- [ ] hypothesis (for property-based testing - optional)

---

## Known Constraints

1. **External API Dependencies**: Mock all Ergast API calls
2. **File I/O**: Use temporary directories (tmp_path fixture)
3. **Model Training**: Use small sample data for speed
4. **Database**: Use in-memory or temporary SQLite for tests
5. **LLM Calls**: Mock all OpenAI/Anthropic API calls

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Coverage drops after changes | CI/CD fails on coverage drop |
| Flaky tests block CI | Use pytest-timeout, proper mocking |
| Tests too slow | Parallel execution, small data samples |
| External API failures | Mock all external dependencies |
| Test maintenance burden | Clear organization, good documentation |

---

## Next Steps

1. **Week 1**: Start with Phase 1 & 2 (pytest setup + API tests)
2. **Weekly Reviews**: Check coverage at end of each week
3. **Incremental Merge**: Merge test PRs as phases complete
4. **Optimization**: Profile test execution time and optimize
5. **Documentation**: Create testing guide for contributors

---

**Document Version**: 1.0
**Last Updated**: 2025-11-09
**Status**: Ready for implementation phase 1
