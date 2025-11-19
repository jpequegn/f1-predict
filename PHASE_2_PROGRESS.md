# Issue #38 Phase 2 Progress - Multi-Modal Learning (Speed Traces)

**Date:** 2025-11-18
**Branch:** `feature/speed-trace-issue-38`
**Status:** In Progress (30% complete)

---

## Completed Work ✅

### 1. SpeedTraceGenerator Module
**File:** `src/f1_predict/models/speed_trace_generator.py` (116 lines)

**Features Implemented:**
- `__init__()`: Initialize with configurable output directory
- `generate_trace()`: Create single speed trace PNG for race/driver
  - Input: race_id, driver_id, lap_data list
  - Output: PNG path with lap vs speed plot
  - Includes min/max/avg statistics on plot
- `generate_batch()`: Bulk generation for multiple races
  - Handles multiple races and drivers
  - Error resilience with continue-on-fail
  - Returns dict mapping `race_id_driver_id` to paths

**Technical Details:**
- Uses matplotlib for visualization (800x600 plots)
- Creates organized directory structure: `output_dir/race_id/driver_id.png`
- Validates lap_data (empty, None, non-numeric detection)
- Proper error messages and logging on failures

### 2. Comprehensive Test Suite
**File:** `tests/models/test_speed_trace_generator.py` (357 lines, 24 tests)

**Test Coverage:**
- **Initialization (3 tests):** Default/custom output_dir, Path conversion
- **Trace Generation (7 tests):** File creation, dir structure, single/many laps, speed patterns, idempotency
- **Batch Operations (4 tests):** Dict return, multiple drivers/races, correct key formats
- **Error Handling (6 tests):** Empty data, None data, invalid speeds, negative speeds, missing fields
- **Output Validation (4 tests):** PNG validation, file size checks, separate directories, same driver different races

**Test Results:** ✅ 24/24 passing

---

## Remaining Phase 2 Work (70%)

### 3. Training Pipeline Infrastructure (Next)

**Files to Create:**
- `src/f1_predict/models/multimodal_trainer.py` (~200-250 lines)
- `tests/models/test_multimodal_trainer.py` (~300+ lines, 15+ tests)

**Components Needed:**
```python
class MultiModalTrainer:
    """Training orchestration for multi-modal models."""

    def __init__(self, model, dataset, optimizer, criterion):
        # Initialize trainer with model, data, optimizer, loss

    def train_epoch(self) -> Dict:
        # Forward pass, backprop, return loss/metrics

    def validate(self) -> Dict:
        # Evaluation on validation set

    def train(self, num_epochs: int) -> Dict:
        # Full training loop with early stopping

    def save_checkpoint(self, path: str):
        # Save model weights + optimizer state

    def load_checkpoint(self, path: str):
        # Load model weights + optimizer state
```

**Tests Required:**
- Initialization tests
- Single epoch training tests
- Validation tests
- Early stopping tests
- Checkpoint save/load tests
- Loss tracking tests
- Metrics computation tests

---

### 4. Dataset Cache Generator

**Files to Create:**
- `src/f1_predict/models/dataset_cache_builder.py` (~150-200 lines)
- `tests/models/test_dataset_cache_builder.py` (~200+ lines, 10+ tests)

**Purpose:** Build `dataset_cache.json` from processed race data

**Key Methods:**
```python
class DatasetCacheBuilder:
    """Build training dataset cache from processed race data."""

    def __init__(self, processed_data_dir: str, output_dir: str):
        pass

    def extract_lap_speeds(self, race_id: str, driver_id: str) -> List[float]:
        """Extract lap speeds from race telemetry"""

    def extract_tabular_features(self, race_id: str, driver_id: str) -> List[float]:
        """Extract weather, tire, driver, circuit features"""

    def build_cache(self, races: List[Dict]) -> Dict:
        """Generate complete dataset_cache.json"""

    def validate_cache(self) -> bool:
        """Verify cache completeness and correctness"""
```

---

### 5. Integration Tests

**Files to Create:**
- `tests/integration/test_multimodal_training.py` (~200+ lines, 4+ tests)

**Integration Test Cases:**
1. `test_train_one_epoch_sanity()`
   - Generate 50 speed traces
   - Create dataset
   - Train 1 epoch
   - Verify loss decreases

2. `test_baseline_vs_multimodal_comparison()`
   - Train tabular-only baseline
   - Train multi-modal model
   - Verify multi-modal >= baseline accuracy
   - Measure improvement %

3. `test_inference_latency()`
   - Generate 100 test samples
   - Measure inference time per sample
   - Assert < 1.0 second/sample

4. `test_missing_images_graceful_handling()`
   - Create dataset with ~20% missing images
   - Train for 1 epoch
   - Verify model handles gracefully
   - Measure accuracy impact

---

## Remaining Phase 2 Milestones

| Component | LOC | Tests | Priority |
|-----------|-----|-------|----------|
| MultiModalTrainer | 200-250 | 15+ | HIGH |
| DatasetCacheBuilder | 150-200 | 10+ | MEDIUM |
| Integration Tests | 200+ | 4+ | HIGH |
| Ruff Linting | - | - | HIGH |
| Final PR & Merge | - | - | HIGH |

**Total Estimated:** ~600-700 lines code, 30+ tests

---

## Key Design Decisions

### Training Architecture
- Adam optimizer, LR=1e-3 (from design doc)
- CrossEntropyLoss (position classification) or MSE (regression)
- ResNet-18 frozen (no fine-tuning in Phase 2)
- Tabular encoder: input_dim → hidden → 128 dims
- Fusion network: 640 → 256 → 128 → output_dim
- Batch size: 32 (configurable)
- Early stopping: monitor validation loss

### Validation Strategy
- 80/20 train/val split on races
- Accuracy metric: position prediction accuracy
- Baseline: EnsemblePredictor (tabular-only)
- Success criterion: >5% improvement over baseline

---

## Next Session Checklist

- [ ] Start with MultiModalTrainer implementation (TDD)
- [ ] Write comprehensive trainer tests first (RED phase)
- [ ] Implement trainer class (GREEN phase)
- [ ] Build DatasetCacheBuilder
- [ ] Create integration tests
- [ ] Run full Phase 2 test suite
- [ ] Verify all tests pass
- [ ] Commit and create PR
- [ ] Code review and merge

---

## Branch Info

**Current Branch:** `feature/speed-trace-issue-38`
**Remote:** `origin/feature/speed-trace-issue-38`
**Latest Commit:** `220d48d` - SpeedTraceGenerator implementation

**To Continue:**
```bash
# In next session
cd /Users/julienpequegnot/Code/f1-predict/.worktrees/speed-trace-issue-38
git pull origin feature/speed-trace-issue-38
# Ready to implement training pipeline
```

---

**Phase 2 Estimated Completion:** 1-2 more development sessions (4-6 hours coding)
