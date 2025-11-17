# Issue #38 Phase 1 Design: Multi-Modal Learning (Synthetic Speed Traces + Early Fusion)

**Date:** 2025-01-17
**Issue:** #38 - Multi-Modal Learning (Images + Tabular Data)
**Phase:** Phase 1 (Weeks 1-2)
**Status:** Design Approved

---

## Executive Summary

Phase 1 implements a proof-of-concept multi-modal learning system that combines synthetic speed-trace visualizations with tabular race features using early fusion architecture. The system generates PNG visualizations from telemetry data, extracts image features via ResNet-18, and concatenates them with tabular features for joint prediction.

**Key Design Decisions:**
- Image source: Synthetic speed-trace plots (no licensing concerns, fully controllable)
- Architecture: Early fusion (simplest, fastest inference)
- Vision backbone: ResNet-18 (pretrained, frozen features)
- Success criterion: >5% improvement over tabular-only baseline

---

## Section 1: System Architecture

### Data Pipeline

**Input → Processing → Output Flow:**

1. **Input:** Race telemetry data (from `data/processed/`)
   - Lap-by-lap speed profiles per driver per race
   - Driver info, circuit, weather, tire compound, results

2. **Image Generation:**
   - `SpeedTraceGenerator` creates 800x600px PNG showing speed vs lap number
   - Matplotlib-based plotting with consistent styling
   - Stored in `data/multimodal/speed_traces/[race_id]/[driver_id].png`

3. **Feature Extraction:**
   - Images: ResNet-18 backbone → 512-dimensional feature vector
   - Tabular: Driver, circuit, weather, tire, result features → Dense layer → 128 dims
   - No ResNet fine-tuning in Phase 1; use as frozen feature extractor

4. **Fusion & Prediction:**
   - Early fusion: Concatenate [img_features (512), tab_features (128)] → 640 dims
   - Fusion network: 640 → 256 → 128 → output (position or finish probability)
   - Output: Prediction of finishing position or probability of finishing

### Model Architecture Details

**Vision Encoder (ResNet-18):**
```
Input Image (224x224, 3-channel)
  → ResNet-18 (ImageNet pretrained)
  → Global average pooling
  → Output: 512-dimensional feature vector
```

**Tabular Encoder:**
```
Tabular Features (variable dims)
  → Dense(input_dims → 128)
  → ReLU
  → Output: 128-dimensional feature vector
```

**Fusion Network (Early Fusion):**
```
Image Features (512) + Tabular Features (128)
  → Concatenate → 640 dims
  → Dense(640 → 256) + ReLU + Dropout(0.3)
  → Dense(256 → 128) + ReLU + Dropout(0.3)
  → Dense(128 → output_size)
  → Softmax (if classification) or Sigmoid (if regression)
```

**Training Strategy:**
- ResNet-18: Frozen (no gradient updates)
- Fusion network: Trained end-to-end
- Tabular encoder: Trained end-to-end
- Loss: CrossEntropy (position classification) or MSE (regression to position)
- Optimizer: Adam with learning rate 1e-3

### Device Handling & Inference

- Models support CPU/GPU seamlessly
- Image preprocessing: normalization using ImageNet stats
- Batch inference: Process multiple (image, tabular) pairs in parallel
- Inference latency target: <1 second per prediction (including image load/encode)

---

## Section 2: Core Components & File Structure

### New Modules to Create

**In `src/f1_predict/models/`:**

#### 1. `vision_encoder.py`
```python
class VisionEncoder(nn.Module):
    """ResNet-18 based image feature extractor."""

    def __init__(self, pretrained=True, feature_dim=512):
        # Load pretrained ResNet-18
        # Remove classification head
        # Frozen gradients (no fine-tuning)

    def forward(self, images: Tensor) -> Tensor:
        """Images (B, 3, 224, 224) -> Features (B, 512)"""

    def encode_single(self, image: PIL.Image) -> Tensor:
        """Convenience method for single image"""
```

#### 2. `multimodal_fusion.py`
```python
class MultiModalFusionModel(nn.Module):
    """Early fusion of image and tabular features."""

    def __init__(self, image_feature_dim=512, tabular_input_dim=None,
                 output_dim=20):  # 20 positions
        # Vision encoder (frozen ResNet-18)
        # Tabular encoder (Dense layers)
        # Fusion network (concatenation + dense layers)

    def forward(self, images: Optional[Tensor], tabular_features: Tensor) -> Tensor:
        """
        Args:
            images: (B, 3, 224, 224) or None if missing
            tabular_features: (B, feature_dim)
        Returns:
            predictions: (B, output_dim)
        """
        # Handle missing images (return zeros)
        # Extract image features (if present)
        # Extract tabular features
        # Concatenate and fuse
        # Return predictions

    def eval_mode(self): ...
    def train_mode(self): ...
```

#### 3. `speed_trace_generator.py`
```python
class SpeedTraceGenerator:
    """Generates synthetic speed-trace visualizations."""

    def __init__(self, output_dir='data/multimodal/speed_traces'):
        self.output_dir = output_dir

    def generate_trace(self, race_id: str, driver_id: str,
                       lap_data: List[float]) -> str:
        """
        Generate speed trace PNG.

        Args:
            race_id: Unique race identifier
            driver_id: Unique driver identifier
            lap_data: List of lap times or speeds

        Returns:
            Path to generated PNG
        """
        # Create matplotlib figure
        # Plot lap number vs speed
        # Add labels, title
        # Save to data/multimodal/speed_traces/[race_id]/[driver_id].png
        # Return path

    def generate_batch(self, races: List[Dict]) -> Dict[str, str]:
        """Generate traces for multiple races."""
```

#### 4. `multimodal_dataset.py`
```python
class MultiModalDataset(Dataset):
    """PyTorch Dataset for paired image + tabular data."""

    def __init__(self, cache_file='data/multimodal/dataset_cache.json',
                 image_dir='data/multimodal/speed_traces'):
        # Load metadata from cache
        # Initialize transforms

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Optional[Tensor], Tensor, int]:
        """
        Returns:
            image_tensor: (3, 224, 224) or None if missing
            tabular_features: (feature_dim,)
            label: finishing position or 0/1 for finish
        """
        # Load metadata (race_id, driver_id, features)
        # Try to load image from disk
        # If missing, return None (model handles gracefully)
        # Load and normalize tabular features
        # Return tuple

    def _load_image(self, image_path: str) -> Optional[Tensor]:
        """Load and preprocess image, return None if file missing."""
```

### Storage Structure

```
data/multimodal/
├── speed_traces/
│   ├── 2024_race_1/
│   │   ├── driver_1.png
│   │   ├── driver_2.png
│   │   └── ...
│   ├── 2024_race_2/
│   │   └── ...
│   └── ...
├── dataset_cache.json
└── README.md
```

**`dataset_cache.json` Schema:**
```json
{
  "metadata": [
    {
      "race_id": "2024_race_1",
      "driver_id": "driver_1",
      "image_path": "data/multimodal/speed_traces/2024_race_1/driver_1.png",
      "tabular_features": [feature1, feature2, ...],
      "label": 1,
      "finished": true
    },
    ...
  ],
  "feature_names": ["driver_rating", "circuit_id", "weather", "tire_compound", ...],
  "generated_at": "2025-01-17T10:00:00Z"
}
```

### Dependencies to Add

Add to `pyproject.toml`:
```toml
[dependencies]
torchvision = "^0.16.0"  # For pretrained ResNet-18
pillow = "^10.0"         # Image processing
matplotlib = "^3.8"      # Speed trace visualization
```

---

## Section 3: Data Flow & Training Pipeline

### Data Preparation Phase (Pre-Training)

**Step 1: Load Race Data**
```python
# Load from data/processed/
races = load_processed_races(seasons=[2024, 2023, 2022])  # ~600 races
```

**Step 2: Generate Speed Traces**
```python
generator = SpeedTraceGenerator()
for race in races:
    for driver in race.drivers:
        lap_data = extract_lap_speeds(race, driver)
        image_path = generator.generate_trace(race.id, driver.id, lap_data)
        # Log: race.id, driver.id, image_path
```

**Step 3: Build Dataset Cache**
```python
cache = {
    "metadata": [
        {
            "race_id": race.id,
            "driver_id": driver.id,
            "image_path": ...,
            "tabular_features": [driver_rating, circuit_id, weather, ...],
            "label": driver.finishing_position,
            "finished": driver.dnf == False
        }
        for race in races
        for driver in race.drivers
    ]
}
# Save to data/multimodal/dataset_cache.json
```

### Training Loop

**Pseudocode:**
```python
# Initialize
encoder = VisionEncoder()
model = MultiModalFusionModel(
    image_feature_dim=512,
    tabular_input_dim=len(feature_names),
    output_dim=20  # positions 1-20
)
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()
dataset = MultiModalDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train for 20 epochs
for epoch in range(20):
    for batch_idx, (images, tabular, labels) in enumerate(dataloader):
        # Forward pass
        predictions = model(images, tabular)
        loss = criterion(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    # Validation
    val_acc = evaluate_on_validation_set(model)
    print(f"Epoch {epoch}, Validation Accuracy: {val_acc}")
```

### Validation & Metrics

**Baseline Comparison:**
```python
# Train tabular-only model (existing EnsemblePredictor)
baseline_accuracy = train_and_evaluate_baseline()

# Train multi-modal model
multimodal_accuracy = train_and_evaluate_multimodal()

# Compute improvement
improvement = (multimodal_accuracy - baseline_accuracy) / baseline_accuracy * 100
print(f"Improvement: {improvement:.2f}%")
assert improvement > 5.0, "Must exceed 5% improvement threshold"
```

**Inference Latency Measurement:**
```python
import time

model.eval()
with torch.no_grad():
    start = time.time()
    for _ in range(100):
        images = load_batch_images(4)  # batch of 4
        tabular = load_batch_tabular(4)
        predictions = model(images, tabular)
    elapsed = (time.time() - start) / 100 / 4  # avg per sample
    print(f"Inference time per sample: {elapsed*1000:.2f}ms")
    assert elapsed < 1.0, "Must meet <1 second per prediction target"
```

### Handling Missing Images

**Design:**
- `MultiModalDataset.__getitem__()` returns `None` if image file missing
- `MultiModalFusionModel.forward()` checks for `None` and uses zero-vector as fallback
- Loss computation: model still produces predictions using tabular features alone
- Metrics: track % of samples with missing images vs with images

**Implementation in forward():**
```python
def forward(self, images: Optional[Tensor], tabular_features: Tensor) -> Tensor:
    # Image features
    if images is not None:
        image_features = self.vision_encoder(images)  # (B, 512)
    else:
        # Graceful degradation: use zeros
        batch_size = tabular_features.shape[0]
        image_features = torch.zeros(batch_size, 512, device=tabular_features.device)

    # Tabular features
    tabular_encoded = self.tabular_encoder(tabular_features)  # (B, 128)

    # Concatenate and fuse
    fused = torch.cat([image_features, tabular_encoded], dim=1)  # (B, 640)
    output = self.fusion_network(fused)  # (B, output_dim)

    return output
```

---

## Section 4: Testing & Error Handling

### Unit Tests

**Location:** `tests/models/`

#### `test_vision_encoder.py`
```python
class TestVisionEncoder:
    """Test ResNet-18 feature extraction."""

    def test_output_shape(self):
        """Verify (batch, 3, 224, 224) -> (batch, 512)"""

    def test_different_input_sizes(self):
        """Test resizing of non-224x224 images"""

    def test_device_placement(self):
        """Verify CPU/GPU handling"""

    def test_pretrained_weights(self):
        """Verify ImageNet weights loaded correctly"""

    def test_gradients_frozen(self):
        """Verify no gradients computed for ResNet"""
```

#### `test_multimodal_fusion.py`
```python
class TestMultiModalFusionModel:
    """Test fusion architecture."""

    def test_forward_pass_shape(self):
        """(B, 3, 224, 224) + (B, feature_dim) -> (B, output_dim)"""

    def test_missing_image_handling(self):
        """Verify None image input -> zero-vector fallback"""

    def test_gradient_flow(self):
        """Verify gradients flow through fusion, not ResNet"""

    def test_batch_consistency(self):
        """Same input twice -> same output"""

    def test_output_range(self):
        """Output probabilities in valid range [0, 1]"""
```

#### `test_speed_trace_generator.py`
```python
class TestSpeedTraceGenerator:
    """Test synthetic image generation."""

    def test_generate_trace_creates_file(self):
        """Verify PNG file created at expected path"""

    def test_output_dimensions(self):
        """Verify output image is loadable and correct size"""

    def test_edge_cases(self):
        """1 lap, 100 laps, missing data points"""

    def test_consistent_styling(self):
        """Multiple calls produce visually similar traces"""

    def test_invalid_input_handling(self):
        """Empty list, None data -> graceful error"""
```

#### `test_multimodal_dataset.py`
```python
class TestMultiModalDataset:
    """Test data loading."""

    def test_lazy_loading(self):
        """Verify images loaded on-demand, not all at init"""

    def test_missing_image_fallback(self):
        """File not found -> returns None tensor"""

    def test_batch_stacking(self):
        """DataLoader handles mixed None/tensor batches"""

    def test_feature_normalization(self):
        """Tabular features normalized correctly"""

    def test_cache_file_format(self):
        """Verify cache.json loads and parses correctly"""
```

### Integration Test

**Location:** `tests/integration/`

#### `test_multimodal_integration.py`
```python
class TestMultiModalIntegration:
    """End-to-end multi-modal training."""

    def test_train_one_epoch(self):
        """
        - Generate 100 speed traces
        - Create dataset
        - Train model for 1 epoch
        - Verify loss decreases
        - No crashes or OOM
        """

    def test_baseline_vs_multimodal_comparison(self):
        """
        - Train tabular-only model
        - Train multi-modal model
        - Verify multi-modal >= tabular-only accuracy
        - Measure improvement percentage
        """

    def test_inference_latency(self):
        """
        - Generate 100 test samples
        - Measure inference time per sample
        - Assert < 1.0 second per sample
        """

    def test_missing_images_mixed_batch(self):
        """
        - Create dataset with ~20% missing images
        - Train for 1 epoch
        - Verify model handles gracefully
        - Measure accuracy impact of missing images
        """
```

### Error Handling Strategy

**Image Generation Failures:**
- Issue: `SpeedTraceGenerator` fails (bad data, file write error)
- Handling: Catch exception, log warning with race_id, driver_id
- Action: Continue processing next race (skip this one)
- Log: Write to `data/multimodal/generation_errors.log`

**Out-of-Memory During Processing:**
- Issue: Large batch size causes OOM when loading images
- Handling: Detect in DataLoader, reduce batch size automatically
- Action: Emit warning, retry with smaller batch
- Fallback: Process images one at a time

**Missing Telemetry Data:**
- Issue: Race/driver has no lap speed data available
- Handling: Detect during cache building, skip that race
- Action: Log to `data/multimodal/skipped_races.json`
- Count: Track total skipped to report in summary

**Training Convergence Issues:**
- Issue: Loss not decreasing, model not learning
- Handling: Log learning rate, feature distributions, sample gradients
- Action: Provide diagnostics for debugging
- Fallback: Use lower learning rate (5e-4)

**Inference Shape Mismatches:**
- Issue: Batch with mixed image/None tensors causes shape error
- Handling: Pad None images to zeros in DataLoader collate function
- Action: Ensure all batches have consistent tensor shapes
- Test: Explicitly test mixed-image batches

---

## Acceptance Criteria

- [ ] **Image Generation:** Successfully generate 500+ speed-trace images from race data
- [ ] **Architecture:** Implement VisionEncoder, MultiModalFusionModel, SpeedTraceGenerator, MultiModalDataset
- [ ] **Training:** Train multi-modal model on 400+ races, no crashes
- [ ] **Performance:** Achieve >5% improvement over tabular-only baseline
- [ ] **Inference:** <1.0 second per prediction (image + tabular processing)
- [ ] **Robustness:** Handle missing images gracefully without crashes
- [ ] **Testing:** 40+ unit tests + 1 integration test, ≥80% code coverage
- [ ] **Documentation:** Design doc + code comments + error logging

---

## Dependencies

**New Python Packages:**
- `torchvision>=0.16.0` – Pretrained ResNet-18
- `pillow>=10.0` – Image processing
- `matplotlib>=3.8` – Visualization

**Existing Packages Used:**
- `torch` – Neural networks
- `pandas` – Data manipulation
- `numpy` – Numerical operations

---

## Timeline & Effort

- **Week 1:** VisionEncoder + SpeedTraceGenerator + MultiModalDataset (~15 hours)
- **Week 2:** MultiModalFusionModel + Training pipeline + Tests (~20 hours)
- **Total:** ~35 hours (fits within 2-week Phase 1 estimate)

---

## Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| Image generation success rate | ≥95% | 500+ images generated |
| Model training convergence | Loss decreases | 1-epoch sanity check |
| Accuracy improvement | >5% vs baseline | Validation accuracy comparison |
| Inference latency | <1.0 sec/sample | 100-sample timing test |
| Code coverage | ≥80% | pytest --cov output |
| Test count | ≥40 tests | test discovery count |

---

## Open Questions / Future Phases

- **Phase 2:** Should we fine-tune ResNet or add attention mechanisms?
- **Phase 3:** Integrate other image modalities (circuit diagrams, weather icons)?
- **Phase 3:** Apply to real F1 broadcast images (licensing permitting)?

---

**Approved by:** User
**Date:** 2025-01-17
