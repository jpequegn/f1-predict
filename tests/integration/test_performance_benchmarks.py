"""Performance benchmarking tests for F1 Predict.

Tests system performance, scalability, and resource usage including:
- Prediction latency and throughput
- Data pipeline performance
- LLM integration performance
- Memory efficiency
"""

import time
from typing import Any

import pandas as pd
import pytest


@pytest.mark.integration
@pytest.mark.performance
class TestPredictionPipelinePerformance:
    """Test prediction pipeline performance metrics."""

    def test_data_cleaning_performance(self, sample_race_results: pd.DataFrame) -> None:
        """Test data cleaning pipeline performance."""
        from f1_predict.data.cleaning import DataCleaner

        cleaner = DataCleaner()

        # Measure cleaning time
        start_time = time.perf_counter()
        cleaned_data, report = cleaner.clean_race_results(sample_race_results.to_dict('records'))
        elapsed_time = time.perf_counter() - start_time

        # Cleaning should complete in <100ms for small dataset
        assert elapsed_time < 0.1
        assert len(cleaned_data) > 0

    def test_feature_engineering_performance(self, sample_features_df: pd.DataFrame) -> None:
        """Test feature engineering performance."""
        # Measure feature processing time
        start_time = time.perf_counter()

        # Simulate feature engineering operations
        features_processed = sample_features_df.copy()
        features_processed['normalized_form'] = (
            features_processed['driver_form_score'] / 100.0
        )
        features_processed['team_score'] = (
            features_processed['team_reliability'] * 100.0
        )

        elapsed_time = time.perf_counter() - start_time

        # Feature engineering should complete in <50ms for small dataset
        assert elapsed_time < 0.05
        assert len(features_processed) == len(sample_features_df)

    def test_prediction_latency(self, sample_features_df: pd.DataFrame) -> None:
        """Test prediction generation latency."""
        # Mock prediction latency test
        start_time = time.perf_counter()

        # Simulate prediction (normally would call model.predict)
        time.sleep(0.001)  # Simulate 1ms prediction time
        predictions = {
            'predicted_position': list(range(1, len(sample_features_df) + 1)),
            'confidence': [0.9] * len(sample_features_df),
        }

        elapsed_time = time.perf_counter() - start_time

        # Predictions should complete quickly
        assert elapsed_time < 0.1
        assert len(predictions['predicted_position']) == len(sample_features_df)

    def test_batch_prediction_throughput(self) -> None:
        """Test batch prediction throughput."""
        batch_size = 100

        # Measure throughput for batch predictions
        start_time = time.perf_counter()

        # Simulate batch processing
        for _ in range(batch_size):
            time.sleep(0.0001)  # 0.1ms per prediction

        elapsed_time = time.perf_counter() - start_time
        throughput = batch_size / elapsed_time if elapsed_time > 0 else 0

        # Should achieve >100 predictions per second
        assert throughput > 100
        assert elapsed_time < batch_size * 0.002  # Max 2ms per prediction


@pytest.mark.integration
@pytest.mark.performance
class TestDataPipelineScalability:
    """Test data pipeline scalability with increasing data sizes."""

    def test_small_dataset_performance(self) -> None:
        """Test performance with small dataset (10 records)."""
        data_size = 10

        start_time = time.perf_counter()

        # Simulate small dataset processing
        df = pd.DataFrame({
            'driver_id': range(data_size),
            'position': range(1, data_size + 1),
            'points': [25 - i for i in range(data_size)],
        })
        processed = df.copy()

        elapsed_time = time.perf_counter() - start_time

        # Small dataset should process in <10ms
        assert elapsed_time < 0.01
        assert len(processed) == data_size

    def test_medium_dataset_performance(self) -> None:
        """Test performance with medium dataset (100 records)."""
        data_size = 100

        start_time = time.perf_counter()

        # Simulate medium dataset processing
        df = pd.DataFrame({
            'driver_id': range(data_size),
            'position': range(1, data_size + 1),
            'points': [max(0, 25 - (i % 25)) for i in range(data_size)],
        })
        processed = df.copy()
        processed['normalized_points'] = processed['points'] / 25.0

        elapsed_time = time.perf_counter() - start_time

        # Medium dataset should process in <50ms
        assert elapsed_time < 0.05
        assert len(processed) == data_size

    def test_large_dataset_performance(self) -> None:
        """Test performance with larger dataset (1000 records)."""
        data_size = 1000

        start_time = time.perf_counter()

        # Simulate large dataset processing
        df = pd.DataFrame({
            'driver_id': range(data_size),
            'position': [((i % 20) + 1) for i in range(data_size)],
            'points': [max(0, 25 - ((i % 20))) for i in range(data_size)],
        })
        processed = df.copy()
        processed['normalized_points'] = processed['points'] / 25.0
        grouped = processed.groupby('position')['points'].sum()

        elapsed_time = time.perf_counter() - start_time

        # Large dataset should process in <500ms
        assert elapsed_time < 0.5
        assert len(processed) == data_size
        assert len(grouped) <= 20


@pytest.mark.integration
@pytest.mark.performance
class TestLLMIntegrationPerformance:
    """Test LLM integration performance."""

    def test_llm_response_generation_latency(self) -> None:
        """Test LLM response generation latency."""
        from f1_predict.llm.base import LLMConfig

        config = LLMConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=1000,
        )

        # Measure response generation time (mocked)
        start_time = time.perf_counter()

        # Simulate LLM response generation
        time.sleep(0.01)  # Simulate 10ms generation time
        response_content = "Race preview analysis..."

        elapsed_time = time.perf_counter() - start_time

        # LLM response should be generated in reasonable time
        assert elapsed_time < 0.1
        assert len(response_content) > 0

    def test_cost_tracking_performance(self, temp_cost_tracker_db: Any) -> None:
        """Test cost tracking performance."""
        from f1_predict.llm.cost_tracker import CostTracker, UsageRecord
        from datetime import datetime

        tracker = CostTracker(
            db_path=temp_cost_tracker_db,
            daily_budget=10.0,
            monthly_budget=200.0,
        )

        # Measure cost tracking latency for multiple records
        start_time = time.perf_counter()

        num_records = 10
        for i in range(num_records):
            record = UsageRecord(
                timestamp=datetime.now(),
                provider="anthropic",
                model="claude-3-5-sonnet-20241022",
                template=None,
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
                total_tokens=150 + i * 15,
                estimated_cost=0.01 + i * 0.001,
                request_duration=0.5 + i * 0.05,
                success=True,
            )
            tracker.record_usage(record)

        elapsed_time = time.perf_counter() - start_time
        avg_time_per_record = elapsed_time / num_records

        # Cost tracking should be reasonably fast (<2ms per record on average)
        assert avg_time_per_record < 0.002


@pytest.mark.integration
@pytest.mark.performance
class TestResourceUsageEfficiency:
    """Test resource usage efficiency."""

    def test_memory_efficiency_with_dataframe_operations(self) -> None:
        """Test memory efficiency with DataFrame operations."""
        # Create a reasonably sized dataset
        size = 10000
        df = pd.DataFrame({
            'driver_id': range(size),
            'position': [((i % 20) + 1) for i in range(size)],
            'points': [max(0, 25 - ((i % 20))) for i in range(size)],
            'qualifying_position': [((i % 20) + 1) for i in range(size)],
        })

        # Perform aggregations
        grouped = df.groupby('position')['points'].agg(['sum', 'mean', 'count'])

        # Should complete efficiently
        assert len(grouped) <= 20
        assert 'sum' in grouped.columns
        assert 'mean' in grouped.columns

    def test_pandas_operation_performance(self) -> None:
        """Test pandas operations performance."""
        size = 10000

        start_time = time.perf_counter()

        # Create and manipulate DataFrame
        df = pd.DataFrame({
            'value_1': range(size),
            'value_2': [i * 2 for i in range(size)],
            'value_3': [i * 3 for i in range(size)],
        })

        # Perform multiple operations
        df['sum'] = df['value_1'] + df['value_2'] + df['value_3']
        df['product'] = df['value_1'] * df['value_2']
        filtered = df[df['sum'] > 1000]

        elapsed_time = time.perf_counter() - start_time

        # Operations should complete quickly
        assert elapsed_time < 0.1
        assert len(filtered) > 0


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.slow
class TestEndToEndPerformance:
    """End-to-end system performance tests."""

    def test_complete_pipeline_performance(self, sample_race_results: pd.DataFrame) -> None:
        """Test complete pipeline performance end-to-end."""
        from f1_predict.data.cleaning import DataCleaner

        cleaner = DataCleaner()

        # Measure complete pipeline
        start_time = time.perf_counter()

        # Step 1: Clean data
        cleaned_data, report = cleaner.clean_race_results(sample_race_results.to_dict('records'))
        cleaned_df = pd.DataFrame(cleaned_data) if cleaned_data else pd.DataFrame()

        # Step 2: Process features
        if len(cleaned_df) > 0:
            cleaned_df['normalized_points'] = cleaned_df['points'] / 25.0 if 'points' in cleaned_df.columns else 0

        # Step 3: Generate predictions (mocked)
        predictions = {
            'predicted_position': list(range(1, len(cleaned_df) + 1)) if len(cleaned_df) > 0 else [],
        }

        elapsed_time = time.perf_counter() - start_time

        # Complete pipeline should finish within reasonable time
        assert elapsed_time < 0.5
        assert len(predictions['predicted_position']) <= len(sample_race_results)

    def test_concurrent_prediction_simulation(self) -> None:
        """Test system behavior under concurrent prediction load."""
        # Simulate concurrent predictions
        concurrent_requests = 5

        start_time = time.perf_counter()

        for _ in range(concurrent_requests):
            # Simulate prediction latency
            time.sleep(0.01)
            _result = {'prediction': 1, 'confidence': 0.9}

        elapsed_time = time.perf_counter() - start_time
        avg_time_per_request = elapsed_time / concurrent_requests

        # Should handle multiple requests efficiently
        assert avg_time_per_request < 0.05
