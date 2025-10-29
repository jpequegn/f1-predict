"""Tests for search space registry."""

import pytest

from f1_predict.optimization.search_spaces import SearchSpaceRegistry


class TestSearchSpaceRegistry:
    """Test cases for SearchSpaceRegistry."""

    def test_get_xgboost_search_space(self):
        """Test that get_search_space returns valid dict for xgboost."""
        space = SearchSpaceRegistry.get_search_space("xgboost")

        assert isinstance(space, dict)
        assert len(space) > 0

        # Check expected keys are present
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(space.keys()) == expected_keys

        # Check parameter structure
        assert space["n_estimators"]["type"] == "int"
        assert space["n_estimators"]["low"] == 100
        assert space["n_estimators"]["high"] == 500

        assert space["learning_rate"]["type"] == "float"
        assert space["learning_rate"]["log"] is True

    def test_get_lightgbm_search_space(self):
        """Test that get_search_space returns valid dict for lightgbm."""
        space = SearchSpaceRegistry.get_search_space("lightgbm")

        assert isinstance(space, dict)
        assert len(space) > 0

        # Check expected keys are present
        expected_keys = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(space.keys()) == expected_keys

        # Check parameter structure
        assert space["n_estimators"]["type"] == "int"
        assert space["learning_rate"]["log"] is True

    def test_get_random_forest_search_space(self):
        """Test that get_search_space returns valid dict for random_forest."""
        space = SearchSpaceRegistry.get_search_space("random_forest")

        assert isinstance(space, dict)
        assert len(space) > 0

        # Check expected keys are present
        expected_keys = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        }
        assert set(space.keys()) == expected_keys

        # Check parameter structure
        assert space["n_estimators"]["type"] == "int"
        assert space["max_features"]["type"] == "categorical"
        assert space["max_features"]["choices"] == ["sqrt", "log2"]

    def test_invalid_model_type_raises_error(self):
        """Test that get_search_space raises ValueError for invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type: invalid"):
            SearchSpaceRegistry.get_search_space("invalid")

    def test_all_spaces_have_required_keys(self):
        """Test that all registered spaces have proper structure."""
        for model_type in ["xgboost", "lightgbm", "random_forest"]:
            space = SearchSpaceRegistry.get_search_space(model_type)

            # All spaces should have at least one parameter
            assert len(space) > 0

            # Each parameter should have a type
            for param_name, param_config in space.items():
                assert "type" in param_config, f"{model_type}.{param_name} missing 'type'"

                # Check type-specific requirements
                if param_config["type"] in ["int", "float"]:
                    assert "low" in param_config, f"{model_type}.{param_name} missing 'low'"
                    assert "high" in param_config, f"{model_type}.{param_name} missing 'high'"
                elif param_config["type"] == "categorical":
                    assert "choices" in param_config, f"{model_type}.{param_name} missing 'choices'"

    def test_xgboost_lightgbm_spaces_identical(self):
        """Test that xgboost and lightgbm have identical search spaces."""
        xgb_space = SearchSpaceRegistry.get_search_space("xgboost")
        lgb_space = SearchSpaceRegistry.get_search_space("lightgbm")

        assert xgb_space == lgb_space

    def test_search_space_parameter_ranges(self):
        """Test that parameter ranges are sensible."""
        xgb_space = SearchSpaceRegistry.get_search_space("xgboost")

        # n_estimators should be reasonable
        assert 50 <= xgb_space["n_estimators"]["low"] <= 200
        assert 300 <= xgb_space["n_estimators"]["high"] <= 1000

        # max_depth should be reasonable
        assert 1 <= xgb_space["max_depth"]["low"] <= 5
        assert 5 <= xgb_space["max_depth"]["high"] <= 20

        # learning_rate should be small
        assert 0.0001 <= xgb_space["learning_rate"]["low"] <= 0.01
        assert 0.1 <= xgb_space["learning_rate"]["high"] <= 1.0
