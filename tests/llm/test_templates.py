"""Tests for prompt template system."""

from pathlib import Path

import pytest
from jinja2 import TemplateError

from f1_predict.llm.exceptions import TemplateError as LLMTemplateError
from f1_predict.llm.templates import PromptTemplateManager, get_default_template


@pytest.fixture
def templates_dir(tmp_path):
    """Create temporary templates directory with test templates."""
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()

    # Create simple test template
    simple_template = templates_dir / "simple.jinja2"
    simple_template.write_text("Hello {{ name }}!")

    # Create template with loop
    loop_template = templates_dir / "loop.jinja2"
    loop_template.write_text(
        "Items:\n{% for item in items %}\n- {{ item }}\n{% endfor %}"
    )

    # Create template with conditionals
    conditional_template = templates_dir / "conditional.jinja2"
    conditional_template.write_text(
        "{% if show_header %}Header: {{ title }}{% endif %}\nContent: {{ content }}"
    )

    return templates_dir


@pytest.fixture
def manager(templates_dir):
    """Create template manager with test templates directory."""
    return PromptTemplateManager(templates_dir=templates_dir)


class TestPromptTemplateManager:
    """Test prompt template manager."""

    def test_initialization_with_custom_dir(self, templates_dir):
        """Test initialization with custom templates directory."""
        manager = PromptTemplateManager(templates_dir=templates_dir)
        assert manager.templates_dir == templates_dir

    def test_initialization_with_default_dir(self):
        """Test initialization with default templates directory."""
        manager = PromptTemplateManager()
        expected_dir = Path(__file__).parents[2] / "config" / "prompts"
        assert manager.templates_dir == expected_dir

    def test_render_simple_template(self, manager):
        """Test rendering simple template."""
        result = manager.render("simple.jinja2", name="World")
        assert result == "Hello World!"

    def test_render_template_with_loop(self, manager):
        """Test rendering template with loops."""
        result = manager.render("loop.jinja2", items=["apple", "banana", "cherry"])
        assert "- apple" in result
        assert "- banana" in result
        assert "- cherry" in result

    def test_render_template_with_conditional(self, manager):
        """Test rendering template with conditionals."""
        # With header
        result = manager.render(
            "conditional.jinja2",
            show_header=True,
            title="Test Title",
            content="Test content"
        )
        assert "Header: Test Title" in result
        assert "Content: Test content" in result

        # Without header
        result = manager.render(
            "conditional.jinja2",
            show_header=False,
            content="Test content"
        )
        assert "Header" not in result
        assert "Content: Test content" in result

    def test_render_missing_template(self, manager):
        """Test error handling for missing template."""
        with pytest.raises(LLMTemplateError, match="Template rendering failed"):
            manager.render("nonexistent.jinja2", name="World")

    def test_render_missing_variable(self, manager):
        """Test error handling for missing template variable."""
        # Jinja2 by default doesn't error on missing variables (renders as empty)
        result = manager.render("simple.jinja2")  # Missing 'name'
        assert result == "Hello !"

    def test_render_string_simple(self, manager):
        """Test rendering from string template."""
        template_str = "Hello {{ name }}!"
        result = manager.render_string(template_str, name="World")
        assert result == "Hello World!"

    def test_render_string_complex(self, manager):
        """Test rendering complex string template."""
        template_str = """
        {% for item in items %}
        {{ loop.index }}. {{ item }}
        {% endfor %}
        """
        result = manager.render_string(template_str, items=["first", "second", "third"])
        assert "1. first" in result
        assert "2. second" in result
        assert "3. third" in result

    def test_render_string_with_error(self, manager):
        """Test error handling for invalid string template."""
        template_str = "Hello {{ name }"  # Invalid syntax
        with pytest.raises(LLMTemplateError, match="Template rendering failed"):
            manager.render_string(template_str, name="World")

    def test_get_template_method(self, manager):
        """Test direct template access."""
        template = manager.get_template("simple.jinja2")
        assert template is not None
        result = template.render(name="World")
        assert result == "Hello World!"

    def test_get_nonexistent_template(self, manager):
        """Test error for nonexistent template."""
        with pytest.raises(LLMTemplateError):
            manager.get_template("nonexistent.jinja2")


class TestDefaultTemplates:
    """Test default template functionality."""

    def test_get_default_race_preview_template(self):
        """Test getting race preview default template."""
        template = get_default_template("race_preview")
        assert "race preview" in template.lower()
        assert "{{ race_name }}" in template
        assert "{{ circuit_name }}" in template

    def test_get_default_prediction_explanation_template(self):
        """Test getting prediction explanation default template."""
        template = get_default_template("prediction_explanation")
        assert "prediction" in template.lower()
        assert "{{ driver_name }}" in template
        assert "{{ position }}" in template
        assert "{{ confidence }}" in template

    def test_get_unknown_default_template(self):
        """Test error for unknown default template."""
        with pytest.raises(ValueError, match="Unknown template"):
            get_default_template("nonexistent_template")


class TestRealTemplates:
    """Test with actual project templates if they exist."""

    @pytest.fixture
    def real_manager(self):
        """Create manager with real project templates."""
        return PromptTemplateManager()  # Uses default directory

    def test_render_race_preview_template(self, real_manager):
        """Test rendering actual race preview template."""
        try:
            result = real_manager.render(
                "race_preview.jinja2",
                race_name="Monaco Grand Prix",
                circuit_name="Circuit de Monaco",
                race_date="2024-05-26",
                round_number=8,
                circuit_characteristics="Street circuit with tight corners",
                past_winner="Max Verstappen",
                lap_record="1:10.166 (Lewis Hamilton, 2021)",
                top_drivers=[
                    {"name": "Max Verstappen", "points": 150},
                    {"name": "Sergio Perez", "points": 120},
                ],
                predicted_winner="Max Verstappen",
                confidence=85,
                predicted_podium="Verstappen, Perez, Hamilton",
                key_factors=["Track position", "Qualifying pace"],
            )
            assert "Monaco Grand Prix" in result
            assert "Max Verstappen" in result
        except LLMTemplateError:
            pytest.skip("Race preview template not found")

    def test_render_prediction_explanation_template(self, real_manager):
        """Test rendering actual prediction explanation template."""
        try:
            result = real_manager.render(
                "prediction_explanation.jinja2",
                driver_name="Max Verstappen",
                position=1,
                confidence=92,
                model_name="XGBoost",
                top_features=[
                    ("qualifying_position", 0.35, "Started from pole"),
                    ("recent_form", 0.25, "Won last 3 races"),
                ],
                circuit="Monaco",
                past_performance="3 wins in last 5 years",
                recent_results="1st, 1st, 2nd, 1st, 1st",
            )
            assert "Max Verstappen" in result
            assert "92" in result or "confidence" in result.lower()
        except LLMTemplateError:
            pytest.skip("Prediction explanation template not found")
