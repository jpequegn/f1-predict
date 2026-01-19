"""Prompt template management system.

This module provides a Jinja2-based template system for managing
LLM prompts with variable interpolation and template validation.
"""

from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, Template, TemplateError
import structlog

from f1_predict.llm.exceptions import TemplateError as LLMTemplateError

logger = structlog.get_logger(__name__)


class PromptTemplateManager:
    """Manager for LLM prompt templates.

    Handles loading, rendering, and validating Jinja2 templates
    for F1 analysis generation.
    """

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template manager.

        Args:
            templates_dir: Directory containing template files.
                          Defaults to config/prompts/
        """
        if templates_dir is None:
            # Default to config/prompts relative to project root
            templates_dir = Path(__file__).parents[3] / "config" / "prompts"

        self.templates_dir = templates_dir
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=False,  # We're generating prompts, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self.logger = logger.bind(templates_dir=str(self.templates_dir))
        self.logger.info("template_manager_initialized")

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a template with provided variables.

        Args:
            template_name: Name of template file (e.g., 'race_preview.jinja2')
            **kwargs: Variables to interpolate into template

        Returns:
            Rendered template string

        Raises:
            LLMTemplateError: If template not found or rendering fails
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**kwargs)

            self.logger.debug(
                "template_rendered",
                template=template_name,
                vars_count=len(kwargs),
                output_length=len(rendered),
            )

            return rendered

        except TemplateError as e:
            msg = f"Template rendering failed for {template_name}: {e}"
            self.logger.error("template_error", template=template_name, error=str(e))
            raise LLMTemplateError(msg) from e

    def render_string(self, template_string: str, **kwargs: Any) -> str:
        """Render a template from string.

        Args:
            template_string: Template content as string
            **kwargs: Variables to interpolate into template

        Returns:
            Rendered template string

        Raises:
            LLMTemplateError: If rendering fails
        """
        try:
            template = Template(template_string)
            rendered = template.render(**kwargs)

            self.logger.debug(
                "string_template_rendered",
                vars_count=len(kwargs),
                output_length=len(rendered),
            )

            return rendered

        except TemplateError as e:
            msg = f"String template rendering failed: {e}"
            self.logger.error("string_template_error", error=str(e))
            raise LLMTemplateError(msg) from e

    def list_templates(self) -> list[str]:
        """List available template files.

        Returns:
            List of template file names
        """
        if not self.templates_dir.exists():
            return []

        templates = [f.name for f in self.templates_dir.glob("*.jinja2")]

        self.logger.debug("templates_listed", count=len(templates))
        return templates

    def validate_template(self, template_name: str) -> bool:
        """Validate that a template exists and can be parsed.

        Args:
            template_name: Name of template file

        Returns:
            True if template is valid

        Raises:
            LLMTemplateError: If template is invalid
        """
        try:
            self.env.get_template(template_name)
            self.logger.debug("template_validated", template=template_name)
            return True
        except TemplateError as e:
            msg = f"Template validation failed for {template_name}: {e}"
            self.logger.error(
                "template_validation_error", template=template_name, error=str(e)
            )
            raise LLMTemplateError(msg) from e

    def get_template(self, template_name: str) -> "Template":
        """Get a template object directly.

        Args:
            template_name: Name of template file

        Returns:
            Jinja2 Template object

        Raises:
            LLMTemplateError: If template not found or invalid
        """
        try:
            template = self.env.get_template(template_name)
            self.logger.debug("template_retrieved", template=template_name)
            return template
        except TemplateError as e:
            msg = f"Template retrieval failed for {template_name}: {e}"
            self.logger.error(
                "template_retrieval_error", template=template_name, error=str(e)
            )
            raise LLMTemplateError(msg) from e


# Pre-defined templates as fallbacks if files don't exist
DEFAULT_TEMPLATES = {
    "race_preview": """You are an F1 race analyst. Generate a race preview for:

Race: {{ race_name }}
Circuit: {{ circuit_name }}
Date: {{ race_date }}

Historical Data:
- Past winner: {{ past_winner }}
- Circuit characteristics: {{ circuit_characteristics }}

Current Form:
{% for driver in top_drivers %}
- {{ driver.name }}: {{ driver.recent_form }}
{% endfor %}

ML Predictions:
- Predicted winner: {{ predicted_winner }} ({{ confidence }}% confidence)
- Predicted podium: {{ predicted_podium }}

Provide:
1. Circuit analysis (2-3 sentences)
2. Key drivers to watch (3-4 drivers)
3. Race strategy considerations
4. Bold prediction with reasoning

Tone: Professional but engaging, like Sky Sports F1 commentary.
Word count: 400-600 words.""",
    "prediction_explanation": """Explain this F1 race prediction in plain English:

Prediction: {{ driver_name }} to finish {{ position }}
Confidence: {{ confidence }}%
Model: {{ model_name }}

Contributing Factors:
{% for feature, importance in top_features %}
- {{ feature }}: {{ importance }}
{% endfor %}

Historical Context:
- Driver's past performance at {{ circuit }}: {{ past_performance }}
- Recent form: {{ recent_results }}

Explain:
1. Why this prediction makes sense
2. Key factors influencing the outcome
3. Potential variables that could change the prediction
4. How confident we should be in this prediction

Use analogies and context that F1 fans would understand.
Keep explanation under 300 words.""",
    "driver_comparison": """Compare the following two F1 drivers:

**{{ driver1.name }}**
- Wins: {{ driver1.wins }}
- Podiums: {{ driver1.podiums }}
- Points: {{ driver1.points }}
- Recent Form: {{ driver1.recent_form }}

**{{ driver2.name }}**
- Wins: {{ driver2.wins }}
- Podiums: {{ driver2.podiums }}
- Points: {{ driver2.points }}
- Recent Form: {{ driver2.recent_form }}

Provide a concise comparison (3-4 sentences) highlighting:
1. Key statistical differences
2. Recent performance trends
3. Strengths of each driver
4. Overall verdict on who has the current edge""",
    "post_race_analysis": """Write a post-race analysis for:

Race: {{ race_name }}
Winner: {{ winner }}

Results (Top 10):
{% for result in results[:10] %}
{{ result.position }}. {{ result.driver }} - {{ result.gap }}
{% endfor %}

{% if key_moments %}
Key Moments:
{% for moment in key_moments %}
- {{ moment }}
{% endfor %}
{% endif %}

Provide analysis covering:
1. Winner's performance (2-3 sentences)
2. Key battles and overtakes
3. Strategic insights
4. Championship implications

Word count: 300-400 words.""",
    "championship_analysis": """Analyze the current F1 championship situation:

Season: {{ season }}
Round: {{ current_round }} of {{ total_rounds }}

Drivers' Championship:
{% for standing in driver_standings[:5] %}
{{ standing.position }}. {{ standing.driver }} - {{ standing.points }} pts
{% endfor %}

Provide analysis of:
1. Current title battle state
2. Key contenders and their chances
3. Mathematical scenarios
4. Prediction for season outcome

Be analytical and insightful.""",
    "qualifying_preview": """Generate a qualifying preview for:

Race: {{ race_name }}
Circuit: {{ circuit_name }}

Practice Pace Leaders:
{% for driver in practice_leaders %}
{{ loop.index }}. {{ driver.name }} - {{ driver.time }}
{% endfor %}

Weather: {{ weather_conditions }}

Preview should include:
1. Pole favorites based on practice
2. Circuit qualifying characteristics
3. Potential surprises
4. Pole prediction""",
    "strategy_analysis": """Analyze race strategy options for:

Race: {{ race_name }}
Laps: {{ total_laps }}
Pit Lane Delta: {{ pit_lane_delta }}s

Tire Compounds:
{% for compound in tire_compounds %}
- {{ compound.name }}: ~{{ compound.stint_length }} laps
{% endfor %}

Provide:
1. Optimal strategy recommendation
2. Alternative approaches
3. Key pit windows
4. Safety car impact scenarios""",
}


def get_default_template(template_type: str) -> str:
    """Get default template content.

    Args:
        template_type: Type of template (e.g., 'race_preview')

    Returns:
        Default template string

    Raises:
        LLMTemplateError: If template type not found
    """
    if template_type not in DEFAULT_TEMPLATES:
        msg = f"No default template for type: {template_type}"
        raise LLMTemplateError(msg)

    return DEFAULT_TEMPLATES[template_type]
