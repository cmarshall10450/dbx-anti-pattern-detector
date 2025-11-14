"""Pydantic models for rule schema validation.

This module defines the strongly-typed rule schema that ensures consistency
across YAML and Delta table sources.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cloud_ceo.rule_engine.types import DetectionMethod, RuleCategory, RuleSeverity

if TYPE_CHECKING:
    from sqlglot import exp

# Security: Allowed module prefixes for detector classes
# This prevents arbitrary code execution by restricting which modules can be loaded
# Only built-in detectors are allowed
ALLOWED_DETECTOR_MODULES = [
    "cloud_ceo.detectors.",
]


class RuleMetadata(BaseModel):
    """Metadata associated with a rule."""

    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    created_by: str | None = Field(None, description="User or system that created the rule")
    approved_by: str | None = Field(None, description="User who approved the rule")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    version: str = Field(default="1.0", description="Schema version for backward compatibility")


class RuleSchema(BaseModel):
    """Complete rule schema with validation.

    Rules define configuration for detection logic implemented in detector classes.
    Each rule references a Python detector class that implements the actual
    AST-based detection logic.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rule_id": "SPARK_001",
                "name": "Avoid SELECT * in production queries",
                "category": "universal",
                "severity": "high",
                "detection_method": "AST",
                "detector_class": "cloud_ceo.detectors.spark:SelectStarDetector",
                "explanation": "Using SELECT * can cause performance issues and breaks when schema changes. Always specify explicit column names for maintainability and performance.",
                "dependencies": ["spark_catalog"],
                "custom_thresholds": {},
                "custom_message": None,
                "metadata": {
                    "confidence_score": 0.95,
                    "created_by": "system",
                    "approved_by": "data_team",
                    "version": "1.0",
                },
            }
        }
    )

    rule_id: str = Field(
        ...,
        pattern=r"^[A-Z]+_\d{3}$",
        description="Rule identifier in format PREFIX_NNN (e.g., SPARK_001)",
    )
    name: str = Field(..., min_length=10, max_length=200, description="Rule name")
    category: RuleCategory = Field(..., description="Rule category")
    severity: RuleSeverity = Field(..., description="Rule severity level")
    detection_method: DetectionMethod = Field(..., description="Detection method")
    detector_class: str = Field(
        ...,
        description="Fully qualified Python class in module:ClassName format (e.g., cloud_ceo.detectors.spark:SelectStarDetector)",
        pattern=r"^[\w\.]+:[A-Z][\w]+$",
    )
    enabled: bool = Field(
        default=True,
        description="Enable/disable this rule",
    )
    explanation: str = Field(..., min_length=50, description="Rule explanation (minimum 50 chars)")
    dependencies: list[str] = Field(
        default_factory=list, description="Required data sources for context-dependent rules"
    )
    custom_thresholds: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific configuration parameters",
    )
    custom_message: str | None = Field(
        default=None,
        description="Custom violation message override",
    )
    metadata: RuleMetadata = Field(..., description="Rule metadata")

    @field_validator("rule_id")
    @classmethod
    def validate_rule_id(cls, v: str) -> str:
        """Ensure rule_id is non-empty and follows format."""
        if not v or not v.strip():
            raise ValueError("rule_id cannot be empty")
        return v

    @field_validator("detector_class")
    @classmethod
    def validate_detector_class(cls, v: str, info) -> str:
        """Ensure detector_class follows module:ClassName format and passes security checks."""
        if not v or not v.strip():
            raise ValueError("detector_class cannot be empty")

        if ":" not in v:
            raise ValueError(
                "detector_class must be in module:ClassName format "
                "(e.g., cloud_ceo.detectors.spark:SelectStarDetector)"
            )

        parts = v.split(":")
        if len(parts) != 2:
            raise ValueError(
                "detector_class must have exactly one colon separator "
                "(e.g., module.path:ClassName)"
            )

        module_path, class_name = parts
        if not module_path or not class_name:
            raise ValueError("Both module path and class name must be non-empty")

        if not class_name[0].isupper():
            raise ValueError(
                f"Class name '{class_name}' must start with uppercase letter"
            )

        # Security: Prevent arbitrary code execution by restricting module paths
        # Get allowed modules from validation context (if config passed) or use default
        allowed_modules = ALLOWED_DETECTOR_MODULES
        if info.context and 'config' in info.context:
            config = info.context['config']
            if hasattr(config, 'allowed_detector_modules'):
                allowed_modules = config.allowed_detector_modules

        # Normalize module_path with trailing dot for prefix matching
        # This ensures "myorg.detectors" matches both "myorg.detectors" and "myorg.detectors."
        module_path_normalized = module_path + "."

        # Normalize allowed prefixes to have trailing dots
        allowed_normalized = [
            allowed if allowed.endswith(".") else allowed + "."
            for allowed in allowed_modules
        ]

        if not any(module_path_normalized.startswith(allowed) for allowed in allowed_normalized):
            raise ValueError(
                f"Detector class '{v}' is not from an allowed module. "
                f"Add module prefix to 'allowed_detector_modules' in config. "
                f"Allowed prefixes: {', '.join(allowed_modules)}"
            )

        return v

    def model_dump_json_schema(self) -> dict[str, Any]:
        """Export the model as JSON Schema."""
        return self.model_json_schema()
