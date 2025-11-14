"""Type definitions and enums for the rule engine.

This module contains enum types that are shared across the rule engine.
Separated to avoid circular imports between rule_engine.schema and config.schema.
"""

from enum import Enum


class RuleCategory(str, Enum):
    """Rule category classification."""

    UNIVERSAL = "universal"
    CONTEXT_DEPENDENT = "context-dependent"


class RuleSeverity(str, Enum):
    """Rule severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DetectionMethod(str, Enum):
    """Detection method types."""

    AST = "AST"
    PATTERN = "pattern"
    LLM = "LLM"
    QUERY_PLAN = "query_plan"
