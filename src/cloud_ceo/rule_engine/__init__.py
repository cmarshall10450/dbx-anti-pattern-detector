"""Rule engine module for Cloud CEO."""

from pathlib import Path

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.exceptions import (
    ConcurrentModificationError,
    InvalidRuleUpdateError,
    RuleManagementError,
    RuleNotFoundError,
    RuleValidationError,
)
from cloud_ceo.rule_engine.loader import (
    BuiltInRuleSource,
    DeltaRuleSource,
    RuleSource,
    YAMLRuleSource,
    initialize_rules,
)
from cloud_ceo.rule_engine.precedence import RulePrecedenceManager
from cloud_ceo.rule_engine.schema import (
    DetectionMethod,
    RuleCategory,
    RuleMetadata,
    RuleSchema,
    RuleSeverity,
)

# Built-in rules directory (moved from cloud_ceo.rules.built_in)
BUILT_IN_RULES_DIR = Path(__file__).parent.parent / "rules" / "built_in"

__all__ = [
    "RuleSchema",
    "RuleMetadata",
    "RuleCategory",
    "RuleSeverity",
    "DetectionMethod",
    "RuleSource",
    "BuiltInRuleSource",
    "YAMLRuleSource",
    "DeltaRuleSource",
    "initialize_rules",
    "RulePrecedenceManager",
    "RuleDetector",
    "DetectionContext",
    "Violation",
    "RuleManagementError",
    "RuleNotFoundError",
    "InvalidRuleUpdateError",
    "ConcurrentModificationError",
    "RuleValidationError",
    "BUILT_IN_RULES_DIR",
]
