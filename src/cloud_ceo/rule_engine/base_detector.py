"""Abstract base class for all detectors.

This module provides the common interface that both AST-based detectors
(RuleDetector) and LLM-based detectors (LLMDetector) inherit from.
"""

import structlog
from abc import ABC, abstractmethod

from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class BaseDetector(ABC):
    """Abstract base class for all SQL anti-pattern detectors.

    Both RuleDetector (AST-based analysis) and LLMDetector (LLM-based analysis)
    inherit from this common interface, enabling polymorphic detector usage
    in the detection pipeline.

    The base class provides:
    - Common initialization with rule metadata
    - Shared structured logging infrastructure
    - Abstract detect() method that all subclasses must implement

    Subclasses must implement the detect() method to perform actual
    anti-pattern detection using their specific approach (AST or LLM).

    Example:
        >>> # Cannot instantiate abstract class directly
        >>> detector = BaseDetector("SPARK_001", RuleSeverity.HIGH)
        TypeError: Can't instantiate abstract class BaseDetector

        >>> # But concrete subclasses can be instantiated
        >>> class ConcreteDetector(BaseDetector):
        ...     def detect(self, sql: str, context: DetectionContext | None = None) -> list[Violation]:
        ...         return []
        >>> detector = ConcreteDetector("SPARK_001", RuleSeverity.HIGH)
        >>> detector.rule_id
        'SPARK_001'
    """

    def __init__(self, rule_id: str, severity: RuleSeverity) -> None:
        """Initialize detector with rule metadata.

        Args:
            rule_id: Unique rule identifier (e.g., "SPARK_001", "LLM_SEMANTIC_001")
            severity: Default severity level for detected violations

        Example:
            >>> detector = ConcreteDetector("SPARK_001", RuleSeverity.HIGH)
            >>> detector.rule_id
            'SPARK_001'
            >>> detector.severity
            <RuleSeverity.HIGH: 'high'>
        """
        self.rule_id = rule_id
        self.severity = severity
        self._logger = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    def detect(
        self, sql: str, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect violations in SQL query.

        This method must be implemented by all concrete detector subclasses.
        It performs the actual anti-pattern detection logic and returns
        a list of violations found (may be empty).

        Args:
            sql: SQL query string to analyze
            context: Optional detection context with metadata (table stats,
                    partition info, organizational guidelines, etc.)

        Returns:
            List of Violation objects representing detected anti-patterns.
            Returns empty list if no violations found or if detection fails.

        Example:
            >>> detector = ConcreteDetector("SPARK_001", RuleSeverity.HIGH)
            >>> violations = detector.detect("SELECT * FROM table")
            >>> len(violations)
            0
        """
        pass
