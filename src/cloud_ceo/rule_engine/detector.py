"""Base classes for rule detection system.

This module provides the abstract base classes and data structures for
implementing SQL anti-pattern detectors using AST-based analysis.
"""

import logging
from abc import abstractmethod
from typing import Any

from sqlglot import exp

from cloud_ceo.rule_engine.base_detector import BaseDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity

logger = logging.getLogger(__name__)


class RuleDetector(BaseDetector):
    """Abstract base class for AST-based rule detectors.

    Detectors implement SQL anti-pattern detection logic using AST-based
    analysis with sqlglot. Each detector corresponds to one or more rules
    and is responsible for:

    1. Traversing the AST to find anti-patterns
    2. Creating Violation objects with location and context
    3. Suggesting fixes for detected violations

    Subclasses must implement the _detect_violations() method which contains
    the actual detection logic.

    Note:
        SQL parsing is handled by the caller (typically core.analyze()).
        Detectors receive pre-parsed AST to avoid duplicate parsing and
        ensure consistent dialect handling.
    """

    def detect(self, ast: exp.Expression, context: DetectionContext | None = None) -> list[Violation]:
        """Detect violations in parsed SQL AST.

        Args:
            ast: Pre-parsed SQL Abstract Syntax Tree
            context: Optional detection context with metadata

        Returns:
            List of violations found in the AST.

        Note:
            This method expects a pre-parsed AST. SQL parsing should be
            done once by the caller to avoid duplicate parsing and ensure
            consistent dialect usage across all detectors.
        """
        if ast is None:
            return []

        try:
            return self._detect_violations(ast, context)
        except Exception as e:
            self._logger.error(
                "Unexpected error during violation detection",
                exc_info=True,
                extra={"rule_id": self.rule_id, "error": str(e)}
            )
            return []

    @abstractmethod
    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect violations in parsed SQL AST.

        This method must be implemented by subclasses to perform the actual
        anti-pattern detection logic.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context with metadata

        Returns:
            List of violations found in the query
        """
        pass

    def _extract_table_name(self, select: exp.Select) -> str:
        """Extract table name from SELECT statement.

        Args:
            select: The SELECT expression to analyze

        Returns:
            Table name or "unknown" if not found
        """
        from_clause = select.find(exp.From)
        if not from_clause:
            return "unknown"

        table = from_clause.find(exp.Table)
        return table.name if table else "unknown"

    def _is_in_cte(self, node: exp.Expression) -> bool:
        """Check if node is inside a CTE (Common Table Expression).

        Args:
            node: AST node to check

        Returns:
            True if node is within a CTE
        """
        parent = node.parent
        while parent:
            if isinstance(parent, exp.CTE):
                return True
            parent = parent.parent
        return False

    def _is_in_where_clause(self, node: exp.Expression) -> bool:
        """Check if node is inside WHERE clause.

        Args:
            node: AST node to check

        Returns:
            True if node is within a WHERE clause
        """
        parent = node.parent
        while parent:
            if isinstance(parent, exp.Where):
                return True
            parent = parent.parent
        return False

    def suggest_fix(self, violation: Violation) -> str | None:
        """Generate fix suggestion for a violation.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string or None if no fix available
        """
        return None

    def create_violation(
        self,
        message: str,
        location: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        fix_suggestion: str | None = None,
        severity: RuleSeverity | None = None,
    ) -> Violation:
        """Helper method to create a violation with detector defaults.

        Args:
            message: Human-readable description
            location: Location information (line, column, etc.)
            context: Additional context about the violation
            fix_suggestion: Suggested fix for the violation
            severity: Override default severity

        Returns:
            Configured Violation object
        """
        return Violation(
            rule_id=self.rule_id,
            severity=severity or self.severity,
            message=message,
            location=location or {},
            context=context or {},
            fix_suggestion=fix_suggestion,
        )
