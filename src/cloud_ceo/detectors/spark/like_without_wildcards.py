"""SPARK_008: Detect LIKE pattern matching without wildcards.

This detector identifies LIKE predicates that don't use wildcards (% or _),
which should use the equality operator (=) instead for better performance.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class LikeWithoutWildcardsDetector(RuleDetector):
    """Detector for LIKE without wildcards anti-pattern (SPARK_008).

    Using LIKE for exact string matching (without % or _ wildcards) is
    inefficient because LIKE invokes pattern matching logic that is
    unnecessary for exact comparisons.
    """

    def __init__(self) -> None:
        """Initialize LIKE without wildcards detector."""
        super().__init__(rule_id="SPARK_008", severity=RuleSeverity.LOW)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect LIKE without wildcards in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for like_expr in ast.find_all(exp.Like):
            pattern = like_expr.args.get("this")
            expression = like_expr.args.get("expression")

            if not pattern:
                continue

            if isinstance(expression, exp.Literal):
                pattern_value = str(expression.this)

                if not self._contains_wildcards(pattern_value):
                    column_name = "unknown"
                    if isinstance(pattern, exp.Column):
                        column_name = pattern.name

                    violation = self.create_violation(
                        message=(
                            f"LIKE '{pattern_value}' does not use wildcards. "
                            f"Use = operator for exact matches to improve performance."
                        ),
                        location={
                            "line": getattr(like_expr, "line", None),
                            "column": getattr(like_expr, "col", None),
                        },
                        context={
                            "column": column_name,
                            "pattern": pattern_value,
                            "like_expression": like_expr.sql(dialect="spark"),
                        },
                    )
                    violation.fix_suggestion = self.suggest_fix(violation)
                    violations.append(violation)

        return violations

    def _contains_wildcards(self, pattern: str) -> bool:
        """Check if pattern contains LIKE wildcards (% or _).

        Args:
            pattern: The LIKE pattern to check

        Returns:
            True if pattern contains wildcards, False otherwise
        """
        return "%" in pattern or "_" in pattern

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for LIKE without wildcards.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        column = violation.context.get("column", "column")
        pattern = violation.context.get("pattern", "value")

        return (
            f"Replace LIKE with equality operator. "
            f"Example: {column} = '{pattern}'"
        )
