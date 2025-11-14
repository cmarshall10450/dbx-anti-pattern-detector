"""SPARK_002: Detect SELECT * usage.

This detector identifies queries using SELECT * which retrieves all columns
from a table, even when only a subset is needed downstream.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class SelectStarDetector(RuleDetector):
    """Detector for SELECT * anti-pattern (SPARK_002).

    SELECT * causes several issues:
    - Wastes I/O bandwidth, memory, and network resources
    - Breaks when schema changes
    - Makes queries harder to maintain
    - Prevents column pruning optimizations
    """

    def __init__(self) -> None:
        """Initialize SELECT * detector."""
        super().__init__(rule_id="SPARK_002", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect SELECT * in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for select in ast.find_all(exp.Select):
            # Only check stars in this SELECT's expressions, not nested SELECTs
            # Check if this SELECT has a star in its projection list
            has_star = False
            if select.expressions:
                for expr in select.expressions:
                    if isinstance(expr, exp.Star):
                        has_star = True
                        star = expr
                        break

            if has_star:
                table_name = self._extract_table_name(select)
                severity = self._assess_severity(select)

                violation = self.create_violation(
                    message=(
                        f"SELECT * retrieves all columns from {table_name}. "
                        "Explicitly specify only required columns for better "
                        "performance and maintainability."
                    ),
                    location={
                        "line": getattr(star, "line", None),
                        "column": getattr(star, "col", None),
                    },
                    context={
                        "table": table_name,
                        "select_expression": select.sql(dialect="spark"),
                    },
                    severity=severity,
                )
                violation.fix_suggestion = self.suggest_fix(violation)
                violations.append(violation)

        return violations

    def _assess_severity(self, select: exp.Select) -> RuleSeverity:
        """Assess severity based on query context.

        SELECT * in a CTE that's immediately projected is lower severity.
        SELECT * in final output is higher severity.

        Args:
            select: The SELECT expression to assess

        Returns:
            Appropriate severity level
        """
        if self._is_in_cte(select):
            return RuleSeverity.MEDIUM

        return RuleSeverity.HIGH

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for SELECT *.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        table = violation.context.get("table", "table")
        return (
            f"Replace SELECT * with explicit column list. "
            f"Example: SELECT col1, col2, col3 FROM {table}"
        )
