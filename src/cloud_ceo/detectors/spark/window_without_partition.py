"""SPARK_010: Detect window functions without PARTITION BY on large datasets.

This detector identifies window functions that don't include PARTITION BY,
which forces all data into a single partition and eliminates parallelism.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class WindowWithoutPartitionDetector(RuleDetector):
    """Detector for window functions without PARTITION BY (SPARK_010).

    Window functions without PARTITION BY force all data to be processed in
    a single partition, eliminating parallelism and causing severe performance
    bottlenecks on large datasets.
    """

    def __init__(self) -> None:
        """Initialize window without partition detector."""
        super().__init__(rule_id="SPARK_010", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect window functions without PARTITION BY in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for window in ast.find_all(exp.Window):
            partition_by = window.args.get("partition_by")

            if not partition_by:
                func_name = self._extract_function_name(window)

                violation = self.create_violation(
                    message=(
                        f"Window function {func_name}() lacks PARTITION BY clause. "
                        "This forces all data into a single partition, eliminating parallelism "
                        "and causing severe performance issues on large datasets."
                    ),
                    location={
                        "line": getattr(window, "line", None),
                        "column": getattr(window, "col", None),
                    },
                    context={
                        "function": func_name,
                        "window_expression": window.sql(dialect="spark"),
                    },
                )
                violation.fix_suggestion = self.suggest_fix(violation)
                violations.append(violation)

        return violations

    def _extract_function_name(self, window: exp.Window) -> str:
        """Extract function name from window expression.

        Args:
            window: The window expression

        Returns:
            Function name or "unknown"
        """
        this = window.this

        # Try sql_name() method first - this is the most reliable in sqlglot 27.x
        if hasattr(this, 'sql_name'):
            try:
                name = this.sql_name()
                if name:
                    return name
            except Exception:
                pass

        # Fallback: extract from SQL representation
        try:
            sql = window.sql(dialect="spark")
            if "(" in sql and "OVER" in sql:
                func_part = sql.split("OVER")[0].strip()
                if "(" in func_part:
                    func_name = func_part.split("(")[0].strip()
                    if func_name:
                        return func_name
        except Exception:
            pass

        # Last resort: try direct name attribute (but this often contains the argument, not the function name)
        if hasattr(this, 'name') and this.name:
            return this.name

        return "unknown"

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for window without PARTITION BY.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        func_name = violation.context.get("function", "window_function")

        return (
            f"Add PARTITION BY clause to {func_name} window function. "
            "Example: OVER (PARTITION BY partition_column ORDER BY order_column). "
            "Choose a partition column that provides good data distribution."
        )
