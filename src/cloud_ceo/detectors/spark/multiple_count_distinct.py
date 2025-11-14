"""SPARK_009: Detect multiple COUNT DISTINCT in single query.

This detector identifies queries using multiple COUNT(DISTINCT) aggregations,
which prevents optimization and causes performance issues.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class MultipleCountDistinctDetector(RuleDetector):
    """Detector for multiple COUNT DISTINCT anti-pattern (SPARK_009).

    Using multiple COUNT(DISTINCT) aggregations in a single query prevents
    Spark from using partial aggregation optimizations and forces full data
    shuffles for each distinct count. This causes:
    - Separate hash tables for each COUNT(DISTINCT)
    - Multiple shuffle operations
    - Dramatically increased memory pressure
    - Significantly slower query execution

    Example bad query:
        SELECT
          COUNT(DISTINCT user_id) as unique_users,
          COUNT(DISTINCT session_id) as unique_sessions
        FROM page_views

    Example fixed query:
        SELECT
          approx_count_distinct(user_id) as unique_users,
          approx_count_distinct(session_id) as unique_sessions
        FROM page_views

    References:
        - https://docs.databricks.com/sql/language-manual/functions/count.html
        - https://docs.databricks.com/sql/language-manual/functions/approx_count_distinct.html
    """

    def __init__(self) -> None:
        """Initialize multiple COUNT DISTINCT detector."""
        super().__init__(rule_id="SPARK_009", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect multiple COUNT DISTINCT in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for select in ast.find_all(exp.Select):
            count_distincts = self._find_count_distinct_in_select(select)

            if len(count_distincts) >= 2:
                columns = [cd["column"] for cd in count_distincts]

                violation = self.create_violation(
                    message=(
                        f"Query contains {len(count_distincts)} COUNT(DISTINCT) aggregations "
                        f"on columns: {', '.join(columns)}. Multiple COUNT(DISTINCT) operations "
                        "prevent partial aggregation and force full shuffles. Consider using "
                        "approx_count_distinct() for better performance or splitting into "
                        "separate queries."
                    ),
                    location={
                        "line": getattr(select, "line", None),
                        "column": getattr(select, "col", None),
                    },
                    context={
                        "num_count_distinct": len(count_distincts),
                        "columns": columns,
                        "count_distinct_expressions": [cd["sql"] for cd in count_distincts],
                    },
                )
                violation.fix_suggestion = self.suggest_fix(violation)
                violations.append(violation)

        return violations

    def _find_count_distinct_in_select(
        self, select: exp.Select
    ) -> list[dict]:
        """Find all COUNT(DISTINCT ...) expressions in a SELECT.

        Args:
            select: The SELECT expression to analyze

        Returns:
            List of dictionaries containing COUNT DISTINCT details
        """
        count_distincts = []

        if not select.expressions:
            return count_distincts

        for expr in select.expressions:
            count_distinct_nodes = self._find_count_distinct_nodes(expr)
            count_distincts.extend(count_distinct_nodes)

        return count_distincts

    def _find_count_distinct_nodes(self, expr: exp.Expression) -> list[dict]:
        """Recursively find COUNT(DISTINCT) nodes in an expression.

        Args:
            expr: Expression to search

        Returns:
            List of COUNT DISTINCT details
        """
        results = []

        for count_func in expr.find_all(exp.Count):
            distinct_arg = count_func.args.get("this")

            if distinct_arg and isinstance(distinct_arg, exp.Distinct):
                column_name = self._extract_column_name_from_distinct(distinct_arg)

                results.append({
                    "column": column_name,
                    "sql": count_func.sql(dialect="spark"),
                    "node": count_func
                })

        return results

    def _extract_column_name_from_distinct(self, distinct: exp.Distinct) -> str:
        """Extract column name from DISTINCT expression.

        Args:
            distinct: The DISTINCT expression

        Returns:
            Column name or 'unknown'
        """
        if distinct.expressions:
            first_expr = distinct.expressions[0]

            if isinstance(first_expr, exp.Column):
                return first_expr.name if hasattr(first_expr, 'name') else "unknown"

            try:
                return first_expr.sql(dialect="spark")[:50]
            except Exception:
                return "unknown"

        return "unknown"

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for multiple COUNT DISTINCT.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        num_count = violation.context.get("num_count_distinct", 0)
        columns = violation.context.get("columns", [])

        if columns:
            approx_examples = [
                f"approx_count_distinct({col})" for col in columns[:3]
            ]
            approx_str = ", ".join(approx_examples)
            if len(columns) > 3:
                approx_str += ", ..."

            return (
                f"Replace {num_count} COUNT(DISTINCT) with approx_count_distinct() "
                f"for ~2% error tolerance and much better performance. "
                f"Example: SELECT {approx_str} FROM ... "
                "Alternatively, split into separate queries if exact counts are required."
            )

        return (
            "Consider using approx_count_distinct() instead of COUNT(DISTINCT) "
            "for better performance, or split into multiple queries if exact "
            "counts are required."
        )
