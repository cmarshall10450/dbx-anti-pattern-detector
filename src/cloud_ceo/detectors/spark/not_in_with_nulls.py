"""SPARK_010: Detect NOT IN with subqueries or lists that could contain NULLs.

This detector identifies NOT IN usage that can return incorrect results (empty set)
when NULL values are present in the subquery result or list.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class NotInWithNullsDetector(RuleDetector):
    """Detector for NOT IN with NULL values anti-pattern (SPARK_010).

    NOT IN with subqueries returns an empty result set if the subquery contains any NULL values,
    due to three-valued logic (TRUE, FALSE, UNKNOWN) in SQL. This causes:
    - Incorrect query results (returns no rows when NULLs present)
    - Silent data loss without errors
    - Confusing behavior for developers
    - Difficult-to-debug issues

    The problem: `x NOT IN (1, 2, NULL)` evaluates to:
    - x = 1: FALSE
    - x = 2: FALSE
    - x = 3: NULL (because x != NULL is UNKNOWN)
    Result: No rows returned because WHERE NULL filters everything out.

    Example bad query:
        SELECT * FROM users
        WHERE user_id NOT IN (SELECT admin_id FROM admins)
        -- Returns no rows if admins.admin_id contains any NULLs

    Example fixed query:
        SELECT * FROM users
        WHERE NOT EXISTS (
            SELECT 1 FROM admins WHERE admins.admin_id = users.user_id
        )
        -- Correctly handles NULLs with proper three-valued logic

    Alternative fix:
        SELECT * FROM users
        WHERE user_id NOT IN (
            SELECT admin_id FROM admins WHERE admin_id IS NOT NULL
        )

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-null-semantics.html
        - https://spark.apache.org/docs/latest/sql-ref-null-semantics.html
    """

    def __init__(self) -> None:
        """Initialize NOT IN with NULLs detector."""
        super().__init__(rule_id="SPARK_010", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect NOT IN usage with potential NULL values.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for not_in in ast.find_all(exp.Not):
            in_expr = not_in.find(exp.In)
            if in_expr:
                violation = self._check_not_in(not_in, in_expr)
                if violation:
                    violations.append(violation)

        return violations

    def _check_not_in(self, not_expr: exp.Not, in_expr: exp.In) -> Violation | None:
        """Check if NOT IN could have NULL issues.

        Args:
            not_expr: The NOT expression
            in_expr: The IN expression inside the NOT

        Returns:
            Violation if NOT IN has potential NULL issues, None otherwise
        """
        column = self._get_checked_column(in_expr)
        if not column:
            return None

        in_values = in_expr.expressions if hasattr(in_expr, 'expressions') else []

        has_subquery = False
        has_literal_list = False
        subquery_info = None

        for expr in in_values:
            if isinstance(expr, exp.Subquery):
                has_subquery = True
                subquery_info = self._analyze_subquery(expr)
            elif isinstance(expr, exp.Literal):
                has_literal_list = True

        if has_subquery:
            if subquery_info and subquery_info["has_null_filter"]:
                return None

            return self.create_violation(
                message=(
                    f"NOT IN with subquery on column '{column}' can return incorrect results "
                    f"(empty set) if subquery contains NULL values. Use NOT EXISTS or add "
                    f"explicit IS NOT NULL filter to subquery."
                ),
                location={
                    "line": getattr(not_expr, "line", None),
                    "column": getattr(not_expr, "col", None),
                },
                context={
                    "column": column,
                    "has_subquery": True,
                    "subquery_sql": subquery_info["sql"] if subquery_info else "",
                    "subquery_column": subquery_info["column"] if subquery_info else "",
                    "not_in_expression": not_expr.sql(dialect="spark"),
                },
                fix_suggestion=self._suggest_subquery_fix(
                    column,
                    subquery_info["column"] if subquery_info else "column",
                    subquery_info["sql"] if subquery_info else "",
                ),
            )

        if has_literal_list and len(in_values) > 0:
            return self.create_violation(
                message=(
                    f"NOT IN with literal list on column '{column}' should be verified for NULL handling. "
                    f"If list could contain NULLs, use alternative approach (NOT EXISTS or explicit checks)."
                ),
                location={
                    "line": getattr(not_expr, "line", None),
                    "column": getattr(not_expr, "col", None),
                },
                context={
                    "column": column,
                    "has_literal_list": True,
                    "value_count": len(in_values),
                    "not_in_expression": not_expr.sql(dialect="spark")[:200],
                },
                severity=RuleSeverity.MEDIUM,
                fix_suggestion=self._suggest_literal_fix(column),
            )

        return None

    def _get_checked_column(self, in_expr: exp.In) -> str | None:
        """Get the column being checked in the IN expression.

        Args:
            in_expr: The IN expression

        Returns:
            Column name or None
        """
        if hasattr(in_expr, 'this') and isinstance(in_expr.this, exp.Column):
            return self._get_column_name(in_expr.this)
        return None

    def _analyze_subquery(self, subquery: exp.Subquery) -> dict:
        """Analyze subquery for NULL handling.

        Args:
            subquery: The subquery to analyze

        Returns:
            Dictionary with subquery analysis
        """
        select = subquery.find(exp.Select)
        if not select:
            return {
                "sql": subquery.sql(dialect="spark")[:200],
                "column": "unknown",
                "has_null_filter": False,
            }

        selected_column = self._get_subquery_selected_column(select)

        has_null_filter = self._has_is_not_null_filter(select, selected_column)

        return {
            "sql": subquery.sql(dialect="spark")[:200],
            "column": selected_column,
            "has_null_filter": has_null_filter,
        }

    def _get_subquery_selected_column(self, select: exp.Select) -> str:
        """Get the column selected by the subquery.

        Args:
            select: The SELECT in the subquery

        Returns:
            Column name or 'unknown'
        """
        if select.expressions and len(select.expressions) > 0:
            first_expr = select.expressions[0]
            if isinstance(first_expr, exp.Column):
                return self._get_column_name(first_expr)
            column = first_expr.find(exp.Column)
            if column:
                return self._get_column_name(column)

        return "unknown"

    def _has_is_not_null_filter(self, select: exp.Select, column_name: str) -> bool:
        """Check if subquery filters out NULLs with IS NOT NULL.

        Args:
            select: The SELECT in the subquery
            column_name: Column name to check for NULL filter

        Returns:
            True if subquery has IS NOT NULL filter on the column
        """
        where = select.find(exp.Where)
        if not where:
            return False

        for is_expr in where.find_all(exp.Is):
            column = is_expr.find(exp.Column)
            if column and self._get_column_name(column) == column_name:
                if is_expr.find(exp.Not) and is_expr.find(exp.Null):
                    return True

        return False

    def _get_column_name(self, column: exp.Column) -> str:
        """Get column name from column expression.

        Args:
            column: Column expression

        Returns:
            Column name
        """
        if hasattr(column, 'table') and column.table:
            return f"{column.table}.{column.name}"
        return column.name if hasattr(column, 'name') else "unknown"

    def _suggest_subquery_fix(
        self, outer_column: str, subquery_column: str, subquery_sql: str
    ) -> str:
        """Generate fix suggestion for NOT IN with subquery.

        Args:
            outer_column: Column being checked in outer query
            subquery_column: Column selected in subquery
            subquery_sql: Subquery SQL

        Returns:
            Fix suggestion string
        """
        outer_table = outer_column.split(".")[0] if "." in outer_column else "outer_table"

        return (
            f"Replace NOT IN with NOT EXISTS for NULL-safe semantics. "
            f"Example:\n"
            f"WHERE NOT EXISTS (\n"
            f"    SELECT 1 FROM (...) sub WHERE sub.{subquery_column} = {outer_table}.{outer_column}\n"
            f")\n"
            f"Alternative: Add IS NOT NULL filter to subquery:\n"
            f"WHERE {outer_column} NOT IN (\n"
            f"    SELECT {subquery_column} FROM ... WHERE {subquery_column} IS NOT NULL\n"
            f")"
        )

    def _suggest_literal_fix(self, column: str) -> str:
        """Generate fix suggestion for NOT IN with literal list.

        Args:
            column: Column being checked

        Returns:
            Fix suggestion string
        """
        return (
            f"Verify literal list does not contain NULLs. "
            f"If NULLs are possible, consider:\n"
            f"1. Remove NULLs from the list\n"
            f"2. Add explicit check: WHERE ({column} NOT IN (...) OR {column} IS NULL)\n"
            f"3. Use CASE statement for explicit NULL handling"
        )

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for NOT IN with NULLs.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        return violation.fix_suggestion or "Replace NOT IN with NOT EXISTS or add NULL filters"
