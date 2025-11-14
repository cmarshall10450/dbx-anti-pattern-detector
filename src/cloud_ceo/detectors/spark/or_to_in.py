"""SPARK_007: Detect OR conditions that could be IN clauses.

This detector identifies multiple OR conditions comparing the same column
to different values, which should use the IN clause instead.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class OrToInDetector(RuleDetector):
    """Detector for OR-to-IN conversion opportunity (SPARK_007).

    Multiple OR conditions comparing the same column to different values
    create verbose, hard-to-read queries and may prevent optimizer optimizations.
    The IN clause is:
    - Semantically clearer and more concise
    - Easier to maintain
    - Better for optimizer to leverage partition pruning and predicate pushdown

    Example bad query:
        SELECT *
        FROM products
        WHERE category = 'Electronics'
           OR category = 'Computers'
           OR category = 'Software'

    Example fixed query:
        SELECT *
        FROM products
        WHERE category IN ('Electronics', 'Computers', 'Software')

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-syntax-qry-select-where.html
        - https://spark.apache.org/docs/latest/sql-performance-tuning.html#predicate-pushdown
    """

    # Minimum number of OR conditions to trigger detection
    MIN_OR_CONDITIONS = 3

    def __init__(self) -> None:
        """Initialize OR-to-IN detector."""
        super().__init__(rule_id="SPARK_007", severity=RuleSeverity.MEDIUM)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect OR chains that could be IN clauses in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        # Find all OR expressions, but only process top-level ones
        all_or_exprs = list(ast.find_all(exp.Or))
        top_level_ors = self._filter_top_level_ors(all_or_exprs)

        for or_expr in top_level_ors:
            or_chain = self._collect_or_chain(or_expr)

            if len(or_chain) >= self.MIN_OR_CONDITIONS:
                column_groups = self._group_by_column(or_chain)

                for column, conditions in column_groups.items():
                    if len(conditions) >= self.MIN_OR_CONDITIONS:
                        values = [cond["value"] for cond in conditions]

                        violation = self.create_violation(
                            message=(
                                f"Multiple OR conditions on column '{column}' "
                                f"({len(conditions)} conditions) should use IN clause. "
                                "IN clauses are more concise and enable better optimizer optimizations."
                            ),
                            location={
                                "line": getattr(or_expr, "line", None),
                                "column": getattr(or_expr, "col", None),
                            },
                            context={
                                "column": column,
                                "num_conditions": len(conditions),
                                "values": values[:10],  # Limit to first 10 for readability
                                "or_expression": or_expr.sql(dialect="spark"),
                            },
                        )
                        violation.fix_suggestion = self.suggest_fix(violation)
                        violations.append(violation)

        return violations

    def _filter_top_level_ors(self, all_or_exprs: list[exp.Or]) -> list[exp.Or]:
        """Filter to only top-level OR expressions (not nested within other ORs).

        Args:
            all_or_exprs: List of all OR expressions found

        Returns:
            List of top-level OR expressions only
        """
        top_level = []

        for or_expr in all_or_exprs:
            # Check if this OR is nested inside any other OR in the list
            is_nested = False
            for other_or in all_or_exprs:
                if other_or is or_expr:
                    continue
                # Check if or_expr is a descendant of other_or
                if or_expr in list(other_or.walk()):
                    is_nested = True
                    break

            if not is_nested:
                top_level.append(or_expr)

        return top_level

    def _collect_or_chain(self, or_expr: exp.Or) -> list[exp.Expression]:
        """Collect all conditions in an OR chain.

        Args:
            or_expr: The OR expression to analyze

        Returns:
            List of individual conditions in the OR chain
        """
        conditions = []

        def traverse_or(expr: exp.Expression) -> None:
            """Recursively traverse OR expressions."""
            if isinstance(expr, exp.Or):
                traverse_or(expr.left)
                traverse_or(expr.right)
            else:
                conditions.append(expr)

        traverse_or(or_expr)
        return conditions

    def _group_by_column(
        self, conditions: list[exp.Expression]
    ) -> dict[str, list[dict]]:
        """Group equality conditions by column name.

        Args:
            conditions: List of condition expressions

        Returns:
            Dictionary mapping column names to lists of condition details
        """
        groups: dict[str, list[dict]] = {}

        for condition in conditions:
            if isinstance(condition, exp.EQ):
                column_name, value = self._extract_eq_parts(condition)

                if column_name and value:
                    if column_name not in groups:
                        groups[column_name] = []

                    groups[column_name].append({
                        "value": value,
                        "expression": condition
                    })

        return groups

    def _extract_eq_parts(self, eq: exp.EQ) -> tuple[str | None, str | None]:
        """Extract column name and value from equality expression.

        Args:
            eq: The equality expression

        Returns:
            Tuple of (column_name, value) or (None, None) if not extractable
        """
        left = eq.left
        right = eq.right

        column_name = None
        value = None

        if isinstance(left, exp.Column):
            column_name = self._get_column_name(left)
            if isinstance(right, exp.Literal):
                value = str(right.this)
        elif isinstance(right, exp.Column):
            column_name = self._get_column_name(right)
            if isinstance(left, exp.Literal):
                value = str(left.this)

        return column_name, value

    def _get_column_name(self, column: exp.Column) -> str:
        """Extract column name from Column expression.

        Args:
            column: The column expression

        Returns:
            Column name (with table prefix if present)
        """
        if hasattr(column, 'table') and column.table:
            return f"{column.table}.{column.name}"
        return column.name if hasattr(column, 'name') else "unknown"

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for OR-to-IN conversion.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        column = violation.context.get("column", "column")
        values = violation.context.get("values", [])

        if values:
            value_list = ", ".join(f"'{v}'" for v in values[:5])
            if len(values) > 5:
                value_list += ", ..."

            return (
                f"Replace OR chain with IN clause. "
                f"Example: WHERE {column} IN ({value_list})"
            )

        return (
            f"Replace multiple OR conditions with IN clause. "
            f"Example: WHERE {column} IN (value1, value2, value3)"
        )
