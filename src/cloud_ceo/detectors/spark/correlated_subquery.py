"""SPARK_008: Detect correlated subqueries that should be JOINs.

This detector identifies correlated subqueries that reference outer query columns,
which typically perform 10-1000x slower than equivalent JOINs.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class CorrelatedSubqueryDetector(RuleDetector):
    """Detector for correlated subquery anti-pattern (SPARK_008).

    Correlated subqueries execute the subquery once for each row in the outer query,
    resulting in catastrophic performance degradation. This causes:
    - 10-1000x slower performance than JOINs
    - N * M execution complexity (nested loop behavior)
    - Prevents query optimization and predicate pushdown
    - Massive resource consumption

    Example bad query:
        SELECT
            user_id,
            name,
            (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) as order_count
        FROM users

    Example fixed query:
        SELECT
            u.user_id,
            u.name,
            COALESCE(o.order_count, 0) as order_count
        FROM users u
        LEFT JOIN (
            SELECT user_id, COUNT(*) as order_count
            FROM orders
            GROUP BY user_id
        ) o ON u.id = o.user_id

    References:
        - https://docs.databricks.com/optimizations/subquery-performance.html
        - https://spark.apache.org/docs/latest/sql-performance-tuning.html#subquery-optimization
    """

    def __init__(self) -> None:
        """Initialize correlated subquery detector."""
        super().__init__(rule_id="SPARK_008", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect correlated subqueries in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for subquery in ast.find_all(exp.Subquery):
            if self._is_correlated(subquery):
                violation_info = self._analyze_correlation(subquery)
                if violation_info:
                    violation = self.create_violation(
                        message=(
                            f"Correlated subquery detected referencing outer query column(s): "
                            f"{', '.join(violation_info['outer_columns'])}. "
                            f"Correlated subqueries execute once per outer row (10-1000x slower). "
                            f"Rewrite as JOIN or window function for better performance."
                        ),
                        location={
                            "line": getattr(subquery, "line", None),
                            "column": getattr(subquery, "col", None),
                        },
                        context={
                            "outer_columns": violation_info["outer_columns"],
                            "subquery_type": violation_info["subquery_type"],
                            "subquery_sql": subquery.sql(dialect="spark")[:200],
                        },
                    )
                    violation.fix_suggestion = self.suggest_fix(violation)
                    violations.append(violation)

        return violations

    def _is_correlated(self, subquery: exp.Subquery) -> bool:
        """Check if subquery references outer query columns.

        Args:
            subquery: The subquery to check

        Returns:
            True if subquery is correlated with outer query
        """
        outer_columns = self._get_outer_column_references(subquery)
        return len(outer_columns) > 0

    def _get_outer_column_references(self, subquery: exp.Subquery) -> set[str]:
        """Find columns in subquery that reference outer query.

        Args:
            subquery: The subquery to analyze

        Returns:
            Set of outer column references
        """
        outer_refs = set()

        inner_select = subquery.find(exp.Select)
        if not inner_select:
            return outer_refs

        defined_tables = self._get_subquery_tables(inner_select)

        for column in inner_select.find_all(exp.Column):
            table_ref = column.table if hasattr(column, 'table') else None

            if table_ref and table_ref not in defined_tables:
                column_name = self._get_full_column_name(column)
                outer_refs.add(column_name)

        return outer_refs

    def _get_subquery_tables(self, select: exp.Select) -> set[str]:
        """Get all table names and aliases defined in subquery.

        Args:
            select: The SELECT in the subquery

        Returns:
            Set of table names/aliases available in subquery
        """
        tables = set()

        from_clause = select.find(exp.From)
        if from_clause:
            for table in from_clause.find_all(exp.Table):
                if hasattr(table, 'name'):
                    tables.add(table.name)
                if hasattr(table, 'alias') and table.alias:
                    tables.add(table.alias)

        for join in select.find_all(exp.Join):
            for table in join.find_all(exp.Table):
                if hasattr(table, 'name'):
                    tables.add(table.name)
                if hasattr(table, 'alias') and table.alias:
                    tables.add(table.alias)

        return tables

    def _analyze_correlation(self, subquery: exp.Subquery) -> dict | None:
        """Analyze correlation details for reporting.

        Args:
            subquery: The correlated subquery

        Returns:
            Dictionary with correlation details or None
        """
        outer_columns = self._get_outer_column_references(subquery)
        if not outer_columns:
            return None

        subquery_type = self._classify_subquery_type(subquery)

        return {
            "outer_columns": sorted(list(outer_columns)),
            "subquery_type": subquery_type,
        }

    def _classify_subquery_type(self, subquery: exp.Subquery) -> str:
        """Classify the type of subquery for better fix suggestions.

        Args:
            subquery: The subquery to classify

        Returns:
            Subquery type string
        """
        parent = subquery.parent

        if isinstance(parent, exp.Select):
            return "scalar_subquery_in_select"

        if self._is_in_where_clause(subquery):
            inner_select = subquery.find(exp.Select)
            if inner_select:
                for agg in inner_select.find_all(exp.AggFunc):
                    return "aggregate_subquery_in_where"
            return "subquery_in_where"

        return "correlated_subquery"

    def _get_full_column_name(self, column: exp.Column) -> str:
        """Get full column name with table prefix if available.

        Args:
            column: Column expression

        Returns:
            Full column name string
        """
        if hasattr(column, 'table') and column.table:
            return f"{column.table}.{column.name}"
        return column.name if hasattr(column, 'name') else "unknown"

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for correlated subquery.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        subquery_type = violation.context.get("subquery_type", "correlated_subquery")
        outer_columns = violation.context.get("outer_columns", [])
        col_refs = ", ".join(outer_columns[:3])

        if subquery_type == "scalar_subquery_in_select":
            return (
                f"Rewrite scalar subquery as LEFT JOIN with aggregation. "
                f"Example: LEFT JOIN (SELECT {col_refs}, aggregate_func(...) "
                f"FROM table GROUP BY {col_refs}) sub ON outer.col = sub.col"
            )

        if subquery_type == "aggregate_subquery_in_where":
            return (
                f"Rewrite as JOIN with pre-aggregated data. "
                f"Example: JOIN (SELECT {col_refs}, agg_value FROM table GROUP BY {col_refs}) "
                f"sub ON outer.col = sub.col WHERE sub.agg_value ..."
            )

        return (
            f"Rewrite correlated subquery as JOIN. "
            f"Move correlation condition (referencing {col_refs}) to ON clause. "
            f"Example: LEFT JOIN table sub ON outer.col = sub.col"
        )
