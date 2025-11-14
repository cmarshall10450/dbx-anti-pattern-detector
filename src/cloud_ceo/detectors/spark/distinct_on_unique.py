"""SPARK_006: Detect DISTINCT on already unique columns.

This detector identifies DISTINCT operations on columns that are already
guaranteed to be unique, which adds unnecessary overhead.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class DistinctOnUniqueDetector(RuleDetector):
    """Detector for DISTINCT on unique columns anti-pattern (SPARK_006).

    Using DISTINCT on columns that are already guaranteed to be unique (such as
    primary keys or unique constraints) adds unnecessary overhead. The DISTINCT
    operation requires:
    - Sorting or hashing to eliminate duplicates
    - Additional memory for deduplication structures
    - Extra CPU cycles for comparison operations

    When no duplicates exist, this processing is wasted.

    Example bad query:
        SELECT DISTINCT user_id
        FROM users
        WHERE created_date >= '2024-01-01'

    Example fixed query:
        SELECT user_id
        FROM users
        WHERE created_date >= '2024-01-01'

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-syntax-qry-select-distinct.html
        - https://spark.apache.org/docs/latest/sql-performance-tuning.html

    Note:
        This detector requires catalog metadata to identify unique columns.
        Without catalog access, it will use heuristics based on column names.
    """

    # Common unique column name patterns
    UNIQUE_COLUMN_PATTERNS = {
        "id", "uuid", "guid",
        "user_id", "customer_id", "account_id", "order_id",
        "transaction_id", "session_id", "request_id", "event_id",
        "product_id", "item_id", "device_id", "merchant_id",
        "invoice_id", "payment_id", "subscription_id",
    }

    def __init__(self) -> None:
        """Initialize DISTINCT on unique detector."""
        super().__init__(rule_id="SPARK_006", severity=RuleSeverity.MEDIUM)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect DISTINCT on unique columns in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context with metadata

        Returns:
            List of violations found
        """
        violations = []

        for select in ast.find_all(exp.Select):
            if not select.args.get("distinct"):
                continue

            violations.extend(self._check_distinct_select(select, context))

        return violations

    def _check_distinct_select(
        self,
        select: exp.Select,
        context: DetectionContext | None
    ) -> list[Violation]:
        """Check a SELECT DISTINCT for unique columns.

        Args:
            select: The SELECT expression with DISTINCT
            context: Detection context with metadata

        Returns:
            List of violations found
        """
        violations = []

        if not select.expressions:
            return violations

        selected_columns = self._extract_selected_columns(select)

        if not selected_columns:
            return violations

        table_name = self._extract_table_name(select)

        unique_columns = self._get_unique_columns(table_name, context)

        distinct_on_unique = []
        for col in selected_columns:
            if self._is_unique_column(col, unique_columns):
                distinct_on_unique.append(col)

        if distinct_on_unique:
            violation = self._create_distinct_violation(
                select, table_name, distinct_on_unique
            )
            violations.append(violation)

        return violations

    def _extract_selected_columns(self, select: exp.Select) -> list[str]:
        """Extract column names from SELECT list.

        Args:
            select: The SELECT expression

        Returns:
            List of column names
        """
        columns = []

        for expr in select.expressions:
            if isinstance(expr, exp.Column):
                if hasattr(expr, 'name'):
                    columns.append(expr.name)
            elif isinstance(expr, exp.Alias):
                aliased_expr = expr.this
                if isinstance(aliased_expr, exp.Column) and hasattr(aliased_expr, 'name'):
                    columns.append(aliased_expr.name)

        return columns

    def _get_unique_columns(
        self,
        table_name: str,
        context: DetectionContext | None
    ) -> set[str]:
        """Get unique columns for a table.

        Args:
            table_name: Name of the table
            context: Detection context with metadata

        Returns:
            Set of unique column names
        """
        unique_cols = set()

        if context and context.metadata:
            catalog_unique = context.get_unique_columns(table_name)
            if catalog_unique:
                unique_cols.update(col.lower() for col in catalog_unique)

        unique_cols.update(self._infer_unique_columns())

        return unique_cols

    def _infer_unique_columns(self) -> set[str]:
        """Infer likely unique columns from patterns.

        Returns:
            Set of column name patterns that suggest uniqueness
        """
        return {pattern.lower() for pattern in self.UNIQUE_COLUMN_PATTERNS}

    def _is_unique_column(self, column_name: str, unique_columns: set[str]) -> bool:
        """Check if a column is unique.

        Args:
            column_name: Name of the column
            unique_columns: Set of known unique column names/patterns

        Returns:
            True if column is unique
        """
        column_lower = column_name.lower()

        if column_lower in unique_columns:
            return True

        for pattern in unique_columns:
            if pattern in column_lower and column_lower.endswith(pattern):
                return True

        return False

    def _create_distinct_violation(
        self,
        select: exp.Select,
        table_name: str,
        unique_columns: list[str]
    ) -> Violation:
        """Create violation for DISTINCT on unique columns.

        Args:
            select: The SELECT expression
            table_name: Name of the table
            unique_columns: Unique columns in the SELECT

        Returns:
            Violation object
        """
        return self.create_violation(
            message=(
                f"DISTINCT on unique column(s) '{', '.join(unique_columns)}' from "
                f"table '{table_name}' is unnecessary. These columns are already unique, "
                "so DISTINCT adds overhead without removing duplicates. Remove DISTINCT "
                "to improve performance."
            ),
            location={
                "line": getattr(select, "line", None),
                "column": getattr(select, "col", None),
            },
            context={
                "table": table_name,
                "unique_columns": unique_columns,
                "select_expression": select.sql(dialect="spark"),
            },
            fix_suggestion=self.suggest_fix({
                "context": {
                    "table": table_name,
                    "unique_columns": unique_columns,
                }
            }),
        )

    def suggest_fix(self, violation: Violation | dict) -> str:
        """Generate fix suggestion for DISTINCT on unique.

        Args:
            violation: The violation to fix (or dict with context)

        Returns:
            Fix suggestion string
        """
        if isinstance(violation, dict):
            context = violation.get("context", {})
        else:
            context = violation.context

        table = context.get("table", "table")
        unique_cols = context.get("unique_columns", [])

        if unique_cols:
            col_list = ", ".join(unique_cols)
            return (
                f"Remove DISTINCT keyword. Columns {col_list} are already unique. "
                f"Example: SELECT {col_list} FROM {table} WHERE ..."
            )

        return (
            "Remove DISTINCT keyword when querying unique columns to "
            "avoid unnecessary deduplication overhead."
        )
