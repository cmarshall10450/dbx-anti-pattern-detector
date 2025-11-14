"""SPARK_005: Detect inefficient GROUP BY with high cardinality columns first.

This detector identifies GROUP BY clauses that place high-cardinality columns
before low-cardinality columns, which can reduce aggregation efficiency.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class GroupByCardinalityDetector(RuleDetector):
    """Detector for GROUP BY cardinality ordering anti-pattern (SPARK_005).

    When grouping by multiple columns, the order matters for hash aggregation
    efficiency. Placing high-cardinality columns first forces Spark to create
    more intermediate groups, increasing:
    - Memory pressure from larger hash tables
    - Shuffle data volume
    - Risk of out-of-memory errors

    Reordering GROUP BY columns from low to high cardinality minimizes hash
    table size and reduces memory consumption during aggregation.

    Example bad query:
        SELECT user_id, country, COUNT(*) as cnt
        FROM user_events
        GROUP BY user_id, country

    Example fixed query:
        SELECT country, user_id, COUNT(*) as cnt
        FROM user_events
        GROUP BY country, user_id

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-syntax-qry-select-groupby.html
        - https://spark.apache.org/docs/latest/sql-performance-tuning.html#other-tips
    """

    # Common high-cardinality column patterns
    HIGH_CARDINALITY_PATTERNS = {
        "id", "uid", "uuid", "guid", "user_id", "session_id",
        "transaction_id", "order_id", "request_id", "event_id",
        "customer_id", "product_id", "account_id", "device_id",
        "timestamp", "datetime", "created_at", "updated_at",
        "email", "phone", "ip", "ip_address", "mac_address",
    }

    # Common low-cardinality column patterns
    LOW_CARDINALITY_PATTERNS = {
        "country", "region", "state", "city", "country_code",
        "status", "type", "category", "class", "level",
        "flag", "enabled", "active", "is_", "has_",
        "gender", "role", "tier", "plan", "priority",
        "year", "month", "day", "quarter", "week",
    }

    def __init__(self) -> None:
        """Initialize GROUP BY cardinality detector."""
        super().__init__(rule_id="SPARK_005", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect GROUP BY cardinality issues in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context with cardinality statistics

        Returns:
            List of violations found
        """
        violations = []

        for select in ast.find_all(exp.Select):
            group_by = select.find(exp.Group)

            if not group_by:
                continue

            columns = self._extract_group_by_columns(group_by)

            if len(columns) < 2:
                continue

            violation = self._check_column_order(columns, select, group_by, context)
            if violation:
                violations.append(violation)

        return violations

    def _extract_group_by_columns(self, group_by: exp.Group) -> list[dict]:
        """Extract column information from GROUP BY clause.

        Args:
            group_by: The GROUP BY expression

        Returns:
            List of dictionaries with column information
        """
        columns = []

        if not group_by.expressions:
            return columns

        for i, expr in enumerate(group_by.expressions):
            column_info = self._analyze_column(expr, i)
            if column_info:
                columns.append(column_info)

        return columns

    def _analyze_column(self, expr: exp.Expression, position: int) -> dict | None:
        """Analyze a GROUP BY column expression.

        Args:
            expr: Column expression
            position: Position in GROUP BY clause (0-indexed)

        Returns:
            Dictionary with column analysis or None
        """
        column_name = None

        if isinstance(expr, exp.Column):
            column_name = expr.name if hasattr(expr, 'name') else None
        elif isinstance(expr, exp.Literal):
            return None
        else:
            column = expr.find(exp.Column)
            if column and hasattr(column, 'name'):
                column_name = column.name

        if not column_name:
            return None

        cardinality_hint = self._estimate_cardinality(column_name)

        return {
            "name": column_name,
            "position": position,
            "cardinality_hint": cardinality_hint,
            "sql": expr.sql(dialect="spark"),
        }

    def _estimate_cardinality(self, column_name: str) -> str:
        """Estimate cardinality based on column name patterns.

        Args:
            column_name: Name of the column

        Returns:
            "high", "low", or "unknown"
        """
        column_lower = column_name.lower()

        for pattern in self.HIGH_CARDINALITY_PATTERNS:
            if pattern in column_lower:
                return "high"

        for pattern in self.LOW_CARDINALITY_PATTERNS:
            if pattern in column_lower:
                return "low"

        return "unknown"

    def _check_column_order(
        self,
        columns: list[dict],
        select: exp.Select,
        group_by: exp.Group,
        context: DetectionContext | None,
    ) -> Violation | None:
        """Check if GROUP BY columns are ordered inefficiently.

        Args:
            columns: List of column information
            select: The SELECT expression
            group_by: The GROUP BY expression
            context: Detection context with statistics

        Returns:
            Violation if inefficient ordering detected, None otherwise
        """
        has_high_before_low = False
        first_high_idx = None
        first_low_idx = None

        for col in columns:
            if col["cardinality_hint"] == "high" and first_high_idx is None:
                first_high_idx = col["position"]
            if col["cardinality_hint"] == "low" and first_low_idx is None:
                first_low_idx = col["position"]

        if first_high_idx is not None and first_low_idx is not None:
            if first_high_idx < first_low_idx:
                has_high_before_low = True

        if not has_high_before_low:
            return None

        high_cols = [c["name"] for c in columns if c["cardinality_hint"] == "high"]
        low_cols = [c["name"] for c in columns if c["cardinality_hint"] == "low"]

        suggested_order = [c["name"] for c in sorted(
            columns,
            key=lambda x: (0 if x["cardinality_hint"] == "low" else
                         1 if x["cardinality_hint"] == "unknown" else 2)
        )]

        return self.create_violation(
            message=(
                f"GROUP BY places high-cardinality column(s) ({', '.join(high_cols)}) "
                f"before low-cardinality column(s) ({', '.join(low_cols)}). "
                "This increases memory pressure during aggregation. Reorder GROUP BY "
                "from low to high cardinality for better performance."
            ),
            location={
                "line": getattr(group_by, "line", None),
                "column": getattr(group_by, "col", None),
            },
            context={
                "high_cardinality_columns": high_cols,
                "low_cardinality_columns": low_cols,
                "current_order": [c["name"] for c in columns],
                "suggested_order": suggested_order,
                "group_by_expression": group_by.sql(dialect="spark"),
            },
            fix_suggestion=self.suggest_fix({
                "context": {
                    "suggested_order": suggested_order,
                }
            }),
        )

    def suggest_fix(self, violation: Violation | dict) -> str:
        """Generate fix suggestion for GROUP BY ordering.

        Args:
            violation: The violation to fix (or dict with context)

        Returns:
            Fix suggestion string
        """
        if isinstance(violation, dict):
            context = violation.get("context", {})
        else:
            context = violation.context

        suggested_order = context.get("suggested_order", [])

        if suggested_order:
            column_list = ", ".join(suggested_order)
            return (
                f"Reorder GROUP BY columns from low to high cardinality. "
                f"Suggested order: GROUP BY {column_list}"
            )

        return (
            "Reorder GROUP BY columns to place low-cardinality columns first, "
            "followed by high-cardinality columns. This minimizes hash table "
            "size during aggregation."
        )
