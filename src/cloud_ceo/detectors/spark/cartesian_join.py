"""SPARK_004: Detect Cartesian joins (joins without ON conditions).

This detector identifies joins that create Cartesian products by missing
explicit join conditions, causing exponential data explosion.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class CartesianJoinDetector(RuleDetector):
    """Detector for Cartesian join anti-pattern (SPARK_004).

    Cartesian joins occur when tables are joined without explicit join conditions,
    creating a Cartesian product where every row from the first table is paired
    with every row from the second table. This causes:
    - Exponential data explosion (N * M rows)
    - Catastrophic performance degradation
    - Out-of-memory errors
    - Accidental incorrect results

    Example bad query:
        SELECT o.order_id, c.customer_name
        FROM orders o, customers c
        WHERE o.total > 1000

    Example fixed query:
        SELECT o.order_id, c.customer_name
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.total > 1000

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-syntax-qry-select-join.html
        - https://spark.apache.org/docs/latest/sql-performance-tuning.html#join-strategy-hints
    """

    def __init__(self) -> None:
        """Initialize Cartesian join detector."""
        super().__init__(rule_id="SPARK_004", severity=RuleSeverity.CRITICAL)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect Cartesian joins in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for select in ast.find_all(exp.Select):
            violations.extend(self._check_joins(select))

        return violations

    def _check_joins(self, select: exp.Select) -> list[Violation]:
        """Check for all types of Cartesian joins.

        Args:
            select: The SELECT expression to analyze

        Returns:
            List of violations found
        """
        violations = []
        all_joins = list(select.find_all(exp.Join))

        if not all_joins:
            return violations

        # In sqlglot 27+:
        # - Comma joins are converted to CROSS JOINs with on: None
        # - JOINs without ON get on: TRUE added automatically
        cross_joins_without_on = []
        other_joins_without_on = []

        for join in all_joins:
            on_clause = join.args.get("on")
            join_kind = join.args.get("kind")

            # Check if this is a Cartesian join:
            # 1. CROSS JOIN with no ON clause
            # 2. Any JOIN with ON TRUE (implicit Cartesian from missing ON)
            is_cartesian = False

            if join_kind == "CROSS" and on_clause is None:
                # Explicit CROSS JOIN or comma-separated tables
                is_cartesian = True
                cross_joins_without_on.append(join)
            elif on_clause is not None:
                # Check if ON clause is just TRUE (indicates missing ON in original SQL)
                on_sql = on_clause.sql(dialect="spark").strip().upper()
                if on_sql == "TRUE":
                    is_cartesian = True
                    other_joins_without_on.append(join)

        # If we have multiple CROSS JOINs (likely from comma-separated tables),
        # treat them as implicit comma joins for better error messages
        if len(cross_joins_without_on) > 1:
            tables = self._get_all_table_names(select)
            # Use the first cross join for location info
            first_join = cross_joins_without_on[0]
            violation = self.create_violation(
                message=(
                    f"Implicit Cartesian join detected between {len(tables)} tables "
                    f"({', '.join(tables)}) - missing ON condition. Use explicit JOIN "
                    "syntax with ON conditions to avoid unintended Cartesian product."
                ),
                location={
                    "line": getattr(first_join, "line", None),
                    "column": getattr(first_join, "col", None),
                },
                context={
                    "tables": tables,
                    "join_type": "implicit_comma",
                    "from_expression": select.find(exp.From).sql(dialect="spark") if select.find(exp.From) else "",
                },
            )
            violation.fix_suggestion = self.suggest_fix(violation)
            violations.append(violation)
        elif len(cross_joins_without_on) == 1:
            # Single CROSS JOIN - could be explicit or comma syntax (2 tables)
            # Note: We can't distinguish between "FROM a, b" and "FROM a CROSS JOIN b"
            # in sqlglot 27.x since both produce the same AST. We populate both
            # tables and left_table/right_table for compatibility.
            join = cross_joins_without_on[0]
            left_table = self._extract_table_from_from(select)
            right_table = self._extract_join_table_name(join, "this")
            all_tables = self._get_all_table_names(select)

            violation = self.create_violation(
                message=(
                    f"Cartesian join detected between '{left_table}' and '{right_table}' "
                    "(CROSS JOIN or comma-separated tables). Verify this is intentional and "
                    "add explicit JOIN conditions to avoid unintended Cartesian product."
                ),
                location={
                    "line": getattr(join, "line", None),
                    "column": getattr(join, "col", None),
                },
                context={
                    "tables": all_tables,  # For comma syntax compatibility
                    "left_table": left_table,
                    "right_table": right_table,
                    "join_type": "cross_join",
                    "join_expression": join.sql(dialect="spark"),
                },
            )
            violation.fix_suggestion = self.suggest_fix(violation)
            violations.append(violation)

        # Handle other join types with ON TRUE (missing ON clause in original SQL)
        for join in other_joins_without_on:
            # In sqlglot 27+, join type is in 'side' field (LEFT, RIGHT, FULL)
            # Regular JOIN has side: None
            join_side = join.args.get("side")
            left_table = self._extract_table_from_from(select)
            right_table = self._extract_join_table_name(join, "this")

            # Determine the join type string for the message
            if join_side is None:
                join_type_str = "JOIN"
            elif join_side == "LEFT":
                join_type_str = "LEFT JOIN"
            elif join_side == "RIGHT":
                join_type_str = "RIGHT JOIN"
            elif join_side == "FULL":
                join_type_str = "FULL JOIN"
            else:
                join_type_str = f"{join_side} JOIN"

            violation = self.create_violation(
                message=(
                    f"{join_type_str} between '{left_table}' and '{right_table}' missing "
                    "ON condition. This creates an implicit Cartesian product. "
                    "Add explicit join condition to avoid data explosion."
                ),
                location={
                    "line": getattr(join, "line", None),
                    "column": getattr(join, "col", None),
                },
                context={
                    "left_table": left_table,
                    "right_table": right_table,
                    "join_type": "join_without_on",
                    "join_expression": join.sql(dialect="spark"),
                },
            )
            violation.fix_suggestion = self.suggest_fix(violation)
            violations.append(violation)

        return violations

    def _extract_table_from_from(self, select: exp.Select) -> str:
        """Extract the main table name from the FROM clause.

        Args:
            select: The SELECT expression

        Returns:
            Table name or 'unknown'
        """
        from_clause = select.find(exp.From)
        if not from_clause:
            return "unknown"

        table = from_clause.find(exp.Table)
        return table.name if table and hasattr(table, 'name') else "unknown"

    def _get_all_table_names(self, select: exp.Select) -> list[str]:
        """Get all table names from a SELECT including FROM and all JOINs.

        Args:
            select: The SELECT expression

        Returns:
            List of table names
        """
        tables = []

        # Get table from FROM clause
        from_table = self._extract_table_from_from(select)
        if from_table != "unknown":
            tables.append(from_table)

        # Get tables from all JOINs (use 'this' arg for comma joins)
        for join in select.find_all(exp.Join):
            join_table = self._extract_join_table_name(join, "this")
            if join_table != "unknown":
                tables.append(join_table)

        return tables

    def _extract_join_table_name(self, join: exp.Expression, arg_name: str) -> str:
        """Extract table name from join expression argument.

        Args:
            join: The join expression
            arg_name: Argument name ('this' or 'expression')

        Returns:
            Table name or 'unknown'
        """
        table_expr = join.args.get(arg_name)

        if isinstance(table_expr, exp.Table):
            return table_expr.name if hasattr(table_expr, 'name') else "unknown"

        if table_expr:
            table = table_expr.find(exp.Table)
            if table and hasattr(table, 'name'):
                return table.name

        return "unknown"

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for Cartesian join.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        join_type = violation.context.get("join_type", "")
        left = violation.context.get("left_table", "table1")
        right = violation.context.get("right_table", "table2")

        if join_type == "implicit_comma":
            tables = violation.context.get("tables", [])
            if len(tables) == 2:
                return (
                    f"Use explicit JOIN syntax with ON condition. "
                    f"Example: FROM {tables[0]} JOIN {tables[1]} "
                    f"ON {tables[0]}.key = {tables[1]}.key"
                )
            return (
                "Replace comma-separated table list with explicit JOIN syntax "
                "including ON conditions for each join."
            )

        return (
            f"Add explicit join condition. "
            f"Example: JOIN {right} ON {left}.key = {right}.key"
        )
