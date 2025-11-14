"""SPARK_009: Detect implicit type conversions in WHERE clauses.

This detector identifies comparisons where column types don't match literal types,
causing implicit type conversion that prevents index usage and degrades performance by 5-50x.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class ImplicitTypeConversionDetector(RuleDetector):
    """Detector for implicit type conversion anti-pattern (SPARK_009).

    Implicit type conversions occur when comparing columns to literals of mismatched types.
    This causes:
    - 5-50x performance degradation
    - Prevents index usage and partition pruning
    - Forces full table scans
    - May cause incorrect results with precision loss

    Common patterns:
    - VARCHAR/STRING column compared to NUMBER: phone_number = 5551234567
    - INT column compared to STRING: user_id = '12345'
    - DATE column compared to STRING without proper format: date_col = '2024-1-1'
    - DECIMAL precision mismatches: decimal_col = 100 (INT instead of DECIMAL)

    Example bad query:
        SELECT * FROM users WHERE phone_number = 5551234567
        -- phone_number is VARCHAR, literal is NUMBER

    Example fixed query:
        SELECT * FROM users WHERE phone_number = '5551234567'
        -- Both are STRING type

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-datatypes.html
        - https://spark.apache.org/docs/latest/sql-ref-datatypes.html
    """

    # Map sqlglot data types to high-level categories
    TYPE_CATEGORIES = {
        # String types
        "VARCHAR": "string",
        "CHAR": "string",
        "STRING": "string",
        "TEXT": "string",
        # Numeric types
        "INT": "numeric",
        "INTEGER": "numeric",
        "BIGINT": "numeric",
        "SMALLINT": "numeric",
        "TINYINT": "numeric",
        "DECIMAL": "numeric",
        "NUMERIC": "numeric",
        "FLOAT": "numeric",
        "DOUBLE": "numeric",
        "REAL": "numeric",
        # Date/time types
        "DATE": "datetime",
        "TIMESTAMP": "datetime",
        "DATETIME": "datetime",
        # Boolean
        "BOOLEAN": "boolean",
        "BOOL": "boolean",
    }

    def __init__(self) -> None:
        """Initialize implicit type conversion detector."""
        super().__init__(rule_id="SPARK_009", severity=RuleSeverity.CRITICAL)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect implicit type conversions in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context with metadata (including schema)

        Returns:
            List of violations found
        """
        violations = []

        for where in ast.find_all(exp.Where):
            violations.extend(self._check_where_clause(where, context))

        return violations

    def _check_where_clause(
        self, where: exp.Where, context: DetectionContext | None
    ) -> list[Violation]:
        """Check WHERE clause for implicit type conversions.

        Args:
            where: The WHERE clause to analyze
            context: Detection context with schema information

        Returns:
            List of violations found
        """
        violations = []

        for comparison in where.find_all(exp.Predicate):
            if self._is_comparison_operator(comparison):
                violation = self._check_type_mismatch(comparison, context)
                if violation:
                    violations.append(violation)

        return violations

    def _is_comparison_operator(self, node: exp.Expression) -> bool:
        """Check if node is a comparison operator.

        Args:
            node: AST node to check

        Returns:
            True if node is a comparison operator
        """
        return isinstance(
            node,
            (
                exp.EQ,
                exp.NEQ,
                exp.GT,
                exp.GTE,
                exp.LT,
                exp.LTE,
            ),
        )

    def _check_type_mismatch(
        self, comparison: exp.Expression, context: DetectionContext | None
    ) -> Violation | None:
        """Check if comparison has type mismatch between column and literal.

        Args:
            comparison: The comparison expression
            context: Detection context with schema

        Returns:
            Violation if type mismatch detected, None otherwise
        """
        left = comparison.this if hasattr(comparison, "this") else None
        right = comparison.expression if hasattr(comparison, "expression") else None

        if not left or not right:
            return None

        column, literal = None, None
        column_on_left = True

        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            column = left
            literal = right
            column_on_left = True
        elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
            column = right
            literal = left
            column_on_left = False
        else:
            return None

        column_type = self._infer_column_type(column, context)
        literal_type = self._infer_literal_type(literal)

        if column_type and literal_type:
            if self._types_mismatch(column_type, literal_type):
                column_name = self._get_column_name(column)
                literal_value = self._get_literal_value(literal)

                return self.create_violation(
                    message=(
                        f"Implicit type conversion detected: column '{column_name}' "
                        f"({column_type.upper()}) compared to {literal_type.upper()} literal "
                        f"({literal_value}). This prevents index usage and causes 5-50x "
                        f"performance degradation. Use matching types in comparisons."
                    ),
                    location={
                        "line": getattr(comparison, "line", None),
                        "column": getattr(comparison, "col", None),
                    },
                    context={
                        "column_name": column_name,
                        "column_type": column_type,
                        "literal_type": literal_type,
                        "literal_value": literal_value,
                        "comparison": comparison.sql(dialect="spark"),
                    },
                    fix_suggestion=self._suggest_type_fix(
                        column_name, column_type, literal_type, literal_value, column_on_left
                    ),
                )

        return None

    def _infer_column_type(
        self, column: exp.Column, context: DetectionContext | None
    ) -> str | None:
        """Infer column type from schema or context.

        Args:
            column: Column expression
            context: Detection context with schema

        Returns:
            Column type category or None
        """
        if not context or not context.schema:
            return self._guess_column_type_from_name(column)

        column_name = self._get_column_name(column)

        for col_info in context.schema.get("columns", []):
            col_full_name = col_info.get("name", "")
            if column_name.lower() in col_full_name.lower():
                col_type = col_info.get("type", "").upper()
                return self._normalize_type(col_type)

        return self._guess_column_type_from_name(column)

    def _guess_column_type_from_name(self, column: exp.Column) -> str | None:
        """Guess column type from naming conventions.

        Args:
            column: Column expression

        Returns:
            Guessed type category or None
        """
        column_name = self._get_column_name(column).lower()

        if any(
            pattern in column_name
            for pattern in [
                "phone",
                "zip",
                "postal",
                "ssn",
                "code",
                "name",
                "email",
                "address",
                "description",
            ]
        ):
            return "string"

        if any(
            pattern in column_name
            for pattern in ["_id", "count", "quantity", "amount", "price", "total"]
        ):
            return "numeric"

        if any(pattern in column_name for pattern in ["date", "time", "timestamp"]):
            return "datetime"

        return None

    def _infer_literal_type(self, literal: exp.Literal) -> str | None:
        """Infer literal type from the literal expression.

        Args:
            literal: Literal expression

        Returns:
            Literal type category
        """
        if literal.is_int:
            return "numeric"

        if literal.is_number:
            return "numeric"

        if literal.is_string:
            value = str(literal.this)
            if self._looks_like_date(value):
                return "datetime"
            if self._looks_like_number(value):
                return "string"
            return "string"

        return "unknown"

    def _looks_like_date(self, value: str) -> bool:
        """Check if string value looks like a date.

        Args:
            value: String value to check

        Returns:
            True if value looks like a date
        """
        import re

        date_patterns = [
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{4}/\d{2}/\d{2}$",
            r"^\d{2}-\d{2}-\d{4}$",
            r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
        ]

        return any(re.match(pattern, value) for pattern in date_patterns)

    def _looks_like_number(self, value: str) -> bool:
        """Check if string value looks like it should be numeric.

        Args:
            value: String value to check

        Returns:
            True if value looks like a number string (like phone, zip)
        """
        if not value:
            return False

        numeric_chars = sum(c.isdigit() for c in value)
        total_chars = len(value)

        return numeric_chars / total_chars > 0.5

    def _normalize_type(self, type_str: str) -> str:
        """Normalize database type to category.

        Args:
            type_str: Database type string

        Returns:
            Normalized type category
        """
        type_upper = type_str.upper()

        for db_type, category in self.TYPE_CATEGORIES.items():
            if db_type in type_upper:
                return category

        return "unknown"

    def _types_mismatch(self, column_type: str, literal_type: str) -> bool:
        """Check if column and literal types mismatch.

        Args:
            column_type: Column type category
            literal_type: Literal type category

        Returns:
            True if types mismatch and would cause conversion
        """
        if column_type == "unknown" or literal_type == "unknown":
            return False

        return column_type != literal_type

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

    def _get_literal_value(self, literal: exp.Literal) -> str:
        """Get literal value as string for display.

        Args:
            literal: Literal expression

        Returns:
            Literal value string
        """
        value = str(literal.this)
        if len(value) > 50:
            return value[:47] + "..."
        return value

    def _suggest_type_fix(
        self,
        column_name: str,
        column_type: str,
        literal_type: str,
        literal_value: str,
        column_on_left: bool,
    ) -> str:
        """Generate type-specific fix suggestion.

        Args:
            column_name: Name of the column
            column_type: Column type category
            literal_type: Literal type category
            literal_value: Literal value
            column_on_left: Whether column is on left side of comparison

        Returns:
            Fix suggestion string
        """
        if column_type == "string" and literal_type == "numeric":
            return (
                f"Convert numeric literal to string. "
                f"Example: {column_name} = '{literal_value}'"
            )

        if column_type == "numeric" and literal_type == "string":
            numeric_val = "".join(c for c in literal_value if c.isdigit() or c == ".")
            return (
                f"Remove quotes from numeric literal. "
                f"Example: {column_name} = {numeric_val or literal_value}"
            )

        if column_type == "datetime" and literal_type == "string":
            return (
                f"Use proper date format or CAST function. "
                f"Example: {column_name} = CAST('{literal_value}' AS DATE) "
                f"or use standard format 'YYYY-MM-DD'"
            )

        return (
            f"Ensure literal type matches column type {column_type.upper()}. "
            f"Use explicit CAST if conversion is necessary: "
            f"CAST({column_name} AS target_type) = {literal_value}"
        )

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for implicit type conversion.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        return violation.fix_suggestion or "Use matching types in comparisons"
