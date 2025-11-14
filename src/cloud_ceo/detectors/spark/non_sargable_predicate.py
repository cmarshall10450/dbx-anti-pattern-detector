"""SPARK_001: Detect non-sargable predicates with functions and arithmetic on filtered columns.

This detector identifies WHERE clause predicates that apply functions or arithmetic
operations to columns, preventing the optimizer from using indexes and partition
pruning, causing full table scans.
"""

from typing import Set

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class NonSargablePredicateDetector(RuleDetector):
    """Detector for non-sargable predicate anti-pattern (SPARK_001).

    Sargable = "Search ARGument ABLE" - predicates that can use indexes and
    partition pruning. Non-sargable predicates apply functions or arithmetic
    operations to columns in WHERE clauses, forcing full table scans.

    Common non-sargable patterns:
    - Date functions: YEAR(date_col) = 2024, MONTH(date_col) = 3
    - String functions: UPPER(name) = 'JOHN', LOWER(email) = 'user@example.com'
    - Type conversions: CAST(price AS INT) = 100
    - Math functions: ABS(amount) > 100, ROUND(value, 2) = 50.00
    - String operations: SUBSTRING(code, 1, 3) = 'ABC'
    - Arithmetic operations: price * 1.1 > 100, amount + 10 = 100

    These should be rewritten to isolate the column on one side of the comparison.

    Example bad query:
        SELECT * FROM orders WHERE YEAR(order_date) = 2024

    Example fixed query:
        SELECT * FROM orders
        WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01'
    """

    # Functions that commonly cause non-sargable predicates
    NON_SARGABLE_FUNCTIONS: Set[str] = {
        # Date/time functions
        "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND",
        "DATE", "DATE_TRUNC", "TIMESTAMP_TRUNC", "TO_DATE", "EXTRACT",
        "DAYOFWEEK", "DAYOFMONTH", "DAYOFYEAR", "WEEKOFYEAR",
        "QUARTER", "LAST_DAY", "NEXT_DAY", "DATE_FORMAT",
        "FROM_UNIXTIME", "UNIX_TIMESTAMP", "TO_TIMESTAMP",
        # String functions
        "UPPER", "LOWER", "TRIM", "LTRIM", "RTRIM",
        "SUBSTRING", "SUBSTR", "LEFT", "RIGHT",
        "CONCAT", "CONCAT_WS", "LENGTH", "CHAR_LENGTH",
        "REPLACE", "TRANSLATE", "REVERSE", "REPEAT",
        "LPAD", "RPAD", "SPLIT", "REGEXP_EXTRACT",
        # Type conversion
        "CAST", "CONVERT", "TRY_CAST",
        # Math functions
        "ABS", "ROUND", "FLOOR", "CEIL", "CEILING",
        "SQRT", "POWER", "POW", "EXP", "LN", "LOG",
        "SIN", "COS", "TAN", "ASIN", "ACOS", "ATAN",
        # Other transformations
        "COALESCE", "NVL", "IFNULL", "NULLIF",
    }

    def __init__(self) -> None:
        """Initialize non-sargable predicate detector."""
        super().__init__(rule_id="SPARK_001", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect non-sargable predicates in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        # Find all WHERE clauses in the query
        for where in ast.find_all(exp.Where):
            violations.extend(self._check_where_clause(where))

        return violations

    def _check_where_clause(self, where: exp.Where) -> list[Violation]:
        """Check WHERE clause for non-sargable predicates.

        Args:
            where: The WHERE clause to analyze

        Returns:
            List of violations found in this WHERE clause
        """
        violations = []

        # Find all comparison operations in the WHERE clause
        for comparison in where.find_all(exp.Predicate):
            # Check if this is a comparison we care about
            if self._is_comparison_operator(comparison):
                violation = self._check_predicate(comparison)
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
                exp.EQ,  # =
                exp.NEQ,  # !=, <>
                exp.GT,  # >
                exp.GTE,  # >=
                exp.LT,  # <
                exp.LTE,  # <=
                exp.Like,  # LIKE
                exp.ILike,  # ILIKE
                exp.In,  # IN
            ),
        )

    def _check_predicate(self, predicate: exp.Expression) -> Violation | None:
        """Check if a predicate is non-sargable.

        Args:
            predicate: The predicate expression to check

        Returns:
            Violation if non-sargable pattern detected, None otherwise
        """
        # Get the left side of the comparison
        left = predicate.this if hasattr(predicate, "this") else None
        if not left:
            return None

        # Check if left side contains a function applied to a column
        func_info = self._find_function_on_column(left)
        if func_info:
            func_name, column_name = func_info
            display_name = self._normalize_function_name_for_display(func_name)

            return self.create_violation(
                message=(
                    f"Non-sargable predicate: {display_name}({column_name}) in WHERE clause. "
                    f"Functions on filtered columns prevent index usage and partition "
                    f"pruning, causing full table scans. Rewrite to apply function to "
                    f"literal values instead of columns."
                ),
                location={
                    "line": getattr(predicate, "line", None),
                    "column": getattr(predicate, "col", None),
                },
                context={
                    "function": display_name,  # Use normalized name for consistency
                    "column": column_name,
                    "predicate_expression": predicate.sql(dialect="spark"),
                    "predicate_type": type(predicate).__name__,
                },
                fix_suggestion=self._suggest_fix_for_function(display_name, column_name, predicate),
            )

        # Check if left side contains an arithmetic operation on a column
        arith_info = self._find_arithmetic_on_column(left)
        if arith_info:
            operation_type, column_name, expr_sql = arith_info

            return self.create_violation(
                message=(
                    f"Non-sargable predicate: {expr_sql} in WHERE clause. "
                    f"Arithmetic operations on filtered columns prevent index usage and "
                    f"partition pruning, causing full table scans. Rewrite to isolate the column."
                ),
                location={
                    "line": getattr(predicate, "line", None),
                    "column": getattr(predicate, "col", None),
                },
                context={
                    "operation": operation_type,
                    "column": column_name,
                    "predicate_expression": predicate.sql(dialect="spark"),
                    "predicate_type": type(predicate).__name__,
                },
                fix_suggestion=self._suggest_fix_for_arithmetic(operation_type, column_name, predicate),
            )

        return None

    def _find_function_on_column(self, expr: exp.Expression) -> tuple[str, str] | None:
        """Find if expression contains a function applied to a column.

        Args:
            expr: Expression to analyze

        Returns:
            Tuple of (function_name, column_name) if found, None otherwise
        """
        # Check if this expression itself is a problematic function
        if isinstance(expr, exp.Func):
            func_name = self._get_function_name(expr)
            if func_name in self.NON_SARGABLE_FUNCTIONS:
                # Look for a column inside this function
                column = expr.find(exp.Column)
                if column:
                    return (func_name, column.name)

        # Recursively check child expressions
        for child in expr.iter_expressions():
            result = self._find_function_on_column(child)
            if result:
                return result

        return None

    def _find_arithmetic_on_column(self, expr: exp.Expression) -> tuple[str, str, str] | None:
        """Find if expression contains an arithmetic operation on a column.

        Args:
            expr: Expression to analyze

        Returns:
            Tuple of (operation_type, column_name, expression_sql) if found, None otherwise
        """
        # Check if this expression itself is an arithmetic operation
        if isinstance(expr, (exp.Mul, exp.Div, exp.Add, exp.Sub, exp.Mod, exp.Pow)):
            # Look for a column inside this operation
            column = expr.find(exp.Column)
            if column:
                operation_type = type(expr).__name__
                expr_sql = expr.sql(dialect="spark")
                return (operation_type, column.name, expr_sql)

        # Recursively check child expressions, but skip if we're inside a function
        # to avoid double-reporting (function detector takes precedence)
        for child in expr.iter_expressions():
            if isinstance(child, exp.Func):
                continue
            result = self._find_arithmetic_on_column(child)
            if result:
                return result

        return None

    def _get_function_name(self, func: exp.Func) -> str:
        """Extract function name from function expression.

        Args:
            func: Function expression

        Returns:
            Function name in uppercase
        """
        # Try sql_name() method first (for built-in functions)
        if hasattr(func, "sql_name"):
            try:
                name = func.sql_name()
                if name:
                    return name.upper()
            except Exception:
                pass

        # Try name attribute
        if hasattr(func, "name"):
            return func.name.upper()

        # Fallback to class name
        class_name = type(func).__name__
        # Remove "Func" suffix if present (e.g., "CastFunc" -> "CAST")
        if class_name.endswith("Func"):
            class_name = class_name[:-4]
        return class_name.upper()

    def _normalize_function_name_for_display(self, func_name: str) -> str:
        """Normalize internal sqlglot function names to user-friendly names.

        Args:
            func_name: Internal function name from sqlglot

        Returns:
            User-friendly function name for display
        """
        # Map internal sqlglot names to user-friendly names
        name_mapping = {
            "TIMESTAMP_TRUNC": "DATE_TRUNC",
        }
        return name_mapping.get(func_name, func_name)

    def _suggest_fix_for_function(
        self, func_name: str, column_name: str, predicate: exp.Expression
    ) -> str:
        """Generate context-aware fix suggestion.

        Args:
            func_name: Name of the function
            column_name: Name of the column
            predicate: The full predicate expression

        Returns:
            Specific fix suggestion based on function type
        """
        func_upper = func_name.upper()

        # Date function fixes
        if func_upper in ("YEAR", "MONTH", "DAY", "DATE_TRUNC", "TIMESTAMP_TRUNC", "EXTRACT"):
            return (
                f"Replace {func_name}({column_name}) with range predicate. "
                f"Example: {column_name} >= '2024-01-01' AND {column_name} < '2025-01-01'"
            )

        # String function fixes
        if func_upper in ("UPPER", "LOWER"):
            return (
                f"Instead of {func_name}({column_name}), store data in normalized form "
                f"or use case-insensitive comparison if supported. "
                f"Example: {column_name} = 'normalized_value'"
            )

        if func_upper in ("TRIM", "LTRIM", "RTRIM"):
            return (
                f"Store data without leading/trailing spaces, or compare with trimmed literal. "
                f"Example: {column_name} = 'trimmed_value'"
            )

        if func_upper in ("SUBSTRING", "SUBSTR", "LEFT", "RIGHT"):
            return (
                f"Use wildcard or range predicates instead of {func_name}. "
                f"Example: {column_name} LIKE 'prefix%' or use range comparison"
            )

        # Cast/conversion fixes
        if func_upper in ("CAST", "CONVERT", "TRY_CAST"):
            return (
                f"Apply {func_name} to literal value instead of column. "
                f"Example: {column_name} >= CAST('100' AS expected_type)"
            )

        # Math function fixes
        if func_upper in ("ABS", "ROUND", "FLOOR", "CEIL", "CEILING"):
            return (
                f"Use range predicates instead of {func_name}({column_name}). "
                f"Example: {column_name} BETWEEN -value AND value"
            )

        # Generic fix
        return (
            f"Rewrite predicate to avoid applying {func_name} to {column_name}. "
            f"Consider: 1) Range predicates, 2) Pre-computed columns, "
            f"3) Applying function to literal values"
        )

    def _suggest_fix_for_arithmetic(
        self, operation_type: str, column_name: str, predicate: exp.Expression
    ) -> str:
        """Generate context-aware fix suggestion for arithmetic operations.

        Args:
            operation_type: Type of the arithmetic operation (Mul, Div, Add, Sub, Mod, Pow)
            column_name: Name of the column
            predicate: The full predicate expression

        Returns:
            Specific fix suggestion based on operation type
        """
        # Get the right side of the comparison for better suggestions
        right_value = predicate.expression.sql(dialect="spark") if hasattr(predicate, "expression") else "value"

        if operation_type == "Mul":
            return (
                f"Rewrite to isolate column on one side. "
                f"Example: {column_name} > {right_value} / multiplier"
            )

        if operation_type == "Div":
            return (
                f"Rewrite to isolate column on one side. "
                f"Example: {column_name} > {right_value} * divisor"
            )

        if operation_type == "Add":
            return (
                f"Rewrite to isolate column on one side. "
                f"Example: {column_name} > {right_value} - addend"
            )

        if operation_type == "Sub":
            return (
                f"Rewrite to isolate column on one side. "
                f"Example: {column_name} > {right_value} + subtrahend"
            )

        if operation_type == "Mod":
            return (
                f"Rewrite to avoid modulo on column. "
                f"Consider using a pre-computed column or different approach. "
                f"Example: Create computed column for MOD({column_name}, n)"
            )

        if operation_type == "Pow":
            return (
                f"Rewrite to isolate column on one side. "
                f"Example: {column_name} > POW({right_value}, 1/exponent)"
            )

        # Generic fix
        return (
            f"Rewrite to isolate {column_name} on one side of the comparison. "
            f"Apply inverse arithmetic operation to the literal value instead."
        )

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for non-sargable predicate.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        # This is already set in _check_predicate via fix_suggestion parameter
        # This method exists for consistency with base class
        return violation.fix_suggestion or "Rewrite predicate to be sargable"
