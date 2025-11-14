"""SPARK_012: Detect inefficient MERGE statements without change detection.

This detector identifies MERGE statements that always update matched rows without
checking if values actually changed, causing 10-56% unnecessary overhead.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class InefficientMergeDetector(RuleDetector):
    """Detector for inefficient MERGE anti-pattern (SPARK_012).

    MERGE statements that always update matched rows without checking if values changed
    cause unnecessary overhead. This results in:
    - 10-56% slower performance due to unnecessary updates
    - Excessive file rewrites in Delta Lake
    - Increased transaction log overhead
    - Higher storage costs from more file versions
    - Wasted I/O and compute resources

    Best practice: Add conditions to WHEN MATCHED to check if values actually changed
    before updating. This prevents "no-op" updates that rewrite data unnecessarily.

    Example bad query:
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        -- Always updates, even if no values changed

    Example fixed query:
        MERGE INTO target t
        USING source s ON t.id = s.id
        WHEN MATCHED AND (
            t.name != s.name OR
            t.value != s.value OR
            t.status != s.status
        ) THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        -- Only updates when values actually changed

    Alternative with hash comparison:
        WHEN MATCHED AND hash(t.*) != hash(s.*) THEN UPDATE SET *

    References:
        - https://docs.databricks.com/sql/language-manual/delta-merge-into.html
        - https://www.databricks.com/blog/2020/09/29/diving-into-delta-lake-dml-internals-update-delete-merge.html
    """

    def __init__(self) -> None:
        """Initialize inefficient MERGE detector."""
        super().__init__(rule_id="SPARK_012", severity=RuleSeverity.HIGH)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect inefficient MERGE statements in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for merge in ast.find_all(exp.Merge):
            violation = self._check_merge_efficiency(merge)
            if violation:
                violations.append(violation)

        return violations

    def _check_merge_efficiency(self, merge: exp.Merge) -> Violation | None:
        """Check if MERGE has inefficient update patterns.

        Args:
            merge: The MERGE statement

        Returns:
            Violation if MERGE is inefficient, None otherwise
        """
        target_table = self._get_target_table(merge)

        when_clauses = list(merge.find_all(exp.When))
        if not when_clauses:
            return None

        inefficient_updates = []

        for when_clause in when_clauses:
            matched_condition = when_clause.args.get("matched")
            if not matched_condition:
                continue

            condition = when_clause.args.get("condition")
            update_expr = when_clause.find(exp.Update)

            if update_expr:
                if condition is None:
                    inefficient_updates.append({
                        "when": when_clause,
                        "reason": "no_change_detection",
                        "has_condition": False,
                    })
                elif not self._has_value_change_check(condition):
                    inefficient_updates.append({
                        "when": when_clause,
                        "reason": "condition_without_change_check",
                        "has_condition": True,
                    })

        if inefficient_updates:
            primary_issue = inefficient_updates[0]
            severity = self._assess_severity(merge, inefficient_updates)

            if primary_issue["has_condition"]:
                message = (
                    f"MERGE INTO {target_table} has WHEN MATCHED with condition "
                    f"but no change detection. Add checks to verify values actually changed "
                    f"before updating (e.g., t.col != s.col) for 10-56% better performance."
                )
            else:
                message = (
                    f"MERGE INTO {target_table} has unconditional WHEN MATCHED UPDATE. "
                    f"This always updates matched rows even when no values changed, causing "
                    f"10-56% unnecessary overhead. Add change detection conditions."
                )

            return self.create_violation(
                message=message,
                location={
                    "line": getattr(merge, "line", None),
                    "column": getattr(merge, "col", None),
                },
                context={
                    "target_table": target_table,
                    "inefficient_updates": len(inefficient_updates),
                    "merge_sql": merge.sql(dialect="spark")[:400],
                },
                severity=severity,
                fix_suggestion=self._suggest_merge_fix(target_table, primary_issue),
            )

        return None

    def _get_target_table(self, merge: exp.Merge) -> str:
        """Get target table name from MERGE statement.

        Args:
            merge: The MERGE statement

        Returns:
            Target table name
        """
        into = merge.this if hasattr(merge, 'this') else None
        if isinstance(into, exp.Table):
            return into.name if hasattr(into, 'name') else "unknown"

        table = merge.find(exp.Table)
        return table.name if table and hasattr(table, 'name') else "unknown"

    def _has_value_change_check(self, condition: exp.Expression) -> bool:
        """Check if condition includes value change detection.

        Args:
            condition: The WHEN MATCHED condition

        Returns:
            True if condition checks for value changes
        """
        for neq in condition.find_all(exp.NEQ):
            left = neq.this if hasattr(neq, 'this') else None
            right = neq.expression if hasattr(neq, 'expression') else None

            if self._is_column_comparison(left, right):
                return True

        for func in condition.find_all(exp.Func):
            func_name = self._get_function_name(func)
            if func_name in ("HASH", "MD5", "SHA1", "SHA2"):
                return True

        return False

    def _is_column_comparison(
        self, left: exp.Expression | None, right: exp.Expression | None
    ) -> bool:
        """Check if expressions are comparing columns from different tables.

        Args:
            left: Left side of comparison
            right: Right side of comparison

        Returns:
            True if comparing columns from different tables
        """
        if not left or not right:
            return False

        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            left_table = left.table if hasattr(left, 'table') else None
            right_table = right.table if hasattr(right, 'table') else None

            if left_table and right_table and left_table != right_table:
                return True

        return False

    def _get_function_name(self, func: exp.Func) -> str:
        """Get function name from function expression.

        Args:
            func: Function expression

        Returns:
            Function name in uppercase
        """
        if hasattr(func, "sql_name"):
            try:
                name = func.sql_name()
                if name:
                    return name.upper()
            except Exception:
                pass

        if hasattr(func, "name"):
            return func.name.upper()

        return type(func).__name__.upper()

    def _assess_severity(self, merge: exp.Merge, inefficient_updates: list) -> RuleSeverity:
        """Assess severity based on MERGE complexity.

        Args:
            merge: The MERGE statement
            inefficient_updates: List of inefficient update patterns

        Returns:
            Appropriate severity level
        """
        if len(inefficient_updates) > 1:
            return RuleSeverity.HIGH

        return RuleSeverity.HIGH

    def _suggest_merge_fix(self, target_table: str, issue: dict) -> str:
        """Generate fix suggestion for inefficient MERGE.

        Args:
            target_table: Name of target table
            issue: Issue details

        Returns:
            Fix suggestion string
        """
        if issue["has_condition"]:
            return (
                f"Add change detection to WHEN MATCHED condition. Examples:\n"
                f"\n"
                f"1. Column-by-column comparison:\n"
                f"   WHEN MATCHED AND (\n"
                f"       t.col1 != s.col1 OR\n"
                f"       t.col2 != s.col2 OR\n"
                f"       t.col3 != s.col3\n"
                f"   ) THEN UPDATE SET *\n"
                f"\n"
                f"2. Hash-based comparison (simpler for many columns):\n"
                f"   WHEN MATCHED AND hash(t.*) != hash(s.*)\n"
                f"   THEN UPDATE SET *\n"
                f"\n"
                f"This prevents updating rows when no values actually changed, "
                f"providing 10-56% performance improvement."
            )

        return (
            f"Add change detection condition to WHEN MATCHED clause:\n"
            f"\n"
            f"1. Column-by-column comparison:\n"
            f"   WHEN MATCHED AND (\n"
            f"       {target_table}.col1 != source.col1 OR\n"
            f"       {target_table}.col2 != source.col2 OR\n"
            f"       {target_table}.col3 != source.col3\n"
            f"   ) THEN UPDATE SET *\n"
            f"\n"
            f"2. Hash-based comparison:\n"
            f"   WHEN MATCHED AND hash({target_table}.*) != hash(source.*)\n"
            f"   THEN UPDATE SET *\n"
            f"\n"
            f"This skips updates when values haven't changed, improving performance by 10-56%."
        )

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for inefficient MERGE.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        return violation.fix_suggestion or "Add change detection to WHEN MATCHED clause"
