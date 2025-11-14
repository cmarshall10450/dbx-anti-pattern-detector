"""SPARK_011: Detect UNION without ALL where UNION ALL would be more efficient.

This detector identifies UNION usage (which removes duplicates) where UNION ALL
would provide 30-100% better performance when duplicates are impossible or acceptable.
"""

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity


class UnionWithoutAllDetector(RuleDetector):
    """Detector for UNION without ALL anti-pattern (SPARK_011).

    UNION automatically removes duplicates by performing a DISTINCT operation,
    which requires sorting and deduplication. This causes:
    - 30-100% slower performance compared to UNION ALL
    - Additional shuffle operations in Spark
    - Higher memory consumption
    - Unnecessary overhead when duplicates are impossible or acceptable

    UNION should only be used when:
    1. Duplicate removal is specifically required
    2. Duplicates are actually possible in the result sets
    3. The performance cost is justified

    Example bad query:
        SELECT id, name FROM active_users
        UNION
        SELECT id, name FROM pending_users
        -- If IDs are unique across tables, UNION ALL is better

    Example fixed query:
        SELECT id, name FROM active_users
        UNION ALL
        SELECT id, name FROM pending_users
        -- 30-100% faster, no deduplication overhead

    When to use UNION ALL:
    - Combining data from different time periods (partitions)
    - Combining mutually exclusive datasets
    - When duplicates are acceptable for the use case
    - When you know duplicates don't exist

    References:
        - https://docs.databricks.com/sql/language-manual/sql-ref-syntax-qry-select-setops.html
        - https://spark.apache.org/docs/latest/sql-performance-tuning.html#set-operations
    """

    def __init__(self) -> None:
        """Initialize UNION without ALL detector."""
        super().__init__(rule_id="SPARK_011", severity=RuleSeverity.MEDIUM)

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect UNION without ALL in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        for union in ast.find_all(exp.Union):
            if not self._is_union_all(union):
                union_info = self._analyze_union(union)
                severity = self._assess_severity(union, union_info)

                violation = self.create_violation(
                    message=(
                        f"UNION without ALL detected between {union_info['branch_count']} queries. "
                        f"UNION performs expensive deduplication. If duplicates are impossible "
                        f"or acceptable, use UNION ALL for 30-100% better performance."
                    ),
                    location={
                        "line": getattr(union, "line", None),
                        "column": getattr(union, "col", None),
                    },
                    context={
                        "branch_count": union_info["branch_count"],
                        "union_sql": union.sql(dialect="spark")[:300],
                    },
                    severity=severity,
                )
                violation.fix_suggestion = self.suggest_fix(violation)
                violations.append(violation)

        return violations

    def _is_union_all(self, union: exp.Union) -> bool:
        """Check if UNION has ALL modifier.

        Args:
            union: The UNION expression

        Returns:
            True if UNION ALL, False if plain UNION
        """
        distinct = union.args.get("distinct")
        return distinct is False or (distinct is None and union.args.get("by_name") is not None)

    def _analyze_union(self, union: exp.Union) -> dict:
        """Analyze UNION structure for reporting.

        Args:
            union: The UNION expression

        Returns:
            Dictionary with UNION analysis
        """
        branch_count = self._count_union_branches(union)

        return {
            "branch_count": branch_count,
        }

    def _count_union_branches(self, union: exp.Union) -> int:
        """Count number of queries being UNIONed.

        Args:
            union: The UNION expression

        Returns:
            Number of queries in the UNION
        """
        count = 0

        def count_recursive(expr: exp.Expression) -> None:
            nonlocal count
            if isinstance(expr, exp.Union):
                count_recursive(expr.this)
                count_recursive(expr.expression)
            else:
                count += 1

        count_recursive(union)
        return count

    def _assess_severity(self, union: exp.Union, union_info: dict) -> RuleSeverity:
        """Assess severity based on UNION complexity.

        Args:
            union: The UNION expression
            union_info: Analysis info

        Returns:
            Appropriate severity level
        """
        branch_count = union_info.get("branch_count", 2)

        if branch_count > 3:
            return RuleSeverity.HIGH

        return RuleSeverity.MEDIUM

    def suggest_fix(self, violation: Violation) -> str:
        """Generate fix suggestion for UNION without ALL.

        Args:
            violation: The violation to fix

        Returns:
            Fix suggestion string
        """
        return (
            "Change UNION to UNION ALL if duplicates are impossible or acceptable:\n"
            "1. Review if duplicate rows can occur across the UNIONed queries\n"
            "2. Verify business logic allows duplicate rows in results\n"
            "3. If yes to either, replace UNION with UNION ALL\n"
            "Example: SELECT ... FROM t1 UNION ALL SELECT ... FROM t2\n"
            "\n"
            "Keep UNION (without ALL) only if:\n"
            "- Duplicate removal is specifically required by business logic\n"
            "- Duplicates are actually possible in the result sets\n"
            "- The performance cost is justified"
        )
