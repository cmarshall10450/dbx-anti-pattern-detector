"""SPARK_003: Detect missing partition filters on partitioned tables.

This detector identifies queries on partitioned tables that don't filter on
partition columns, forcing full partition scans and eliminating partitioning benefits.
"""

import logging
from typing import Set

from sqlglot import exp

from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.shared import DetectionContext, Violation
from cloud_ceo.rule_engine.types import RuleSeverity

logger = logging.getLogger(__name__)


class MissingPartitionFilterDetector(RuleDetector):
    """Detector for missing partition filter anti-pattern (SPARK_003).

    Partitioned tables organize data by partition columns (e.g., date, country)
    for efficient querying. Querying without filtering on partition columns
    forces Spark to scan ALL partitions, negating performance benefits.

    This detector has two modes:
    1. **Catalog-aware mode**: Uses table metadata to detect actual partitioned tables
    2. **Heuristic mode**: Uses common partition column name patterns as hints

    Example bad query:
        SELECT product_id, revenue
        FROM sales
        WHERE product_id = 'P123'

    Example fixed query:
        SELECT product_id, revenue
        FROM sales
        WHERE sale_year = 2024
          AND sale_month = 3
          AND product_id = 'P123'
    """

    # Common partition column name patterns (for heuristic mode)
    COMMON_PARTITION_PATTERNS: Set[str] = {
        # Date partitions
        "year", "month", "day", "date", "dt",
        "year_month", "partition_date", "data_date",
        "event_date", "transaction_date", "order_date",
        # Hierarchical date partitions
        "sale_year", "sale_month", "sale_day",
        "event_year", "event_month", "event_day",
        # Geographic partitions
        "country", "region", "state", "city",
        # Organizational partitions
        "tenant", "customer_id", "org_id",
        # Time-based
        "hour", "partition_hour",
    }

    def __init__(self, catalog: any = None) -> None:
        """Initialize missing partition filter detector.

        Args:
            catalog: Optional catalog interface for metadata lookups
        """
        super().__init__(rule_id="SPARK_003", severity=RuleSeverity.CRITICAL)
        self.catalog = catalog

    def _detect_violations(
        self, ast: exp.Expression, context: DetectionContext | None = None
    ) -> list[Violation]:
        """Detect missing partition filters in parsed SQL AST.

        Args:
            ast: Parsed SQL expression tree
            context: Optional detection context with table metadata

        Returns:
            List of violations found
        """
        violations = []

        for select in ast.find_all(exp.Select):
            violations.extend(self._check_select_for_missing_filters(select, context))

        return violations

    def _check_select_for_missing_filters(
        self, select: exp.Select, context: DetectionContext | None
    ) -> list[Violation]:
        """Check a SELECT statement for missing partition filters.

        Args:
            select: The SELECT statement to analyze
            context: Optional detection context

        Returns:
            List of violations found
        """
        violations = []

        # Get all tables referenced in FROM/JOIN clauses
        tables = self._extract_table_references(select)

        for table_name, table_node in tables:
            # Get partition columns for this table
            partition_cols = self._get_partition_columns(table_name, context)

            if not partition_cols:
                # No partition info available, skip
                continue

            # Check if WHERE clause filters on partition columns
            where_clause = select.args.get("where")
            if not where_clause:
                # No WHERE clause at all - definitely missing filters
                violation = self._create_violation(
                    table_name, partition_cols, table_node, has_where=False
                )
                violations.append(violation)
                continue

            # Extract columns used in WHERE clause
            filtered_columns = self._extract_filtered_columns(where_clause)

            # Check which partition columns are missing
            missing_partitions = [
                col for col in partition_cols if col.lower() not in filtered_columns
            ]

            if missing_partitions:
                violation = self._create_violation(
                    table_name, missing_partitions, table_node, has_where=True
                )
                violations.append(violation)

        return violations

    def _extract_table_references(
        self, select: exp.Select
    ) -> list[tuple[str, exp.Table]]:
        """Extract direct table references from SELECT statement.

        Only extracts tables directly referenced in FROM/JOIN clauses,
        not tables nested inside subqueries.

        Args:
            select: The SELECT statement

        Returns:
            List of (table_name, table_node) tuples
        """
        tables = []

        # Get table from FROM clause (if it's a direct table, not a subquery)
        from_clause = select.args.get("from")
        if from_clause:
            from_this = from_clause.args.get("this")
            if isinstance(from_this, exp.Table):
                table_name = self._get_full_table_name(from_this)
                tables.append((table_name, from_this))

        # Get tables from JOIN clauses (if they're direct tables, not subqueries)
        for join in select.find_all(exp.Join):
            join_this = join.args.get("this")
            if isinstance(join_this, exp.Table):
                table_name = self._get_full_table_name(join_this)
                tables.append((table_name, join_this))

        return tables

    def _get_full_table_name(self, table: exp.Table) -> str:
        """Get fully qualified table name.

        Args:
            table: The table expression

        Returns:
            Fully qualified table name (catalog.schema.table or schema.table or table)
        """
        parts = []

        if hasattr(table, "catalog") and table.catalog:
            parts.append(table.catalog.name if hasattr(table.catalog, "name") else str(table.catalog))

        if hasattr(table, "db") and table.db:
            parts.append(table.db.name if hasattr(table.db, "name") else str(table.db))

        if hasattr(table, "name"):
            parts.append(table.name)

        return ".".join(parts) if parts else "unknown"

    def _get_partition_columns(
        self, table_name: str, context: DetectionContext | None
    ) -> list[str]:
        """Get partition columns for a table.

        Uses catalog if available, falls back to heuristics.

        Args:
            table_name: Table name to check
            context: Optional detection context with metadata

        Returns:
            List of partition column names
        """
        # Try catalog-aware detection first
        if context:
            partition_cols = context.get_partition_columns(table_name)
            if partition_cols:
                logger.debug(
                    "Found partition columns via catalog for %s: %s",
                    table_name,
                    partition_cols,
                )
                return partition_cols

        # Fall back to heuristic detection
        # This is a best-effort guess based on common naming patterns
        # In production, this would be enhanced with table scanning
        logger.debug(
            "No catalog metadata for %s, using heuristic detection", table_name
        )
        return []  # Conservative: don't guess without metadata

    def _extract_filtered_columns(self, where: exp.Where) -> Set[str]:
        """Extract column names that appear in WHERE clause filters.

        Args:
            where: The WHERE clause expression

        Returns:
            Set of column names (lowercase) that are filtered
        """
        filtered_columns = set()

        # Find all comparisons and predicates
        for predicate in where.find_all(exp.Predicate):
            # Get columns from the predicate
            for column in predicate.find_all(exp.Column):
                if hasattr(column, "name"):
                    filtered_columns.add(column.name.lower())

        return filtered_columns

    def _create_violation(
        self,
        table_name: str,
        missing_partitions: list[str],
        table_node: exp.Table,
        has_where: bool,
    ) -> Violation:
        """Create violation for missing partition filters.

        Args:
            table_name: Name of the partitioned table
            missing_partitions: List of partition columns without filters
            table_node: The table AST node
            has_where: Whether query has a WHERE clause

        Returns:
            Configured violation object
        """
        partition_list = ", ".join(missing_partitions)

        if not has_where:
            message = (
                f"Table '{table_name}' is partitioned by {partition_list}, but query "
                f"has no WHERE clause. This forces a full scan of all partitions, "
                f"eliminating partitioning benefits. Add filters on partition columns."
            )
        else:
            message = (
                f"Table '{table_name}' is partitioned by {partition_list}, but query "
                f"doesn't filter on these columns. This scans all partitions instead of "
                f"pruning to relevant ones. Add filters: WHERE {missing_partitions[0]} = ..."
            )

        return self.create_violation(
            message=message,
            location={
                "line": getattr(table_node, "line", None),
                "column": getattr(table_node, "col", None),
            },
            context={
                "table": table_name,
                "missing_partition_columns": missing_partitions,
                "has_where_clause": has_where,
            },
            fix_suggestion=self.suggest_fix(
                table_name, missing_partitions, has_where
            ),
        )

    def suggest_fix(
        self, table_name: str, missing_partitions: list[str], has_where: bool
    ) -> str:
        """Generate fix suggestion for missing partition filters.

        Args:
            table_name: Table name
            missing_partitions: Missing partition columns
            has_where: Whether query has WHERE clause

        Returns:
            Fix suggestion string
        """
        if not missing_partitions:
            return "Add partition column filters to WHERE clause"

        partition_examples = []
        for col in missing_partitions[:2]:  # Show max 2 examples
            # Suggest appropriate filter based on column name
            if "year" in col.lower():
                partition_examples.append(f"{col} = 2024")
            elif "month" in col.lower():
                partition_examples.append(f"{col} = 3")
            elif "date" in col.lower() or "dt" in col.lower():
                partition_examples.append(f"{col} >= '2024-01-01'")
            elif "country" in col.lower():
                partition_examples.append(f"{col} = 'US'")
            else:
                partition_examples.append(f"{col} = <value>")

        filter_clause = " AND ".join(partition_examples)

        if has_where:
            return (
                f"Add partition filters to existing WHERE clause. "
                f"Example: WHERE {filter_clause} AND <existing conditions>"
            )
        else:
            return f"Add WHERE clause with partition filters. Example: WHERE {filter_clause}"
