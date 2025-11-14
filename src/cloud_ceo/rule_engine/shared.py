"""Shared data models for the rule detection system.

This module contains the core data structures used across both AST-based
and LLM-based detectors, avoiding circular import issues.
"""

from dataclasses import dataclass, field
from typing import Any

from cloud_ceo.rule_engine.types import RuleSeverity


@dataclass
class Violation:
    """Represents a detected rule violation.

    A violation is created when a detector finds an anti-pattern in SQL code.
    It contains all information needed to report the issue and suggest fixes.
    """

    rule_id: str = field(metadata={"description": "Rule identifier (e.g., SPARK_001)"})
    severity: RuleSeverity = field(metadata={"description": "Severity level of the violation"})
    message: str = field(metadata={"description": "Human-readable description of the violation"})
    location: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Location information (line, column, etc.)"}
    )
    context: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Additional context for the violation"}
    )
    fix_suggestion: str | None = field(
        default=None,
        metadata={"description": "Suggested fix for the violation"}
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary format.

        Returns:
            Dictionary representation of the violation
        """
        return {
            "rule_id": self.rule_id,
            "severity": self.severity.value if isinstance(self.severity, RuleSeverity) else self.severity,
            "message": self.message,
            "location": self.location,
            "context": self.context,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class DetectionContext:
    """Context information available to detectors during analysis.

    Provides access to additional metadata and services that detectors
    may need beyond the SQL query itself.
    """

    catalog: Any | None = field(
        default=None,
        metadata={"description": "Spark catalog interface for metadata lookups"}
    )
    config: Any | None = field(
        default=None,
        metadata={"description": "Configuration settings"}
    )
    statistics: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Table/column statistics (cardinality, etc.)"}
    )
    metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Additional metadata (partitions, constraints, etc.)"}
    )

    def get_partition_columns(self, table: str) -> list[str]:
        """Get partition columns for a table.

        Args:
            table: Table name (catalog.schema.table or schema.table)

        Returns:
            List of partition column names
        """
        if not self.metadata:
            return []

        partitions = self.metadata.get("partitions", {})
        return partitions.get(table, [])

    def get_unique_columns(self, table: str) -> list[str]:
        """Get columns with uniqueness constraints for a table.

        Args:
            table: Table name (catalog.schema.table or schema.table)

        Returns:
            List of unique column names
        """
        if not self.metadata:
            return []

        unique_cols = self.metadata.get("unique_columns", {})
        return unique_cols.get(table, [])

    def get_cardinality(self, table: str, column: str) -> int | None:
        """Get cardinality (distinct count) for a column.

        Args:
            table: Table name
            column: Column name

        Returns:
            Cardinality estimate or None if unknown
        """
        if not self.statistics:
            return None

        table_stats = self.statistics.get(table, {})
        return table_stats.get(column, {}).get("cardinality")
