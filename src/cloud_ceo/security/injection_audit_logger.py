"""Audit logging for prompt injection detection.

This module provides configurable audit logging for detected injection patterns.
Designed for internal trusted environments - logs for visibility without blocking.

Philosophy:
- Assume no malicious intent from trusted employees
- Log for security monitoring and visibility
- Never block queries (false positives frustrate users)
- Configurable to enable/disable as needed
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


class InjectionAuditLogger:
    """Audit logger for prompt injection detection.

    Supports both Databricks Delta tables and local JSONL files for audit trails.
    Does NOT block queries - only logs for monitoring purposes.

    Features:
    - Configurable output (Databricks table or JSONL file)
    - Query hashing for deduplication
    - Rate limiting to prevent log spam
    - Privacy-preserving (limited query preview)
    - Severity classification (info, warning, critical)
    """

    def __init__(
        self,
        audit_table: Optional[str] = None,
        audit_file: Optional[Path] = None,
        spark_session: Optional[Any] = None,
        max_entries_per_minute: int = 100
    ) -> None:
        """Initialize injection audit logger.

        Args:
            audit_table: Databricks table name (e.g., "security.prompt_injection_audit")
            audit_file: Path to JSONL file for local logging
            spark_session: Spark session for Databricks logging (required if audit_table set)
            max_entries_per_minute: Rate limit for audit entries (default: 100)

        Raises:
            ValueError: If audit_table specified without spark_session
        """
        self.audit_table = audit_table
        self.audit_file = Path(audit_file) if audit_file else None
        self.spark = spark_session
        self.max_entries_per_minute = max_entries_per_minute

        # Rate limiting state
        self._minute_start = time.time()
        self._entries_this_minute = 0

        # Validate configuration
        if self.audit_table and not self.spark:
            raise ValueError(
                "Databricks audit logging requires spark_session parameter. "
                "Either provide spark_session or use audit_file for local logging."
            )

        if not self.audit_table and not self.audit_file:
            logger.info(
                "injection_audit_disabled",
                reason="No audit_table or audit_file configured"
            )

        # Create audit file directory if needed
        if self.audit_file:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(
                "injection_audit_file_enabled",
                file_path=str(self.audit_file)
            )

        # Initialize Databricks table if needed
        if self.audit_table and self.spark:
            self._ensure_audit_table_exists()
            logger.info(
                "injection_audit_databricks_enabled",
                table=self.audit_table
            )

    def log_injection_detection(
        self,
        query: str,
        patterns_detected: list[str],
        severity: str = "warning",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        databricks_query_id: Optional[str] = None
    ) -> None:
        """Log detected injection patterns for audit trail.

        Args:
            query: SQL query with detected patterns
            patterns_detected: List of regex patterns that matched
            severity: Classification (info, warning, critical)
            user_id: User who submitted the query (optional)
            session_id: Session identifier (optional)
            databricks_query_id: Databricks query ID if available (optional)
        """
        # Rate limiting check
        if not self._check_rate_limit():
            logger.warning(
                "injection_audit_rate_limited",
                max_entries_per_minute=self.max_entries_per_minute
            )
            return

        # Generate audit entry
        entry = self._create_audit_entry(
            query=query,
            patterns_detected=patterns_detected,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            databricks_query_id=databricks_query_id
        )

        # Write to configured outputs
        if self.audit_file:
            self._write_to_jsonl(entry)

        if self.audit_table and self.spark:
            self._write_to_databricks(entry)

        # Log at appropriate level (warning, not error - not treating as security incident)
        logger.warning(
            "injection_pattern_audited",
            query_hash=entry["query_hash"],
            patterns_count=len(patterns_detected),
            severity=severity
        )

    def _create_audit_entry(
        self,
        query: str,
        patterns_detected: list[str],
        severity: str,
        user_id: Optional[str],
        session_id: Optional[str],
        databricks_query_id: Optional[str]
    ) -> dict[str, Any]:
        """Create audit entry with all metadata.

        Args:
            query: SQL query text
            patterns_detected: Matched patterns
            severity: Classification level
            user_id: User identifier
            session_id: Session identifier
            databricks_query_id: Databricks query ID

        Returns:
            Dictionary with audit entry fields
        """
        # Hash full query for deduplication
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()

        # Truncate query preview (privacy/size)
        query_preview = query[:500] if len(query) > 500 else query

        # Current timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "timestamp": timestamp,
            "query_hash": query_hash,
            "query_preview": query_preview,
            "patterns_detected": patterns_detected,
            "severity": severity,
            "user_id": user_id or "unknown",
            "session_id": session_id or "unknown",
            "databricks_query_id": databricks_query_id or None
        }

    def _write_to_jsonl(self, entry: dict[str, Any]) -> None:
        """Write audit entry to JSONL file.

        Args:
            entry: Audit entry to write
        """
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(
                "injection_audit_file_write_failed",
                error=str(e),
                file_path=str(self.audit_file)
            )

    def _write_to_databricks(self, entry: dict[str, Any]) -> None:
        """Write audit entry to Databricks Delta table.

        Args:
            entry: Audit entry to write
        """
        try:
            # Convert to DataFrame and append
            df = self.spark.createDataFrame([entry])
            df.write.format("delta").mode("append").saveAsTable(self.audit_table)
        except Exception as e:
            logger.error(
                "injection_audit_databricks_write_failed",
                error=str(e),
                table=self.audit_table
            )

    def _ensure_audit_table_exists(self) -> None:
        """Create audit table if it doesn't exist.

        Creates a Delta table partitioned by date for efficient querying.
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.audit_table} (
            timestamp TIMESTAMP,
            query_hash STRING,
            query_preview STRING,
            patterns_detected ARRAY<STRING>,
            severity STRING,
            user_id STRING,
            session_id STRING,
            databricks_query_id STRING
        )
        USING DELTA
        PARTITIONED BY (DATE(timestamp))
        """

        try:
            self.spark.sql(create_table_sql)
            logger.info(
                "injection_audit_table_created",
                table=self.audit_table
            )
        except Exception as e:
            logger.error(
                "injection_audit_table_creation_failed",
                error=str(e),
                table=self.audit_table
            )

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits.

        Returns:
            True if within limits, False if rate limited
        """
        current_time = time.time()

        # Reset counter every minute
        if current_time - self._minute_start >= 60:
            self._minute_start = current_time
            self._entries_this_minute = 0

        # Check limit
        if self._entries_this_minute >= self.max_entries_per_minute:
            return False

        self._entries_this_minute += 1
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about audit logging.

        Returns:
            Dictionary with audit statistics
        """
        stats = {
            "audit_enabled": bool(self.audit_table or self.audit_file),
            "audit_table": self.audit_table,
            "audit_file": str(self.audit_file) if self.audit_file else None,
            "max_entries_per_minute": self.max_entries_per_minute,
            "entries_this_minute": self._entries_this_minute
        }

        # Get file size if JSONL logging enabled
        if self.audit_file and self.audit_file.exists():
            stats["audit_file_size_bytes"] = self.audit_file.stat().st_size
            stats["audit_file_lines"] = sum(1 for _ in open(self.audit_file))

        return stats
