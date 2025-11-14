"""Data models for Cloud CEO SQL analysis.

This module defines the core data structures used throughout the library:
- QueryMetrics: Execution metrics from Databricks queries
- ClusterInfo: Cluster configuration details
- AnalysisResult: Complete analysis results with violations and recommendations
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from cloud_ceo.rule_engine.shared import Violation


@dataclass
class QueryMetrics:
    """Execution metrics for a query.

    Core fields are sourced from Databricks system.query.history table.
    Runtime utilization metrics are from system.compute.node_timeline (if available).

    Attributes:
        duration_ms: Query execution time in milliseconds (from execution_duration_ms)
        bytes_scanned: Total bytes read (from read_bytes)
        rows_scanned: Total rows returned (from produced_rows)
        read_rows: Total rows read before filtering (from read_rows, optional)
        spilled_bytes: Data written to disk during execution (from spilled_local_bytes, optional)
        shuffle_read_bytes: Data sent over network (from shuffle_read_bytes, optional)
        cost_usd: Estimated cost in USD (calculated, optional)
        timestamp: When the query was executed (from start_time, optional)

        peak_memory_percent: Peak memory utilization during query (from node_timeline, optional)
        avg_memory_percent: Average memory utilization (from node_timeline, optional)
        peak_cpu_percent: Peak CPU utilization (from node_timeline, optional)
        avg_cpu_percent: Average CPU utilization (from node_timeline, optional)
        memory_swap_detected: Whether memory swapping occurred (from node_timeline, optional)
        avg_io_wait_percent: Average I/O wait time (from node_timeline, optional)
    """

    duration_ms: int
    bytes_scanned: int
    rows_scanned: int
    read_rows: Optional[int] = None
    spilled_bytes: Optional[int] = None
    shuffle_read_bytes: Optional[int] = None
    cost_usd: Optional[float] = None
    timestamp: Optional[datetime] = None

    peak_memory_percent: Optional[float] = None
    avg_memory_percent: Optional[float] = None
    peak_cpu_percent: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    memory_swap_detected: Optional[bool] = None
    avg_io_wait_percent: Optional[float] = None

    def is_slow(self, threshold_seconds: int = 60) -> bool:
        """Check if query duration exceeds threshold.

        Args:
            threshold_seconds: Duration threshold in seconds (default: 60s)

        Returns:
            True if query took longer than threshold
        """
        return (self.duration_ms / 1000) > threshold_seconds

    def format_duration(self) -> str:
        """Human-readable duration (e.g., '45.0s' or '2.3m').

        Returns:
            Formatted duration string
        """
        seconds = self.duration_ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        return f"{minutes:.1f}m"

    def format_bytes_scanned(self) -> str:
        """Human-readable data volume (e.g., '1.12GB' or '500.0MB').

        Returns:
            Formatted bytes string
        """
        gb = self.bytes_scanned / (1024 ** 3)
        if gb < 1:
            mb = self.bytes_scanned / (1024 ** 2)
            return f"{mb:.1f}MB"
        return f"{gb:.2f}GB"

    def shuffle_read_ratio(self) -> Optional[float]:
        """Calculate shuffle read as ratio of data scanned.

        High ratios (>0.5) indicate significant data movement over network,
        which correlates with join complexity and suboptimal partitioning.

        Returns:
            Ratio of shuffle_read_bytes to bytes_scanned (0.0-1.0+), or None if unavailable

        Example:
            >>> metrics = QueryMetrics(
            ...     duration_ms=60000,
            ...     bytes_scanned=1_000_000_000,  # 1GB scanned
            ...     rows_scanned=1000,
            ...     shuffle_read_bytes=800_000_000  # 800MB shuffled over network
            ... )
            >>> metrics.shuffle_read_ratio()  # Returns 0.8 (80% - high network I/O)
        """
        if self.shuffle_read_bytes is None or self.bytes_scanned == 0:
            return None
        return self.shuffle_read_bytes / self.bytes_scanned

    def has_memory_spill(self) -> bool:
        """Check if query spilled to disk (memory pressure indicator).

        When Spark runs out of memory, it spills data to disk. This is a
        reliable indicator of memory pressure without needing peak memory usage.

        Returns:
            True if query spilled any data to disk
        """
        return (self.spilled_bytes or 0) > 0

    def spill_ratio(self) -> Optional[float]:
        """Calculate disk spill as ratio of data scanned.

        High ratios indicate memory pressure. Any spilling suggests
        insufficient memory for the workload.

        Returns:
            Ratio of spilled_bytes to bytes_scanned, or None if unavailable

        Example:
            >>> metrics = QueryMetrics(
            ...     duration_ms=60000,
            ...     bytes_scanned=1_000_000_000,  # 1GB scanned
            ...     rows_scanned=1000,
            ...     spilled_bytes=300_000_000  # 300MB spilled to disk
            ... )
            >>> metrics.spill_ratio()  # Returns 0.3 (30% spilled - memory pressure!)
        """
        if self.spilled_bytes is None or self.bytes_scanned == 0:
            return None
        return self.spilled_bytes / self.bytes_scanned

    def selectivity_ratio(self) -> Optional[float]:
        """Calculate selectivity (produced/read rows).

        Low values indicate inefficient filtering or missing predicates.
        High selectivity (close to 1.0) means most rows read are returned.

        Returns:
            Ratio of rows_scanned to read_rows (0.0-1.0), or None if unavailable

        Example:
            >>> metrics = QueryMetrics(
            ...     duration_ms=60000,
            ...     bytes_scanned=1_000_000_000,
            ...     rows_scanned=1000,      # Only 1K rows returned
            ...     read_rows=1_000_000     # But 1M rows read
            ... )
            >>> metrics.selectivity_ratio()  # Returns 0.001 (0.1% - poor filtering!)
        """
        if self.read_rows is None or self.read_rows == 0:
            return None
        return self.rows_scanned / self.read_rows


@dataclass
class ClusterInfo:
    """Cluster configuration information.

    All fields are sourced from Databricks system.compute.clusters table.

    Attributes:
        node_type: Worker node instance type (e.g., "i3.2xlarge")
        executor_count: Number of executor nodes
        executor_memory: Memory per executor (e.g., "8GB")
            In most cases this is auto-resolved from system.compute.node_types
        cluster_id: Databricks cluster ID (optional)
        cluster_name: Cluster name (optional)

        driver_node_type: Driver node instance type (optional, may differ from workers)
        driver_memory: Driver memory (optional, auto-resolved if driver_node_type provided)
        core_count: Number of vCPUs per worker node (optional)
        gpu_count: Number of GPUs per worker node (optional)
        dbr_version: Databricks Runtime version (optional)
        owned_by: Cluster owner username (optional)
        cluster_source: Source type: UI, API, JOB, PIPELINE (optional)
        data_security_mode: Access mode (optional)
        spot_instance: Whether cluster uses SPOT instances (optional)

    Note:
        As of v1.x, executor_memory is typically auto-resolved from Databricks
        system.compute.node_types table. Manual specification is only needed when:
        - Running outside Databricks environment
        - PySpark is not available
        - Using custom node types not in system tables
    """

    node_type: str
    executor_count: int
    executor_memory: str
    cluster_id: Optional[str] = None
    cluster_name: Optional[str] = None

    driver_node_type: Optional[str] = None
    driver_memory: Optional[str] = None
    core_count: Optional[int] = None
    gpu_count: Optional[int] = None
    dbr_version: Optional[str] = None
    owned_by: Optional[str] = None
    cluster_source: Optional[str] = None
    data_security_mode: Optional[str] = None
    spot_instance: Optional[bool] = None

    def executor_memory_gb(self) -> int:
        """Parse executor memory string to GB.

        Supports: "8GB", "8g", "8192m", "8388608k"

        Returns:
            Memory in GB (integer)
        """
        import re
        memory_str = self.executor_memory.lower().strip()
        match = re.match(r"(\d+)(gb?|mb?|kb?)?", memory_str)
        if not match:
            return 0

        value = int(match.group(1))
        unit = match.group(2) or "m"

        if unit.startswith("g"):
            return value
        elif unit.startswith("m"):
            return value // 1024
        elif unit.startswith("k"):
            return value // (1024 * 1024)
        return 0

    def total_cluster_memory(self) -> int:
        """Calculate total memory across all executors (in GB).

        Returns:
            Total cluster memory in GB
        """
        return self.executor_count * self.executor_memory_gb()


@dataclass
class AnalysisResult:
    """Unified result object for SQL analysis.

    Works for both pre-execution and post-execution analysis.

    Attributes:
        query_id: Unique identifier for this query
        sql_query: Original SQL query text
        analysis_type: "pre_execution" or "post_execution"
        violations: List of detected anti-patterns
        violation_count: Number of violations found

        # Optional LLM analysis
        llm_enabled: Whether LLM analysis was performed
        llm_recommendations: LLM-generated recommendations
        severity_assessment: LLM severity assessment
        impact_analysis: LLM impact analysis

        # Post-execution specific
        execution_metrics: Query performance metrics (post-execution only)
        cluster_config: Cluster configuration (post-execution only)

        # Databricks metadata
        databricks_query_id: Original Databricks query ID (optional)
        user: User who ran the query (optional)
        warehouse_id: SQL warehouse ID (optional)
    """

    query_id: str
    sql_query: str
    analysis_type: Literal["pre_execution", "post_execution"]
    violations: list[Violation]
    violation_count: int

    # LLM analysis
    llm_enabled: bool = False
    llm_recommendations: list[dict[str, Any]] = field(default_factory=list)
    severity_assessment: Optional[dict[str, Any]] = None
    impact_analysis: Optional[dict[str, Any]] = None
    violation_explanation: Optional[str] = None

    # Execution context
    execution_metrics: Optional[QueryMetrics] = None
    cluster_config: Optional[ClusterInfo] = None

    # Databricks metadata
    databricks_query_id: Optional[str] = None
    user: Optional[str] = None
    warehouse_id: Optional[str] = None

    @property
    def cost_usd(self) -> Optional[float]:
        """Get query cost if available."""
        return self.execution_metrics.cost_usd if self.execution_metrics else None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get query duration in seconds if available."""
        if self.execution_metrics:
            return self.execution_metrics.duration_ms / 1000
        return None

    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return self.violation_count > 0

    def get_critical_violations(self) -> list[Violation]:
        """Get only critical severity violations."""
        return [v for v in self.violations if v.severity.value == "critical"]

    def get_high_violations(self) -> list[Violation]:
        """Get only high severity violations."""
        return [v for v in self.violations if v.severity.value == "high"]

    def summary(self, show_metrics: bool = True) -> str:
        """Generate human-readable text summary.

        Args:
            show_metrics: Include execution metrics in summary

        Returns:
            Formatted text summary
        """
        if not self.has_violations():
            msg = f"✓ No violations found (Query ID: {self.query_id})"
            if self.analysis_type == "post_execution" and show_metrics:
                msg += f"\n  Duration: {self.duration_seconds:.1f}s"
                if self.cost_usd:
                    msg += f" | Cost: ${self.cost_usd:.2f}"
            return msg

        lines = [
            f"\n{'=' * 70}",
            f"Cloud CEO Analysis - {self.analysis_type.replace('_', ' ').title()}",
            f"Query ID: {self.query_id}",
        ]

        if self.databricks_query_id:
            lines.append(f"Databricks Query: {self.databricks_query_id}")

        if self.analysis_type == "post_execution" and show_metrics and self.execution_metrics:
            lines.append(f"Duration: {self.duration_seconds:.1f}s | "
                        f"Data Scanned: {self.execution_metrics.bytes_scanned / (1024**3):.2f}GB")
            if self.cost_usd:
                lines.append(f"Cost: ${self.cost_usd:.2f}")

        lines.append(f"{'=' * 70}\n")

        severity_icons = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🔵",
            "info": "ℹ️",
        }

        for i, v in enumerate(self.violations, 1):
            icon = severity_icons.get(v.severity.value, "•")
            lines.append(f"\n{i}. {icon} {v.rule_id} [{v.severity.value.upper()}]")
            lines.append(f"   {v.message}")

            if v.location.get("line"):
                lines.append(f"   Line: {v.location['line']}")

            if v.fix_suggestion:
                lines.append(f"   Fix: {v.fix_suggestion}")

        lines.append(f"\n{'-' * 70}")
        lines.append(f"Total: {self.violation_count} violation(s)")
        lines.append(f"{'=' * 70}\n")

        if self.llm_enabled and self.severity_assessment:
            lines.append(f"\n{'=' * 70}")
            lines.append("LLM ANALYSIS")
            lines.append(f"{'=' * 70}\n")

            # Show overall severity
            lines.append(f"Severity: {self.severity_assessment.get('overall_severity', self.severity_assessment.get('severity', 'N/A')).upper()}")
            lines.append(f"Confidence: {self.severity_assessment.get('overall_confidence', self.severity_assessment.get('confidence', 0)):.0%}")
            lines.append(f"Evidence: {self.severity_assessment.get('overall_evidence', self.severity_assessment.get('evidence', 'N/A'))}\n")

            # Show per-violation severity assessments if available
            if 'violation_assessments' in self.severity_assessment:
                lines.append(f"\n{'-' * 70}")
                lines.append("PER-VIOLATION SEVERITY BREAKDOWN")
                lines.append(f"{'-' * 70}\n")
                for i, va in enumerate(self.severity_assessment['violation_assessments'], 1):
                    severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}
                    icon = severity_icons.get(va['severity'].lower(), "•")
                    lines.append(f"{i}. {icon} [{va['severity'].upper()}] {va['violation_pattern']}")
                    lines.append(f"   Confidence: {va.get('confidence', 0):.0%}")
                    lines.append(f"   {va['evidence']}")
                    lines.append("")

        if self.llm_enabled and self.violation_explanation:
            lines.append(f"\n{'-' * 70}")
            lines.append("WHAT'S WRONG & WHY IT MATTERS")
            lines.append(f"{'-' * 70}\n")
            lines.append(self.violation_explanation)
            lines.append("")

        if self.llm_enabled and self.impact_analysis:
            lines.append(f"\n{'-' * 70}")
            lines.append("IMPACT ANALYSIS")
            lines.append(f"{'-' * 70}\n")

            # Show per-violation impact analyses if available
            if 'violation_impacts' in self.impact_analysis:
                lines.append("Per-Violation Impact Breakdown:")
                lines.append("")
                for i, vi in enumerate(self.impact_analysis['violation_impacts'], 1):
                    lines.append(f"{i}. {vi['violation_pattern']}")
                    lines.append(f"   Performance: {vi['performance_impact']}")
                    if vi.get('cost_contribution'):
                        lines.append(f"   Cost Contribution: ${vi['cost_contribution']:.2f}/day")
                    if vi.get('affected_resources'):
                        lines.append(f"   Affected: {', '.join(vi['affected_resources'])}")
                    lines.append("")
                lines.append(f"{'-' * 70}\n")

            # Show overall impact summary
            lines.append(f"Root Cause: {self.impact_analysis.get('root_cause', 'N/A')}")
            lines.append(f"Performance Impact: {self.impact_analysis.get('performance_impact', 'N/A')}")
            if self.impact_analysis.get('cost_impact_usd') or self.impact_analysis.get('cost_impact'):
                cost = self.impact_analysis.get('cost_impact_usd', self.impact_analysis.get('cost_impact', 0))
                lines.append(f"Cost Impact: ${cost:.2f}/day")
            lines.append("")

        # DEBUG: Check recommendations state
        import structlog
        debug_logger = structlog.get_logger(__name__)
        debug_logger.info(f"=== DISPLAY RECOMMENDATIONS CHECK ===")
        debug_logger.info(f"  - llm_enabled: {self.llm_enabled}")
        debug_logger.info(f"  - llm_recommendations exists: {hasattr(self, 'llm_recommendations')}")
        debug_logger.info(f"  - llm_recommendations value: {self.llm_recommendations if hasattr(self, 'llm_recommendations') else 'NOT SET'}")
        debug_logger.info(f"  - llm_recommendations len: {len(self.llm_recommendations) if hasattr(self, 'llm_recommendations') and self.llm_recommendations else 0}")

        if self.llm_enabled and self.llm_recommendations:
            lines.append(f"\n{'-' * 70}")
            lines.append(f"RECOMMENDATIONS ({len(self.llm_recommendations)})")
            lines.append(f"{'-' * 70}\n")

            for i, rec in enumerate(self.llm_recommendations, 1):
                lines.append(f"{i}. {rec.get('description', 'No description')}")
                lines.append(f"   Type: {rec.get('type', 'N/A')}")
                lines.append(f"   Expected Improvement: {rec.get('expected_improvement', 'N/A')}")
                lines.append(f"   Effort: {rec.get('effort', 'N/A')}")

                impl = rec.get('implementation', '')
                if impl:
                    lines.append(f"   Implementation:")
                    for line in impl.split('\n'):
                        if line.strip():
                            lines.append(f"      {line}")
                lines.append("")

            lines.append(f"{'=' * 70}\n")

        return "\n".join(lines)

    def as_json(self, indent: int = 2) -> str:
        """Generate JSON output."""
        output = {
            "query_id": self.query_id,
            "analysis_type": self.analysis_type,
            "violation_count": self.violation_count,
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "severity": v.severity.value,
                    "message": v.message,
                    "location": v.location,
                    "fix_suggestion": v.fix_suggestion,
                }
                for v in self.violations
            ],
        }

        if self.execution_metrics:
            output["execution_metrics"] = {
                "duration_ms": self.execution_metrics.duration_ms,
                "bytes_scanned": self.execution_metrics.bytes_scanned,
                "rows_scanned": self.execution_metrics.rows_scanned,
                "read_rows": self.execution_metrics.read_rows,
                "spilled_bytes": self.execution_metrics.spilled_bytes,
                "shuffle_read_bytes": self.execution_metrics.shuffle_read_bytes,
                "cost_usd": self.execution_metrics.cost_usd,
                "peak_memory_percent": self.execution_metrics.peak_memory_percent,
                "avg_memory_percent": self.execution_metrics.avg_memory_percent,
                "peak_cpu_percent": self.execution_metrics.peak_cpu_percent,
                "avg_cpu_percent": self.execution_metrics.avg_cpu_percent,
                "memory_swap_detected": self.execution_metrics.memory_swap_detected,
                "avg_io_wait_percent": self.execution_metrics.avg_io_wait_percent,
            }

        if self.cluster_config:
            output["cluster_config"] = {
                "node_type": self.cluster_config.node_type,
                "executor_count": self.cluster_config.executor_count,
                "executor_memory": self.cluster_config.executor_memory,
                "cluster_id": self.cluster_config.cluster_id,
                "cluster_name": self.cluster_config.cluster_name,
                "driver_node_type": self.cluster_config.driver_node_type,
                "driver_memory": self.cluster_config.driver_memory,
                "core_count": self.cluster_config.core_count,
                "gpu_count": self.cluster_config.gpu_count,
                "dbr_version": self.cluster_config.dbr_version,
                "owned_by": self.cluster_config.owned_by,
                "cluster_source": self.cluster_config.cluster_source,
                "data_security_mode": self.cluster_config.data_security_mode,
                "spot_instance": self.cluster_config.spot_instance,
            }

        if self.llm_enabled:
            output["llm_analysis"] = {
                "enabled": True,
                "recommendations": self.llm_recommendations,
                "severity_assessment": self.severity_assessment,
                "impact_analysis": self.impact_analysis,
                "violation_explanation": self.violation_explanation,
            }

        return json.dumps(output, indent=indent)

    def as_markdown(self) -> str:
        """Generate Markdown output."""
        if not self.has_violations():
            return (
                f"# Cloud CEO Analysis\n\n"
                f"**Query ID:** `{self.query_id}`  \n"
                f"**Type:** {self.analysis_type.replace('_', ' ').title()}\n\n"
                f"✓ No violations found"
            )

        lines = [
            "# Cloud CEO Analysis\n",
            f"**Query ID:** `{self.query_id}`  ",
            f"**Type:** {self.analysis_type.replace('_', ' ').title()}  ",
            f"**Violations:** {self.violation_count}\n",
        ]

        if self.execution_metrics:
            lines.append("\n## Execution Metrics\n")
            lines.append(f"- **Duration:** {self.duration_seconds:.1f}s  ")
            lines.append(f"- **Data Scanned:** {self.execution_metrics.bytes_scanned / (1024**3):.2f}GB  ")
            if self.cost_usd:
                lines.append(f"- **Cost:** ${self.cost_usd:.2f}  ")
            lines.append("\n")

        lines.append("## Violations\n")

        for i, v in enumerate(self.violations, 1):
            lines.append(f"\n### {i}. {v.rule_id} ({v.severity.value})\n")
            lines.append(f"{v.message}\n")

            if v.location.get("line"):
                lines.append(f"**Location:** Line {v.location['line']}\n")

            if v.fix_suggestion:
                lines.append(f"**Fix:** {v.fix_suggestion}\n")

            if self.llm_enabled and self.llm_recommendations:
                for rec in self.llm_recommendations:
                    lines.append(f"\n**LLM Recommendation:**\n")
                    lines.append(f"{rec.get('description', '')}\n")

            lines.append("---\n")

        return "".join(lines)
