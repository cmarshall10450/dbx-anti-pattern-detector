"""Context building for LLM analysis.

This module provides ContextBuilder which formats all context once for reuse
across workflow nodes, achieving 70% token savings and enabling cluster-aware analysis.
"""

import structlog
from dataclasses import dataclass
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cloud_ceo.llm.context_formatter import XMLContextFormatter


@dataclass(frozen=True)
class AnalysisContext:
    """Pre-formatted context for LLM workflow.

    This immutable dataclass holds the formatted context summary and
    metadata about what information is available for analysis.

    Attributes:
        context_summary: Ready-to-use formatted string with all context
        confidence_level: FULL | CLUSTER_ONLY | METRICS_ONLY | MINIMAL
        has_metrics: Whether execution metrics are available
        has_cluster: Whether cluster configuration is available
        has_all_violations: Whether multiple violations are present for cross-reasoning
        violation_count: Total number of violations detected
        has_custom_guidelines: Whether custom Markdown guidelines are included
    """

    context_summary: str
    confidence_level: str
    has_metrics: bool
    has_cluster: bool
    has_all_violations: bool
    violation_count: int
    has_custom_guidelines: bool = False


class ExecutionMetrics:
    """Execution metrics for LLM context.

    Note: peak_memory_mb is NOT available from Databricks system.query.history.
    It will always be 0 when fetched from system tables.
    """

    def __init__(
        self,
        duration_ms: int,
        bytes_scanned: int,
        rows_scanned: int,
        peak_memory_mb: int,  # Not available in system tables (kept for backward compatibility)
        shuffle_bytes: Optional[int] = None  # shuffle_read_bytes from system tables
    ) -> None:
        self.duration_ms = duration_ms
        self.bytes_scanned = bytes_scanned
        self.rows_scanned = rows_scanned
        self.peak_memory_mb = peak_memory_mb
        self.shuffle_bytes = shuffle_bytes

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

    def shuffle_ratio(self) -> Optional[float]:
        """Calculate shuffle-to-scan ratio.

        High ratios (>0.5) indicate inefficient joins or Cartesian products
        where shuffled data exceeds scanned data significantly.

        Returns:
            Ratio of shuffle_bytes to bytes_scanned (0.0-1.0+), or None if unavailable

        Example:
            >>> metrics = ExecutionMetrics(
            ...     duration_ms=60000,
            ...     bytes_scanned=1_000_000_000,  # 1GB scanned
            ...     rows_scanned=1000,
            ...     peak_memory_mb=2048,
            ...     shuffle_bytes=800_000_000     # 800MB shuffled
            ... )
            >>> metrics.shuffle_ratio()  # Returns 0.8 (80% - concerning)
        """
        if self.shuffle_bytes is None or self.bytes_scanned == 0:
            return None
        return self.shuffle_bytes / self.bytes_scanned


class ClusterConfig:
    """Placeholder for cluster configuration data model.

    This will be defined in a separate story. For now, we assume
    it has these attributes.
    """

    def __init__(
        self,
        node_type: str,
        executor_count: int,
        executor_memory: str
    ) -> None:
        self.node_type = node_type
        self.executor_count = executor_count
        self.executor_memory = executor_memory

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
        unit = match.group(2)

        if not unit:
            return value
        elif unit.startswith("g"):
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


class Violation:
    """Placeholder for violation data model.

    This will be defined in a separate story. For now, we assume
    it has these attributes.
    """

    def __init__(
        self,
        pattern: str,
        fragment: str,
        line_number: int,
        default_message: str = "",
        severity: str = "medium",
        fix_suggestion: str = ""
    ) -> None:
        self.pattern = pattern
        self.fragment = fragment
        self.line_number = line_number
        self.default_message = default_message
        self.severity = severity
        self.fix_suggestion = fix_suggestion


class ContextBuilder:
    """Format all context once for reuse across workflow nodes.

    This class implements the ContextBuilder pattern to:
    1. Build context once, reuse everywhere (DRY principle)
    2. Achieve 70% token savings via LangChain message history
    3. Enable cluster-aware severity analysis (KEY DIFFERENTIATOR)
    4. Support cross-violation detection
    5. Include custom Markdown guidelines for company-specific requirements
    """

    def __init__(self, use_xml_format: bool = True) -> None:
        """Initialize context builder.

        Args:
            use_xml_format: If True, use XML-based structured format for Claude (default).
                           If False, use legacy markdown format for backward compatibility.
        """
        self.use_xml_format = use_xml_format
        self._logger = structlog.get_logger(__name__)

        self._logger.info("=== CONTEXT BUILDER INITIALIZATION ===")
        self._logger.info(f"use_xml_format parameter: {use_xml_format}")

        self.xml_formatter = None
        if use_xml_format:
            # Lazy import to avoid circular dependency
            from cloud_ceo.llm.context_formatter import XMLContextFormatter
            self.xml_formatter = XMLContextFormatter()
            self._logger.info(f"XML formatter created: {self.xml_formatter is not None}")
        else:
            self._logger.info("Legacy formatter mode (XML disabled)")

    def build_context(
        self,
        violation: Violation,
        all_violations: List[Violation],
        execution_metrics: Optional[ExecutionMetrics] = None,
        cluster_config: Optional[ClusterConfig] = None,
        custom_guidelines_text: Optional[str] = None,
        sql_query: Optional[str] = None,
        databricks_session: Optional[Any] = None
    ) -> AnalysisContext:
        """Build formatted context with derived metrics and auto-fetched table schemas.

        Args:
            violation: Current violation being analyzed
            all_violations: All detected violations for cross-reasoning
            execution_metrics: Optional execution performance data
            cluster_config: Optional cluster configuration
            custom_guidelines_text: Optional Markdown guidelines text
            sql_query: Optional full SQL query text (for table schema extraction)
            databricks_session: Optional DatabricksSystemTables instance for schema fetching

        Returns:
            AnalysisContext with pre-formatted summary and metadata
        """
        confidence = self._determine_confidence(
            execution_metrics, cluster_config, all_violations
        )

        # Calculate cost if we have both metrics and cluster config
        calculated_cost = None
        if execution_metrics and cluster_config:
            self._logger.info("=== ATTEMPTING COST CALCULATION ===")
            self._logger.info(f"  - execution_metrics: {execution_metrics}")
            self._logger.info(f"  - cluster_config: {cluster_config}")
            calculated_cost = self._calculate_cost(execution_metrics, cluster_config)
            self._logger.info(f"=== COST CALCULATION RESULT: ${calculated_cost:.2f}/day ===" if calculated_cost else "=== COST CALCULATION FAILED ===")

        # Auto-fetch table schemas if SQL query and Databricks session provided
        table_schemas = None
        if sql_query and databricks_session:
            table_schemas = self._fetch_table_schemas(sql_query, databricks_session)

        # Use XML formatter if enabled, otherwise use legacy format
        self._logger.info("=== CONTEXT BUILDER: build_context() ===")
        self._logger.info(f"use_xml_format: {self.use_xml_format}")
        self._logger.info(f"xml_formatter exists: {self.xml_formatter is not None}")
        self._logger.info(f"all_violations count: {len(all_violations)}")

        if self.use_xml_format and self.xml_formatter:
            self._logger.info("✓ Using XML formatter to build context")
            context_summary = self.xml_formatter.format_context(
                violation=violation,
                all_violations=all_violations,
                execution_metrics=execution_metrics,
                cluster_config=cluster_config,
                custom_guidelines_text=custom_guidelines_text,
                table_schemas=table_schemas,
                calculated_cost=calculated_cost
            )
            self._logger.info(f"XML context generated: {len(context_summary)} chars")
            self._logger.info(f"Contains <all_violations>: {'<all_violations' in context_summary}")
            self._logger.info(f"Contains <violations_summary>: {'<violations_summary>' in context_summary}")
        else:
            # Legacy markdown format
            self._logger.info("Using legacy formatter to build context")
            self._logger.info(f"Reason: use_xml_format={self.use_xml_format}, xml_formatter={self.xml_formatter}")
            context_summary = self._build_legacy_context(
                violation,
                all_violations,
                execution_metrics,
                cluster_config,
                custom_guidelines_text,
                table_schemas,
                calculated_cost
            )

        analysis_context = AnalysisContext(
            context_summary=context_summary,
            confidence_level=confidence,
            has_metrics=execution_metrics is not None,
            has_cluster=cluster_config is not None,
            has_all_violations=len(all_violations) > 1,
            violation_count=len(all_violations),
            has_custom_guidelines=bool(custom_guidelines_text)
        )

        import json
        self._logger.info(f"Analysis context: {analysis_context}")

        return analysis_context

    def _fetch_table_schemas(
        self,
        sql_query: str,
        databricks_session: Any
    ) -> Optional[List[Any]]:
        """Fetch table schemas for all tables referenced in SQL query.

        Args:
            sql_query: Full SQL query text
            databricks_session: DatabricksSystemTables instance

        Returns:
            List of schema dicts or None if fetch fails
        """
        try:
            from cloud_ceo.databricks.sql_parser import (
                extract_tables_from_query,
                is_system_table
            )

            # Extract table names from query
            table_names = extract_tables_from_query(sql_query)

            if not table_names:
                return None

            # Fetch schemas for each table (excluding system tables)
            schemas = []
            for table_name in table_names:
                # Skip system tables
                if is_system_table(table_name):
                    continue

                schema = databricks_session.fetch_table_schema(table_name)
                if schema:
                    schemas.append(schema)

            return schemas if schemas else None

        except Exception as e:
            self._logger.debug(f"Failed to fetch table schemas: {e}")
            return None

    def _calculate_cost(
        self,
        metrics: ExecutionMetrics,
        cluster: ClusterConfig
    ) -> Optional[float]:
        """Calculate daily cost impact using DBU-based pricing.

        Args:
            metrics: Execution metrics with duration
            cluster: Cluster configuration

        Returns:
            Daily cost in USD, or None if calculation fails
        """
        try:
            self._logger.info("  - Importing cost_calculator module...")
            from cloud_ceo.cost_calculator import calculate_cost

            self._logger.info(f"  - Calling calculate_cost with:")
            self._logger.info(f"      node_type={cluster.node_type}")
            self._logger.info(f"      executor_count={cluster.executor_count}")
            self._logger.info(f"      duration_ms={metrics.duration_ms}")
            self._logger.info(f"      executor_memory={cluster.executor_memory}")

            cost_estimate = calculate_cost(
                node_type=cluster.node_type,
                executor_count=cluster.executor_count,
                duration_ms=metrics.duration_ms,
                executor_memory=cluster.executor_memory,
                workload_type="all_purpose",  # Conservative estimate
                executions_per_day=24  # Assume hourly execution
            )

            self._logger.info(
                f"  - Calculated cost: ${cost_estimate.daily_cost_usd:.2f}/day "
                f"({cost_estimate.calculation_method})"
            )

            return cost_estimate.daily_cost_usd

        except Exception as e:
            self._logger.warning(f"  - Cost calculation failed: {e}")
            import traceback
            self._logger.warning(f"  - Traceback: {traceback.format_exc()}")
            return None

    def _build_legacy_context(
        self,
        violation: Violation,
        all_violations: List[Violation],
        execution_metrics: Optional[ExecutionMetrics],
        cluster_config: Optional[ClusterConfig],
        custom_guidelines_text: Optional[str],
        table_schemas: Optional[List[Any]] = None,
        calculated_cost: Optional[float] = None
    ) -> str:
        """Build context using legacy markdown format.

        Args:
            violation: Current violation being analyzed
            all_violations: All detected violations for cross-reasoning
            execution_metrics: Optional execution performance data
            cluster_config: Optional cluster configuration
            custom_guidelines_text: Optional Markdown guidelines text
            table_schemas: Optional table schemas
            calculated_cost: Optional pre-calculated daily cost in USD

        Returns:
            Formatted context string in legacy markdown format
        """
        summary_parts = []

        # Start with all violations to emphasize equal importance
        if all_violations:
            summary_parts.append(self._format_all_violations(all_violations))

        # Add table schemas if available
        if table_schemas:
            summary_parts.append(self._format_table_schemas_markdown(table_schemas))

        if execution_metrics and cluster_config:
            summary_parts.append(self._format_derived_metrics(
                execution_metrics, cluster_config, calculated_cost
            ))
        elif execution_metrics:
            summary_parts.append(self._format_metrics_only(execution_metrics))
        elif cluster_config:
            summary_parts.append(self._format_cluster_only(cluster_config))

        if custom_guidelines_text:
            summary_parts.append("\n**Custom Guidelines (Markdown):**\n")
            summary_parts.append(custom_guidelines_text)
            summary_parts.append("\n")

        return "".join(summary_parts)

    def _format_derived_metrics(
        self,
        metrics: ExecutionMetrics,
        cluster: ClusterConfig,
        calculated_cost: Optional[float] = None
    ) -> str:
        """Format metrics with business context.

        Args:
            metrics: Execution performance metrics
            cluster: Cluster configuration
            calculated_cost: Optional pre-calculated daily cost in USD

        Returns:
            Formatted string with derived metrics and risk assessment
        """
        total_memory = cluster.total_cluster_memory()

        lines = ["\n**Execution Metrics & Cluster Context:**\n"]

        duration_s = metrics.duration_ms / 1000
        if duration_s > 60:
            lines.append(f"- Duration: {duration_s:.1f}s (SLOW - needs optimization)\n")
        else:
            lines.append(f"- Duration: {duration_s:.1f}s\n")

        # Add calculated cost if available
        if calculated_cost is not None:
            lines.append(f"- **Calculated Cost Impact:** ${calculated_cost:.2f}/day (assuming hourly execution)\n")

        gb_scanned = metrics.bytes_scanned / (1024**3)
        lines.append(f"- Data Scanned: {gb_scanned:.2f}GB\n")
        lines.append(f"- Rows Processed: {metrics.rows_scanned:,}\n")

        if metrics.peak_memory_mb > 0:
            executor_memory_mb = cluster.executor_memory_gb() * 1024
            memory_utilization_pct = (metrics.peak_memory_mb / executor_memory_mb) * 100
            if memory_utilization_pct > 90:
                lines.append(f"- Memory Used: {metrics.peak_memory_mb}MB ({memory_utilization_pct:.0f}% of executor - HIGH RISK)\n")
            elif memory_utilization_pct > 70:
                lines.append(f"- Memory Used: {metrics.peak_memory_mb}MB ({memory_utilization_pct:.0f}% of executor - MODERATE RISK)\n")
            else:
                lines.append(f"- Memory Used: {metrics.peak_memory_mb}MB ({memory_utilization_pct:.0f}% of executor)\n")

        shuffle_ratio = metrics.shuffle_ratio()
        if shuffle_ratio is not None:
            shuffle_gb = metrics.shuffle_bytes / (1024**3) if metrics.shuffle_bytes else 0
            if shuffle_ratio > 1.0:
                lines.append(
                    f"- Shuffle: {shuffle_gb:.2f}GB ({shuffle_ratio:.0%} of scanned - CARTESIAN JOIN LIKELY)\n"
                )
            elif shuffle_ratio > 0.5:
                lines.append(
                    f"- Shuffle: {shuffle_gb:.2f}GB ({shuffle_ratio:.0%} of scanned - inefficient join)\n"
                )
            elif shuffle_ratio == 0.0:
                lines.append(
                    f"- Shuffle: {shuffle_gb:.0f}GB (broadcast join - efficient but memory-intensive)\n"
                )
            else:
                lines.append(f"- Shuffle: {shuffle_gb:.2f}GB ({shuffle_ratio:.0%})\n")

        lines.append("\n**Cluster:**\n")
        lines.append(f"- Type: {cluster.node_type}\n")
        lines.append(f"- Executors: {cluster.executor_count} x {cluster.executor_memory}\n")
        lines.append(f"- Total Memory: {total_memory}GB\n")

        return "".join(lines)

    def _format_metrics_only(self, metrics: ExecutionMetrics) -> str:
        """Format execution metrics without cluster context.

        Args:
            metrics: Execution performance metrics

        Returns:
            Formatted string with basic metrics
        """
        lines = ["\n**Execution Metrics:**\n"]

        duration_s = metrics.duration_ms / 1000
        lines.append(f"- Duration: {duration_s:.1f}s\n")

        gb_scanned = metrics.bytes_scanned / (1024**3)
        lines.append(f"- Data Scanned: {gb_scanned:.2f}GB\n")
        lines.append(f"- Rows Processed: {metrics.rows_scanned:,}\n")

        shuffle_ratio = metrics.shuffle_ratio()
        if shuffle_ratio is not None:
            shuffle_gb = metrics.shuffle_bytes / (1024**3) if metrics.shuffle_bytes else 0
            if shuffle_ratio > 1.0:
                lines.append(
                    f"- Shuffle: {shuffle_gb:.2f}GB ({shuffle_ratio:.0%} - CARTESIAN JOIN LIKELY)\n"
                )
            elif shuffle_ratio > 0.5:
                lines.append(f"- Shuffle: {shuffle_gb:.2f}GB ({shuffle_ratio:.0%} - inefficient join)\n")
            elif shuffle_ratio == 0.0:
                lines.append(f"- Shuffle: {shuffle_gb:.0f}GB (broadcast join - efficient but memory-intensive)\n")
            else:
                lines.append(f"- Shuffle: {shuffle_gb:.2f}GB ({shuffle_ratio:.0%})\n")

        return "".join(lines)

    def _format_cluster_only(self, cluster: ClusterConfig) -> str:
        """Format cluster configuration without metrics.

        Args:
            cluster: Cluster configuration

        Returns:
            Formatted string with cluster details
        """
        total_memory = cluster.total_cluster_memory()

        lines = ["\n**Cluster Configuration:**\n"]
        lines.append(f"- Type: {cluster.node_type}\n")
        lines.append(f"- Executors: {cluster.executor_count} x {cluster.executor_memory}\n")
        lines.append(f"- Total Memory: {total_memory}GB\n")

        return "".join(lines)

    def _format_all_violations(self, violations: List[Violation]) -> str:
        """Format all violations for cross-reasoning.

        Args:
            violations: All detected violations

        Returns:
            Formatted string with violation summary
        """
        lines = [f"**⚠️ All Detected Violations ({len(violations)} total) - ALL MUST BE ADDRESSED ⚠️**\n\n"]
        for idx, v in enumerate(violations, 1):
            lines.append(f"{idx}. **{v.pattern}** at line {v.line_number}\n")
            lines.append(f"   - Code: `{v.fragment}`\n")
            if v.default_message:
                lines.append(f"   - Explanation: {v.default_message}\n")
            lines.append("\n")
        lines.append(f"**Coverage Requirement:** Generate recommendations that collectively address all {len(violations)} violations listed above.\n")
        lines.append("**Note:** Consider interactions between these violations.\n")
        return "".join(lines)

    def _format_table_schemas_markdown(self, table_schemas: List[Any]) -> str:
        """Format table schemas in markdown format for legacy context.

        Args:
            table_schemas: List of schema dicts

        Returns:
            Formatted markdown string
        """
        parts = [f"\n**Table Schemas ({len(table_schemas)} tables):**\n"]

        for schema in table_schemas:
            parts.append(f"\n- **{schema['table_name']}**\n")

            if schema.get("table_comment"):
                parts.append(f"  - Description: {schema['table_comment']}\n")

            # Unity Catalog table tags
            if schema.get("table_tags"):
                tag_strs = [f"{k}={v}" for k, v in schema["table_tags"].items()]
                parts.append(f"  - Tags: {', '.join(tag_strs)}\n")

            if schema.get("pii_indicators"):
                pii_str = ", ".join(schema["pii_indicators"])
                parts.append(f"  - PII Detected: {pii_str}\n")

            if schema.get("security_tags"):
                tags_str = ", ".join(schema["security_tags"])
                parts.append(f"  - Compliance: {tags_str}\n")

            columns = schema.get("columns", [])
            parts.append(f"  - Columns: {len(columns)}\n")

            # Show columns with tags
            for col in columns:
                col_tags = col.get("column_tags", {})
                if col_tags:
                    tag_strs = [f"{k}={v}" for k, v in col_tags.items()]
                    parts.append(f"    - {col['name']} ({col['type']}): {', '.join(tag_strs)}\n")

        return "".join(parts)

    def _determine_confidence(
        self,
        metrics: Optional[ExecutionMetrics],
        cluster: Optional[ClusterConfig],
        all_violations: List[Violation]
    ) -> str:
        """Determine confidence level based on available context.

        Args:
            metrics: Optional execution metrics
            cluster: Optional cluster configuration
            all_violations: All detected violations

        Returns:
            Confidence level string (FULL, CLUSTER_ONLY, METRICS_ONLY, MINIMAL)
        """
        if metrics and cluster and len(all_violations) > 1:
            return "FULL"
        elif cluster:
            return "CLUSTER_ONLY"
        elif metrics:
            return "METRICS_ONLY"
        else:
            return "MINIMAL"
