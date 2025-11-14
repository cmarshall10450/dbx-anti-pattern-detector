"""XML-based context formatter for LLM analysis.

This module provides XMLContextFormatter which generates optimized XML context
that combines structured parsing with visual hierarchy for Claude 3.5 Sonnet.
"""

import structlog
import textwrap
from typing import Any, List, Optional
from cloud_ceo.llm.context import ExecutionMetrics, ClusterConfig, Violation


class XMLContextFormatter:
    """Generate optimized XML context for LLM analysis.

    This formatter creates a hybrid XML-Markdown structure that:
    1. Uses XML tags for efficient parsing by Claude
    2. Includes visual indicators (❗⚠️✓) for attention prioritization
    3. Provides evidence tags for causal reasoning
    4. Links violations → metrics → impacts → recommendations
    """

    def __init__(self) -> None:
        """Initialize XML context formatter."""
        self._logger = structlog.get_logger(__name__)

    def format_context(
        self,
        violation: Violation,
        all_violations: List[Violation],
        execution_metrics: Optional[ExecutionMetrics] = None,
        cluster_config: Optional[ClusterConfig] = None,
        custom_guidelines_text: Optional[str] = None,
        table_schemas: Optional[List[dict[str, Any]]] = None,
        calculated_cost: Optional[float] = None
    ) -> str:
        """Generate XML-formatted context.

        Args:
            violation: Current violation being analyzed
            all_violations: All detected violations for cross-reasoning
            execution_metrics: Optional execution performance data
            cluster_config: Optional cluster configuration
            custom_guidelines_text: Optional Markdown guidelines text
            table_schemas: Optional list of table schemas (from fetch_table_schema)
            calculated_cost: Optional pre-calculated daily cost in USD

        Returns:
            XML-formatted context string
        """
        self._logger.info("=== XML FORMATTER: format_context() CALLED ===")
        self._logger.info(f"Violations to format: {len(all_violations)}")
        self._logger.info(f"Has metrics: {execution_metrics is not None}")
        self._logger.info(f"Has cluster: {cluster_config is not None}")

        parts = ["<query_context>\n"]

        # ALL VIOLATIONS FIRST - This is the most critical information
        if all_violations:
            self._logger.info(f"Calling _format_all_violations with {len(all_violations)} violations")
            violations_xml = self._format_all_violations(all_violations)
            self._logger.info(f"Received violations XML: {len(violations_xml)} chars")
            parts.append(violations_xml)
        else:
            self._logger.warning("No violations provided to format_context!")

        # Current violation context
        parts.append(self._format_current_violation(violation))

        if table_schemas:
            parts.append(self._format_table_schemas(table_schemas))

        if execution_metrics:
            parts.append(self._format_performance_profile(
                execution_metrics, cluster_config, calculated_cost
            ))

        if execution_metrics:
            parts.append(self._format_resource_utilization(
                execution_metrics, cluster_config
            ))

        if cluster_config:
            parts.append(self._format_cluster_configuration(cluster_config))

        if custom_guidelines_text:
            parts.append(self._format_custom_guidelines(custom_guidelines_text))

        parts.append("</query_context>")

        context_xml = "".join(parts)

        # Log the generated XML structure for debugging
        self._logger.info(
            "=== XML CONTEXT GENERATED ===",
            has_all_violations="<all_violations" in context_xml,
            violation_count=len(all_violations) if all_violations else 0,
            context_length=len(context_xml)
        )

        # Check if <all_violations> section is present
        if all_violations and "<all_violations" not in context_xml:
            self._logger.warning("⚠️ NO <all_violations> section found in context!")
        elif all_violations:
            self._logger.info(f"✓ <all_violations> section included with {len(all_violations)} violations")

        return context_xml

    def _format_current_violation(self, violation: Violation) -> str:
        """Format current violation being analyzed.

        Args:
            violation: Current violation

        Returns:
            XML-formatted violation section
        """
        parts = [
            "    <current_violation>\n",
            f"      <pattern>{violation.pattern}</pattern>\n",
            f"      <code_fragment>{violation.fragment}</code_fragment>\n",
            f"      <line_number>{violation.line_number}</line_number>\n"
        ]

        if violation.default_message:
            parts.append(f"      <explanation>{violation.default_message}</explanation>\n")

        parts.append("    </current_violation>\n\n")

        return "".join(parts)

    def _format_performance_profile(
        self,
        metrics: ExecutionMetrics,
        cluster: Optional[ClusterConfig],
        calculated_cost: Optional[float] = None
    ) -> str:
        """Format high-level performance profile with severity indicator.

        Args:
            metrics: Execution performance metrics
            cluster: Optional cluster configuration
            calculated_cost: Optional pre-calculated daily cost in USD

        Returns:
            XML-formatted performance profile
        """
        duration_s = metrics.duration_ms / 1000

        # Determine severity
        severity = "NORMAL"
        indicator = "✓ Performance within normal range"

        if duration_s > 300:  # 5 minutes
            severity = "CRITICAL"
            indicator = "❗ CRITICAL: Query took over 5 minutes"
        elif duration_s > 60:  # 1 minute
            severity = "WARNING"
            indicator = "⚠️ WARNING: Query took over 1 minute"

        # Check for memory swap if we have the data
        shuffle_ratio = metrics.shuffle_ratio()
        if shuffle_ratio and shuffle_ratio > 1.0:
            severity = "CRITICAL"
            indicator = "❗ CRITICAL: High network I/O detected (shuffle > data scanned)"

        parts = [
            "  <performance_profile>\n",
            f"    <severity>{severity}</severity>\n",
            f"    <indicator>{indicator}</indicator>\n",
            f"    <duration_seconds>{duration_s:.1f}</duration_seconds>\n",
            "    <duration_threshold>Normal: &lt;60s | Warning: 60-300s | Critical: &gt;300s</duration_threshold>\n"
        ]

        # Add calculated cost if available
        if calculated_cost is not None:
            parts.append(f"    <calculated_cost_usd_per_day>{calculated_cost:.2f}</calculated_cost_usd_per_day>\n")

        parts.append("  </performance_profile>\n\n")

        return "".join(parts)

    def _format_resource_utilization(
        self,
        metrics: ExecutionMetrics,
        cluster: Optional[ClusterConfig]
    ) -> str:
        """Format resource utilization with evidence and recommendations.

        Args:
            metrics: Execution performance metrics
            cluster: Optional cluster configuration

        Returns:
            XML-formatted resource utilization section
        """
        parts = ["  <resource_utilization>\n"]

        # Network transfer (shuffle)
        shuffle_section = self._format_network_transfer(metrics)
        if shuffle_section:
            parts.append(shuffle_section)

        # I/O efficiency
        io_section = self._format_io_efficiency(metrics)
        if io_section:
            parts.append(io_section)

        # Memory utilization
        memory_section = self._format_memory_utilization(metrics, cluster)
        if memory_section:
            parts.append(memory_section)

        parts.append("  </resource_utilization>\n\n")

        return "".join(parts)

    def _format_network_transfer(self, metrics: ExecutionMetrics) -> str:
        """Format network transfer (shuffle) metrics.

        Args:
            metrics: Execution performance metrics

        Returns:
            XML-formatted network transfer section or empty string
        """
        shuffle_ratio = metrics.shuffle_ratio()
        if shuffle_ratio is None:
            return ""

        shuffle_gb = metrics.shuffle_bytes / (1024**3) if metrics.shuffle_bytes else 0
        bytes_gb = metrics.bytes_scanned / (1024**3)

        if shuffle_ratio > 1.0:
            severity = "CRITICAL"
            indicator = f"❗ NETWORK: {shuffle_gb:.2f}GB shuffled vs {bytes_gb:.2f}GB scanned"
            evidence = f"Shuffle ratio {shuffle_ratio:.1%} indicates data explosion"
            recommendation = "Likely Cartesian join or missing join predicate - review JOIN conditions"
        elif shuffle_ratio > 0.5:
            severity = "WARNING"
            indicator = f"⚠️ NETWORK: {shuffle_gb:.2f}GB shuffled ({shuffle_ratio:.0%} of scanned)"
            evidence = f"High network I/O relative to data scanned"
            recommendation = "Consider optimizing join order or adding broadcast hints"
        elif shuffle_ratio == 0.0:
            severity = "OPTIMAL"
            indicator = f"✓ NETWORK: 0GB shuffled (no network transfer)"
            evidence = "No shuffle detected - likely using broadcast join"
            recommendation = "Network I/O is optimal for this query"
        else:
            severity = "NORMAL"
            indicator = f"✓ NETWORK: {shuffle_gb:.2f}GB shuffled ({shuffle_ratio:.0%} of scanned)"
            evidence = "Normal shuffle volume for distributed query"
            recommendation = "Network I/O is within normal range"

        return textwrap.dedent(f"""\
            <network_transfer>
              <shuffle_bytes>{metrics.shuffle_bytes or 0}</shuffle_bytes>
              <shuffle_ratio>{shuffle_ratio:.2f}</shuffle_ratio>
              <severity>{severity}</severity>
              <indicator>{indicator}</indicator>
              <evidence>{evidence}</evidence>
              <recommendation>{recommendation}</recommendation>
            </network_transfer>
        """)

    def _format_io_efficiency(self, metrics: ExecutionMetrics) -> str:
        """Format I/O efficiency (data scanning) metrics.

        Args:
            metrics: Execution performance metrics

        Returns:
            XML-formatted I/O efficiency section
        """
        bytes_gb = metrics.bytes_scanned / (1024**3)
        rows_scanned = metrics.rows_scanned

        # Determine severity based on data volume
        if bytes_gb > 100:
            severity = "CRITICAL"
            indicator = f"❗ I/O: {bytes_gb:.2f}GB scanned ({rows_scanned:,} rows)"
            evidence = f"Very large scan - potential full table scan"
            recommendation = "Add partition pruning or column selection to reduce scan volume"
        elif bytes_gb > 10:
            severity = "WARNING"
            indicator = f"⚠️ I/O: {bytes_gb:.2f}GB scanned ({rows_scanned:,} rows)"
            evidence = f"Large scan volume"
            recommendation = "Consider adding WHERE filters or partition pruning"
        else:
            severity = "NORMAL"
            indicator = f"✓ I/O: {bytes_gb:.2f}GB scanned ({rows_scanned:,} rows)"
            evidence = "Reasonable scan volume"
            recommendation = "Data scan volume is acceptable"

        return f"""    <io_efficiency>
      <bytes_scanned>{metrics.bytes_scanned}</bytes_scanned>
      <bytes_scanned_gb>{bytes_gb:.2f}</bytes_scanned_gb>
      <rows_scanned>{rows_scanned}</rows_scanned>
      <severity>{severity}</severity>
      <indicator>{indicator}</indicator>
      <evidence>{evidence}</evidence>
      <recommendation>{recommendation}</recommendation>
    </io_efficiency>
"""

    def _format_memory_utilization(
        self,
        metrics: ExecutionMetrics,
        cluster: Optional[ClusterConfig]
    ) -> str:
        """Format memory utilization metrics.

        Args:
            metrics: Execution performance metrics
            cluster: Optional cluster configuration

        Returns:
            XML-formatted memory utilization section or empty string
        """
        # Only include memory section if we have peak_memory_mb > 0
        if not metrics.peak_memory_mb or metrics.peak_memory_mb == 0:
            return ""

        peak_memory_mb = metrics.peak_memory_mb

        # Determine severity based on memory usage per executor
        if cluster:
            executor_memory_gb = cluster.executor_memory_gb()
            executor_memory_mb = executor_memory_gb * 1024
            memory_percent = (peak_memory_mb / executor_memory_mb) * 100 if executor_memory_mb > 0 else 0

            if memory_percent > 90:
                severity = "CRITICAL"
                indicator = f"❗ MEMORY: {peak_memory_mb}MB used ({memory_percent:.0f}% of {executor_memory_gb}GB executor)"
                evidence = "HIGH RISK - Memory usage extremely high - risk of OOM errors"
                recommendation = "Reduce data shuffle, increase partition count, or scale cluster"
            elif memory_percent > 70:
                severity = "WARNING"
                indicator = f"⚠️ MEMORY: {peak_memory_mb}MB used ({memory_percent:.0f}% of {executor_memory_gb}GB executor)"
                evidence = "MODERATE RISK - Memory usage approaching limits"
                recommendation = "Monitor for OOM errors or consider scaling cluster"
            else:
                severity = "NORMAL"
                indicator = f"✓ MEMORY: {peak_memory_mb}MB used ({memory_percent:.0f}% of {executor_memory_gb}GB executor)"
                evidence = "LOW RISK - Memory usage within safe limits"
                recommendation = "Memory utilization is acceptable"

            return f"""    <memory_utilization>
      <peak_memory_mb>{peak_memory_mb}</peak_memory_mb>
      <executor_memory_gb>{executor_memory_gb}</executor_memory_gb>
      <memory_percent>{memory_percent:.1f}</memory_percent>
      <severity>{severity}</severity>
      <indicator>{indicator}</indicator>
      <evidence>{evidence}</evidence>
      <recommendation>{recommendation}</recommendation>
    </memory_utilization>
"""
        else:
            # No cluster context - just report raw memory
            severity = "INFO"
            indicator = f"ℹ️ MEMORY: {peak_memory_mb}MB peak usage"
            evidence = "Peak memory consumption observed"
            recommendation = "Memory metrics available but cluster config needed for capacity analysis"

            return f"""    <memory_utilization>
      <peak_memory_mb>{peak_memory_mb}</peak_memory_mb>
      <severity>{severity}</severity>
      <indicator>{indicator}</indicator>
      <evidence>{evidence}</evidence>
      <recommendation>{recommendation}</recommendation>
    </memory_utilization>
"""

    def _format_cluster_configuration(self, cluster: ClusterConfig) -> str:
        """Format cluster configuration details.

        Args:
            cluster: Cluster configuration

        Returns:
            XML-formatted cluster configuration section
        """
        total_memory = cluster.total_cluster_memory()

        return f"""  <cluster_configuration>
    <node_type>{cluster.node_type}</node_type>
    <executor_count>{cluster.executor_count}</executor_count>
    <executor_memory>{cluster.executor_memory}</executor_memory>
    <total_memory_gb>{total_memory}</total_memory_gb>
  </cluster_configuration>

"""

    def _format_all_violations(self, violations: List[Violation]) -> str:
        """Format all violations for cross-reasoning with enumeration summary.

        This method prepends a violations inventory to the XML context to ensure
        the LLM sees all violations upfront and addresses each one in recommendations.

        Structure:
        1. <violations_summary> - Numbered list for quick enumeration
        2. Coverage requirements and warnings
        3. Detailed <violation> entries with full context

        Args:
            violations: All detected violations

        Returns:
            XML-formatted violations section with enumeration summary
        """
        self._logger.info(f"=== FORMATTING ALL VIOLATIONS ({len(violations)} total) ===")

        parts = [f"  <all_violations count=\"{len(violations)}\">\n"]

        # PHASE 1 ENHANCEMENT: Add violations summary section for enumeration
        parts.append("    <violations_summary>\n")
        parts.append(f"      <note>📋 VIOLATION INVENTORY: {len(violations)} issues detected</note>\n")
        parts.append("      <note>⚠️ CRITICAL: YOU MUST ADDRESS ALL VIOLATIONS LISTED BELOW ⚠️</note>\n")
        parts.append("      <enumeration>\n")

        # Create numbered list of violations for easy enumeration
        for idx, v in enumerate(violations, 1):
            severity_indicator = self._get_severity_indicator(v.severity)
            parts.append(f"        <item id=\"{idx}\">{severity_indicator} | {v.pattern} | Line {v.line_number}</item>\n")

        parts.append("      </enumeration>\n")
        parts.append(f"      <coverage_requirement>Generate {len(violations)} recommendations - one for EACH violation above</coverage_requirement>\n")
        parts.append("      <duplication_warning>DO NOT create duplicate recommendations. Each must address a DIFFERENT violation.</duplication_warning>\n")
        parts.append("      <cross_reference_instruction>Before generating recommendations, count violations above and verify coverage.</cross_reference_instruction>\n")
        parts.append("    </violations_summary>\n\n")

        # Original detailed violation entries
        parts.append("    <detailed_violations>\n")
        parts.append("      <note>Detailed context for each violation listed in summary above</note>\n\n")

        for idx, v in enumerate(violations, 1):
            severity_indicator = self._get_severity_indicator(v.severity)
            parts.append(f"      <violation id=\"{idx}\" pattern=\"{v.pattern}\" severity=\"{v.severity.upper()}\" line=\"{v.line_number}\">\n")
            parts.append(f"        <severity_indicator>{severity_indicator}</severity_indicator>\n")
            if v.default_message:
                parts.append(f"        <description>{v.default_message}</description>\n")
            if v.fragment:
                parts.append(f"        <code_fragment>{v.fragment}</code_fragment>\n")
            if v.fix_suggestion:
                parts.append(f"        <fix_suggestion>{v.fix_suggestion}</fix_suggestion>\n")
            parts.append(f"      </violation>\n\n")

        parts.append("    </detailed_violations>\n")
        parts.append("  </all_violations>\n\n")

        result = "".join(parts)
        self._logger.info(f"Generated violations XML: {len(result)} chars")
        self._logger.info(f"Includes <violations_summary>: {'<violations_summary>' in result}")
        self._logger.info(f"Includes <all_violations>: {'<all_violations' in result}")

        return result

    def _get_severity_indicator(self, severity: str) -> str:
        """Get visual indicator for severity level.

        Args:
            severity: Severity level string

        Returns:
            Emoji indicator for severity
        """
        severity_lower = severity.lower() if isinstance(severity, str) else str(severity).lower()
        if "critical" in severity_lower:
            return "❗ CRITICAL"
        elif "high" in severity_lower:
            return "⚠️ HIGH"
        elif "medium" in severity_lower:
            return "⚠️ MEDIUM"
        elif "low" in severity_lower:
            return "ℹ️ LOW"
        else:
            return f"• {severity.upper()}"

    def _format_custom_guidelines(self, guidelines_text: str) -> str:
        """Format custom Markdown guidelines.

        Args:
            guidelines_text: Custom guidelines in Markdown format

        Returns:
            XML-formatted custom guidelines section
        """
        return f"""  <custom_guidelines format="markdown">
{guidelines_text}
  </custom_guidelines>

"""

    def _format_table_schemas(self, table_schemas: List[dict[str, Any]]) -> str:
        """Format table schemas for LLM context.

        Provides structured table metadata with PII detection and compliance tags
        to enable industry-aware analysis.

        Args:
            table_schemas: List of schema dicts from fetch_table_schema()

        Returns:
            XML-formatted table schemas section
        """
        parts = [f"  <table_schemas count=\"{len(table_schemas)}\">\n"]
        parts.append("    <note>These schemas are auto-fetched from Databricks. ")
        parts.append("Consider PII/sensitive data when making recommendations.</note>\n\n")

        for schema in table_schemas:
            parts.append(self._format_single_table_schema(schema))

        parts.append("  </table_schemas>\n\n")
        return "".join(parts)

    def _format_single_table_schema(self, schema: dict[str, Any]) -> str:
        """Format a single table schema with compliance context.

        Args:
            schema: Schema dict from fetch_table_schema()

        Returns:
            XML-formatted single table schema
        """
        parts = [f"    <table name=\"{schema['table_name']}\">\n"]

        # Table-level comment
        if schema.get("table_comment"):
            parts.append(f"      <description>{schema['table_comment']}</description>\n")

        # Unity Catalog table tags (if available)
        table_tags = schema.get("table_tags", {})
        if table_tags:
            parts.append("      <table_tags>\n")
            for tag_name, tag_value in table_tags.items():
                parts.append(f'        <tag name="{tag_name}" value="{tag_value}" />\n')
            parts.append("      </table_tags>\n")

        # Security/compliance tags (if any detected from comments)
        if schema.get("security_tags"):
            tags_str = ", ".join(schema["security_tags"])
            parts.append(f"      <compliance_tags>{tags_str}</compliance_tags>\n")

        # PII indicators summary
        if schema.get("pii_indicators"):
            pii_str = ", ".join(schema["pii_indicators"])
            parts.append(f"      <pii_detected>{pii_str}</pii_detected>\n")
            parts.append("      <warning>⚠️ This table contains PII. ")
            parts.append("Ensure data handling complies with GDPR/regulatory requirements.")
            parts.append("</warning>\n")

        # Columns
        columns = schema.get("columns", [])
        parts.append(f"      <columns count=\"{len(columns)}\">\n")

        # Group columns: PII columns first, then others
        pii_columns = [c for c in columns if c.get("pii_detected")]
        regular_columns = [c for c in columns if not c.get("pii_detected")]

        # Format PII columns with special highlighting
        if pii_columns:
            parts.append("        <!-- PII/Sensitive Columns -->\n")
            for col in pii_columns:
                parts.append(self._format_column(col, highlight_pii=True))

        # Format regular columns (condensed if many)
        if len(regular_columns) > 10:
            # Show first 5 and last 5, summarize rest
            parts.append("        <!-- Key Columns (showing first 5 and last 5) -->\n")
            for col in regular_columns[:5]:
                parts.append(self._format_column(col, highlight_pii=False))

            parts.append(f"        <!-- ... {len(regular_columns) - 10} more columns ... -->\n")

            for col in regular_columns[-5:]:
                parts.append(self._format_column(col, highlight_pii=False))
        else:
            # Show all columns
            for col in regular_columns:
                parts.append(self._format_column(col, highlight_pii=False))

        parts.append("      </columns>\n")
        parts.append("    </table>\n\n")

        return "".join(parts)

    def _format_column(self, col: dict[str, Any], highlight_pii: bool = False) -> str:
        """Format a single column with PII highlighting and Unity Catalog tags.

        Args:
            col: Column dict with name, type, comment, nullable, pii_detected, column_tags
            highlight_pii: Whether to highlight this column as containing PII

        Returns:
            XML-formatted column element
        """
        col_name = col["name"]
        col_type = col["type"]
        nullable = "nullable" if col.get("nullable") else "required"
        column_tags = col.get("column_tags", {})

        # Build column opening tag with attributes
        attrs = [f'name="{col_name}"', f'type="{col_type}"', nullable]

        if highlight_pii and col.get("pii_detected"):
            pii_types = ", ".join(col["pii_detected"])
            attrs.append(f'pii="{pii_types}"')

        # If column has Unity Catalog tags, format as multi-line
        if column_tags:
            parts = [f'        <column {" ".join(attrs)}>\n']

            # Add Unity Catalog tags
            parts.append("          <column_tags>\n")
            for tag_name, tag_value in column_tags.items():
                parts.append(f'            <tag name="{tag_name}" value="{tag_value}" />\n')
            parts.append("          </column_tags>\n")

            # Add comment or PII indicator
            if highlight_pii:
                parts.append("          🔒 Sensitive\n")
            elif col.get("comment"):
                comment = col["comment"].replace("<", "&lt;").replace(">", "&gt;")
                parts.append(f"          {comment}\n")

            parts.append("        </column>\n")
            return "".join(parts)

        # Simple single-line format if no tags
        if highlight_pii and col.get("pii_detected"):
            return f'        <column {" ".join(attrs)}>🔒 Sensitive</column>\n'
        elif col.get("comment"):
            comment = col["comment"].replace("<", "&lt;").replace(">", "&gt;")
            return f'        <column {" ".join(attrs)}>{comment}</column>\n'
        else:
            return f'        <column {" ".join(attrs)} />\n'

