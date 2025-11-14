"""Databricks workspace-wide query analysis.

This module provides utilities for workspace administrators to:
1. Analyze query history across the workspace
2. Identify worst performing queries
3. Generate reports by user, cost, and violation patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

from cloud_ceo.databricks.fetch import DatabricksSystemTables
from cloud_ceo.models import AnalysisResult

logger = structlog.get_logger(__name__)


@dataclass
class WorkspaceReport:
    """Workspace-wide analysis report.

    Attributes:
        generated_at: When report was generated
        period_days: Time period analyzed
        total_queries: Total queries analyzed
        queries_with_violations: Number of queries with violations
        total_violations: Total violations found
        total_cost_usd: Total cost of analyzed queries
        worst_queries: Top N worst performing queries
        violation_breakdown: Breakdown by violation type
        user_breakdown: Breakdown by user
    """

    generated_at: datetime
    period_days: int
    total_queries: int
    queries_with_violations: int
    total_violations: int
    total_cost_usd: float
    worst_queries: list[AnalysisResult]
    violation_breakdown: dict[str, int]
    user_breakdown: dict[str, int]

    def summary(self) -> str:
        """Generate human-readable report summary."""
        lines = [
            "\n" + "=" * 80,
            "Cloud CEO Workspace Report",
            "=" * 80,
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Period: Last {self.period_days} days",
            "",
            "Overview:",
            f"  Total Queries Analyzed: {self.total_queries:,}",
            f"  Queries with Violations: {self.queries_with_violations:,} "
            f"({self.queries_with_violations / max(self.total_queries, 1) * 100:.1f}%)",
            f"  Total Violations: {self.total_violations:,}",
            f"  Total Cost: ${self.total_cost_usd:,.2f}",
            "",
            "Top Violation Types:",
        ]

        for violation_type, count in sorted(
            self.violation_breakdown.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            lines.append(f"  {violation_type}: {count:,}")

        lines.extend([
            "",
            f"Top {min(10, len(self.worst_queries))} Worst Performers:",
        ])

        for i, query in enumerate(self.worst_queries[:10], 1):
            cost = f"${query.cost_usd:.2f}" if query.cost_usd else "N/A"
            duration = f"{query.duration_seconds:.1f}s" if query.duration_seconds else "N/A"
            lines.append(
                f"  {i}. Query {query.databricks_query_id or query.query_id} | "
                f"{query.violation_count} violations | "
                f"Cost: {cost} | Duration: {duration}"
            )

        lines.append("=" * 80 + "\n")

        return "\n".join(lines)


def analyze_workspace(
    days: int = 7,
    min_duration_ms: Optional[int] = 10000,  # 10 seconds
    min_cost_usd: Optional[float] = None,
    users: Optional[list[str]] = None,
    limit: int = 1000,
    enable_llm: bool = False,
    top_n: int = 50,
) -> WorkspaceReport:
    """Analyze Databricks workspace for worst performing queries.

    This is the main entry point for workspace administrators to identify
    queries with anti-patterns across the entire workspace.

    Args:
        days: Number of days to analyze (query history window)
        min_duration_ms: Minimum query duration to analyze (default: 10 seconds)
        min_cost_usd: Minimum query cost to analyze (optional)
        users: Filter by specific users (optional, default: all users)
        limit: Maximum queries to fetch and analyze (default: 1000)
        enable_llm: Enable LLM analysis for violations (default: False)
        top_n: Number of top violators to include in report (default: 50)

    Returns:
        WorkspaceReport with analysis results

    Example:
        >>> from cloud_ceo.databricks import analyze_workspace
        >>>
        >>> # Analyze last 7 days, queries costing > $100
        >>> report = analyze_workspace(
        ...     days=7,
        ...     min_cost_usd=100,
        ...     enable_llm=True,
        ...     top_n=20
        ... )
        >>> print(report.summary())
        >>>
        >>> # Analyze specific users
        >>> report = analyze_workspace(
        ...     days=30,
        ...     users=["data_engineer@company.com"],
        ...     enable_llm=False
        ... )
    """
    from cloud_ceo.core import analyze

    logger.info(f"Starting workspace analysis for last {days} days")

    # Connect to Databricks system tables
    system_tables = DatabricksSystemTables()

    # Fetch query history from system tables
    logger.info("Fetching query history from system tables...")

    queries = system_tables.fetch_recent_queries(
        hours=days * 24,
        min_duration_ms=min_duration_ms,
        user=users[0] if users and len(users) == 1 else None,
        limit=limit,
    )

    # Filter by users if multiple specified
    if users and len(users) > 1:
        queries = [q for q in queries if q.user in users]

    # Filter by cost if specified
    if min_cost_usd:
        queries = [
            q for q in queries
            if q.execution_metrics.cost_usd and q.execution_metrics.cost_usd >= min_cost_usd
        ]

    if not queries:
        logger.warning("No queries found matching criteria")
        return WorkspaceReport(
            generated_at=datetime.now(),
            period_days=days,
            total_queries=0,
            queries_with_violations=0,
            total_violations=0,
            total_cost_usd=0.0,
            worst_queries=[],
            violation_breakdown={},
            user_breakdown={},
        )

    # Analyze each query
    logger.info(f"Analyzing {len(queries)} queries...")
    results: list[AnalysisResult] = []

    for i, query_data in enumerate(queries, 1):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(queries)} queries analyzed")

        try:
            result = analyze(
                query_data.query_text,
                query_id=query_data.statement_id,
                execution_metrics=query_data.execution_metrics,
                cluster_config=query_data.cluster_config,
                enable_llm=enable_llm,
                databricks_query_id=query_data.statement_id,
                user=query_data.user,
                warehouse_id=query_data.warehouse_id,
            )

            results.append(result)

        except Exception as e:
            logger.error(f"Failed to analyze query {query_data.statement_id}: {e}")

    # Generate report
    total_violations = sum(r.violation_count for r in results)
    queries_with_violations = sum(1 for r in results if r.has_violations())
    total_cost = sum(
        r.execution_metrics.cost_usd or 0.0 for r in results if r.execution_metrics
    )

    # Violation breakdown
    violation_breakdown: dict[str, int] = {}
    for result in results:
        for violation in result.violations:
            violation_breakdown[violation.rule_id] = (
                violation_breakdown.get(violation.rule_id, 0) + 1
            )

    # User breakdown
    user_breakdown: dict[str, int] = {}
    for result in results:
        if result.user:
            user_breakdown[result.user] = user_breakdown.get(result.user, 0) + result.violation_count

    # Rank worst queries (by violation count, then cost)
    worst_queries = sorted(
        results,
        key=lambda r: (
            r.violation_count,
            r.execution_metrics.cost_usd if r.execution_metrics else 0,
        ),
        reverse=True,
    )[:top_n]

    report = WorkspaceReport(
        generated_at=datetime.now(),
        period_days=days,
        total_queries=len(results),
        queries_with_violations=queries_with_violations,
        total_violations=total_violations,
        total_cost_usd=total_cost,
        worst_queries=worst_queries,
        violation_breakdown=violation_breakdown,
        user_breakdown=user_breakdown,
    )

    logger.info(f"Workspace analysis complete: {total_violations} violations found")

    return report
