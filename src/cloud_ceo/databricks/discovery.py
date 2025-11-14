"""Helper functions to discover and retrieve Databricks statement IDs.

This module provides utilities to help users find statement IDs from:
1. Query history UI
2. Databricks SQL warehouse execution
3. Notebook execution results
4. System tables queries
5. Recent query lookups
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StatementInfo:
    """Minimal info about a statement for discovery.

    Attributes:
        statement_id: The statement ID to use with analyze_databricks_query()
        query_text_preview: First 100 chars of query text
        user: User who executed the query
        start_time: When query started
        duration_ms: Execution duration
        status: FINISHED, FAILED, or CANCELED
        cost_estimate: Estimated cost in USD (if available)
    """

    statement_id: str
    query_text_preview: str
    user: str
    start_time: datetime
    duration_ms: int
    status: str
    cost_estimate: Optional[float] = None

    def __str__(self) -> str:
        """Human-readable representation."""
        cost_str = f"${self.cost_estimate:.2f}" if self.cost_estimate else "N/A"
        return (
            f"{self.statement_id} | {self.user} | "
            f"{self.duration_ms / 1000:.1f}s | {cost_str} | "
            f"{self.query_text_preview[:50]}..."
        )


def find_my_recent_queries(
    hours: int = 24,
    limit: int = 20,
    status: Optional[Literal["FINISHED", "FAILED", "CANCELED"]] = None,
) -> list[StatementInfo]:
    """Find your recent queries from system.query.history.

    This is the easiest way to get statement IDs for queries you've run.

    Args:
        hours: How far back to look (default: 24 hours)
        limit: Maximum number of results (default: 20)
        status: Filter by status (default: all statuses)

    Returns:
        List of StatementInfo objects with statement IDs

    Example:
        >>> from cloud_ceo.databricks import find_my_recent_queries
        >>>
        >>> # Find my queries from last 24 hours
        >>> queries = find_my_recent_queries()
        >>> for q in queries:
        ...     print(f"{q.statement_id}: {q.query_text_preview}")
        >>>
        >>> # Analyze the slowest one
        >>> from cloud_ceo.databricks import analyze_databricks_query
        >>> result = analyze_databricks_query(queries[0].statement_id)
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("CloudCEO").getOrCreate()

        # Get current user
        current_user = spark.sql("SELECT current_user()").collect()[0][0]

        # Build query
        status_filter = f"AND execution_status = '{status}'" if status else ""

        sql = f"""
        SELECT
            statement_id,
            LEFT(statement_text, 100) as query_preview,
            executed_by as user,
            start_time,
            execution_duration_ms,
            execution_status,
            total_task_duration_ms * 0.00001 as cost_estimate
        FROM system.query.history
        WHERE executed_by = '{current_user}'
        AND start_time >= current_timestamp() - INTERVAL {hours} HOURS
        {status_filter}
        ORDER BY start_time DESC
        LIMIT {limit}
        """

        logger.info(f"Finding recent queries for user: {current_user}")

        df = spark.sql(sql)
        rows = df.collect()

        results = [
            StatementInfo(
                statement_id=row.statement_id,
                query_text_preview=row.query_preview,
                user=row.user,
                start_time=row.start_time,
                duration_ms=row.execution_duration_ms or 0,
                status=row.execution_status,
                cost_estimate=row.cost_estimate,
            )
            for row in rows
        ]

        logger.info(f"Found {len(results)} queries")
        return results

    except ImportError:
        raise RuntimeError(
            "PySpark required. Install with: pip install cloud-ceo-dbx[databricks]"
        )
    except Exception as e:
        logger.error(f"Failed to fetch recent queries: {e}")
        return []


def find_slow_queries(
    hours: int = 24,
    min_duration_seconds: int = 60,
    limit: int = 20,
    user: Optional[str] = None,
) -> list[StatementInfo]:
    """Find slow queries from system.query.history.

    Useful for identifying performance problems to analyze.

    Args:
        hours: How far back to look
        min_duration_seconds: Minimum query duration in seconds
        limit: Maximum number of results
        user: Filter by specific user (default: current user)

    Returns:
        List of StatementInfo objects, sorted by duration (slowest first)

    Example:
        >>> from cloud_ceo.databricks import find_slow_queries, analyze_databricks_query
        >>>
        >>> # Find queries that took > 60 seconds
        >>> slow_queries = find_slow_queries(min_duration_seconds=60)
        >>>
        >>> # Analyze the slowest one
        >>> result = analyze_databricks_query(slow_queries[0].statement_id)
        >>> print(result.summary())
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("CloudCEO").getOrCreate()

        # Get current user if not specified
        if user is None:
            user = spark.sql("SELECT current_user()").collect()[0][0]

        min_duration_ms = min_duration_seconds * 1000

        sql = f"""
        SELECT
            statement_id,
            LEFT(statement_text, 100) as query_preview,
            executed_by as user,
            start_time,
            execution_duration_ms,
            execution_status,
            total_task_duration_ms * 0.00001 as cost_estimate
        FROM system.query.history
        WHERE executed_by = '{user}'
        AND start_time >= current_timestamp() - INTERVAL {hours} HOURS
        AND execution_duration_ms >= {min_duration_ms}
        AND execution_status = 'FINISHED'
        ORDER BY execution_duration_ms DESC
        LIMIT {limit}
        """

        logger.info(f"Finding slow queries (>{min_duration_seconds}s)")

        df = spark.sql(sql)
        rows = df.collect()

        results = [
            StatementInfo(
                statement_id=row.statement_id,
                query_text_preview=row.query_preview,
                user=row.user,
                start_time=row.start_time,
                duration_ms=row.execution_duration_ms or 0,
                status=row.execution_status,
                cost_estimate=row.cost_estimate,
            )
            for row in rows
        ]

        logger.info(f"Found {len(results)} slow queries")
        return results

    except Exception as e:
        logger.error(f"Failed to fetch slow queries: {e}")
        return []


def find_expensive_queries(
    hours: int = 24,
    min_cost_usd: float = 1.0,
    limit: int = 20,
    user: Optional[str] = None,
) -> list[StatementInfo]:
    """Find expensive queries from system.query.history.

    Useful for cost optimization.

    Args:
        hours: How far back to look
        min_cost_usd: Minimum estimated cost in USD
        limit: Maximum number of results
        user: Filter by specific user (default: current user)

    Returns:
        List of StatementInfo objects, sorted by cost (most expensive first)

    Example:
        >>> from cloud_ceo.databricks import find_expensive_queries, analyze_databricks_query
        >>>
        >>> # Find queries that cost > $10
        >>> expensive = find_expensive_queries(min_cost_usd=10)
        >>>
        >>> # Analyze the most expensive one
        >>> result = analyze_databricks_query(expensive[0].statement_id)
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("CloudCEO").getOrCreate()

        if user is None:
            user = spark.sql("SELECT current_user()").collect()[0][0]

        # Convert cost to task duration for filtering
        min_task_duration = min_cost_usd / 0.00001

        sql = f"""
        SELECT
            statement_id,
            LEFT(statement_text, 100) as query_preview,
            executed_by as user,
            start_time,
            execution_duration_ms,
            execution_status,
            total_task_duration_ms * 0.00001 as cost_estimate
        FROM system.query.history
        WHERE executed_by = '{user}'
        AND start_time >= current_timestamp() - INTERVAL {hours} HOURS
        AND total_task_duration_ms >= {min_task_duration}
        AND execution_status = 'FINISHED'
        ORDER BY total_task_duration_ms DESC
        LIMIT {limit}
        """

        logger.info(f"Finding expensive queries (>${min_cost_usd})")

        df = spark.sql(sql)
        rows = df.collect()

        results = [
            StatementInfo(
                statement_id=row.statement_id,
                query_text_preview=row.query_preview,
                user=row.user,
                start_time=row.start_time,
                duration_ms=row.execution_duration_ms or 0,
                status=row.execution_status,
                cost_estimate=row.cost_estimate,
            )
            for row in rows
        ]

        logger.info(f"Found {len(results)} expensive queries")
        return results

    except Exception as e:
        logger.error(f"Failed to fetch expensive queries: {e}")
        return []


def search_queries_by_text(
    search_text: str,
    hours: int = 168,  # 7 days
    limit: int = 20,
    user: Optional[str] = None,
) -> list[StatementInfo]:
    """Search for queries containing specific text.

    Useful for finding queries that access specific tables or use certain patterns.

    Args:
        search_text: Text to search for in query (case-insensitive)
        hours: How far back to look (default: 7 days)
        limit: Maximum number of results
        user: Filter by specific user (default: current user)

    Returns:
        List of StatementInfo objects matching the search

    Example:
        >>> from cloud_ceo.databricks import search_queries_by_text
        >>>
        >>> # Find all queries using a specific table
        >>> queries = search_queries_by_text("large_table")
        >>>
        >>> # Find queries with CROSS JOIN
        >>> risky_queries = search_queries_by_text("CROSS JOIN")
    """
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("CloudCEO").getOrCreate()

        if user is None:
            user = spark.sql("SELECT current_user()").collect()[0][0]

        sql = f"""
        SELECT
            statement_id,
            LEFT(statement_text, 100) as query_preview,
            executed_by as user,
            start_time,
            execution_duration_ms,
            execution_status,
            total_task_duration_ms * 0.00001 as cost_estimate
        FROM system.query.history
        WHERE executed_by = '{user}'
        AND start_time >= current_timestamp() - INTERVAL {hours} HOURS
        AND LOWER(statement_text) LIKE '%{search_text.lower()}%'
        ORDER BY start_time DESC
        LIMIT {limit}
        """

        logger.info(f"Searching queries containing: {search_text}")

        df = spark.sql(sql)
        rows = df.collect()

        results = [
            StatementInfo(
                statement_id=row.statement_id,
                query_text_preview=row.query_preview,
                user=row.user,
                start_time=row.start_time,
                duration_ms=row.execution_duration_ms or 0,
                status=row.execution_status,
                cost_estimate=row.cost_estimate,
            )
            for row in rows
        ]

        logger.info(f"Found {len(results)} matching queries")
        return results

    except Exception as e:
        logger.error(f"Failed to search queries: {e}")
        return []


def get_statement_id_from_notebook_result(result) -> Optional[str]:
    """Extract statement ID from a notebook query result.

    When you run SQL in a Databricks notebook, the result object contains
    the statement ID. This helper extracts it.

    Args:
        result: Result object from spark.sql() or display()

    Returns:
        Statement ID string, or None if not found

    Example:
        >>> # In a Databricks notebook
        >>> result = spark.sql("SELECT * FROM table")
        >>> display(result)
        >>>
        >>> # Get the statement ID
        >>> from cloud_ceo.databricks import get_statement_id_from_notebook_result
        >>> statement_id = get_statement_id_from_notebook_result(result)
        >>>
        >>> # Analyze it
        >>> from cloud_ceo.databricks import analyze_databricks_query
        >>> analysis = analyze_databricks_query(statement_id)
    """
    try:
        # Try to get from query execution metadata
        if hasattr(result, "_jdf"):
            # PySpark DataFrame
            query_execution = result._jdf.queryExecution()
            if hasattr(query_execution, "executedPlan"):
                plan = str(query_execution.executedPlan())
                # Statement ID is typically in the plan
                if "statement_id=" in plan.lower():
                    start = plan.lower().index("statement_id=") + len("statement_id=")
                    end = plan.find(",", start)
                    if end == -1:
                        end = plan.find(")", start)
                    return plan[start:end].strip()

        # Try Databricks-specific attributes
        if hasattr(result, "statement_id"):
            return result.statement_id

        logger.warning("Could not extract statement_id from result object")
        return None

    except Exception as e:
        logger.error(f"Error extracting statement_id: {e}")
        return None


def print_query_list(queries: list[StatementInfo]) -> None:
    """Pretty-print a list of queries.

    Args:
        queries: List of StatementInfo objects to display

    Example:
        >>> from cloud_ceo.databricks import find_my_recent_queries, print_query_list
        >>> queries = find_my_recent_queries()
        >>> print_query_list(queries)
    """
    if not queries:
        print("No queries found.")
        return

    print(f"\nFound {len(queries)} queries:\n")
    print(f"{'#':<4} {'Statement ID':<25} {'Duration':<10} {'Cost':<10} {'Preview':<50}")
    print("-" * 100)

    for i, q in enumerate(queries, 1):
        duration = f"{q.duration_ms / 1000:.1f}s"
        cost = f"${q.cost_estimate:.2f}" if q.cost_estimate else "N/A"
        preview = q.query_text_preview[:47] + "..." if len(q.query_text_preview) > 50 else q.query_text_preview

        print(f"{i:<4} {q.statement_id:<25} {duration:<10} {cost:<10} {preview:<50}")

    print()
