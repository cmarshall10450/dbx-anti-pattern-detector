"""Databricks integration for Cloud CEO.

This submodule provides Databricks-specific functionality:
- Automatic query data fetching from system tables
- Statement ID discovery helpers
- Workspace-wide analysis and reporting
"""

from cloud_ceo.databricks.discovery import (
    StatementInfo,
    find_expensive_queries,
    find_my_recent_queries,
    find_slow_queries,
    get_statement_id_from_notebook_result,
    print_query_list,
    search_queries_by_text,
)
from cloud_ceo.databricks.fetch import (
    DatabricksQueryData,
    DatabricksSystemTables,
    NodeTypeResolver,
    analyze_databricks_query,
    fetch_databricks_query,
    get_node_type_resolver,
)
from cloud_ceo.databricks.workspace import WorkspaceReport, analyze_workspace

__all__ = [
    # Core Databricks analysis
    "analyze_databricks_query",
    "fetch_databricks_query",
    "DatabricksQueryData",
    "DatabricksSystemTables",
    # Node type resolution
    "NodeTypeResolver",
    "get_node_type_resolver",
    # Query discovery
    "find_my_recent_queries",
    "find_slow_queries",
    "find_expensive_queries",
    "search_queries_by_text",
    "get_statement_id_from_notebook_result",
    "print_query_list",
    "StatementInfo",
    # Workspace analysis
    "analyze_workspace",
    "WorkspaceReport",
]
