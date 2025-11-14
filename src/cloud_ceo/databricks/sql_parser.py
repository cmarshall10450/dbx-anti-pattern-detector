"""SQL parsing utilities for table extraction and analysis.

This module provides utilities to parse SQL queries and extract referenced tables,
enabling automatic schema enrichment for LLM context.
"""

from typing import Optional

import sqlglot
import structlog

logger = structlog.get_logger(__name__)


def extract_tables_from_query(sql_query: str) -> list[str]:
    """Parse SQL query and extract all referenced table names.

    Uses sqlglot for proper SQL parsing (already in dependencies).
    Handles FROM clauses, JOINs, CTEs, subqueries, and table aliases.

    Args:
        sql_query: SQL query string (Databricks dialect)

    Returns:
        List of fully qualified table names (catalog.schema.table format)
        Returns empty list if parsing fails or no tables found

    Example:
        >>> sql = '''
        ... SELECT c.customer_id, o.order_date
        ... FROM main.sales.customers c
        ... JOIN main.sales.orders o ON c.id = o.customer_id
        ... WHERE c.country = 'UK'
        ... '''
        >>> tables = extract_tables_from_query(sql)
        >>> print(tables)
        ['main.sales.customers', 'main.sales.orders']
    """
    try:
        # Parse SQL with Databricks dialect
        ast = sqlglot.parse_one(sql_query, dialect="databricks")

        # Extract all table references
        tables = set()

        # Find all table nodes in AST
        for table in ast.find_all(sqlglot.exp.Table):
            # Build fully qualified table name
            parts = []

            if table.catalog:
                parts.append(table.catalog)
            if table.db:
                parts.append(table.db)
            if table.name:
                parts.append(table.name)

            if parts:
                # Normalize to catalog.schema.table format
                table_name = ".".join(parts)

                # Handle case where only 2 parts are provided (schema.table)
                # Assume 'main' catalog if not specified (Databricks default)
                if len(parts) == 2:
                    table_name = f"main.{table_name}"
                elif len(parts) == 1:
                    # Single part - likely just table name
                    # Skip or assume default catalog.schema
                    logger.debug(f"Skipping unqualified table: {table_name}")
                    continue

                tables.add(table_name)

        logger.debug(f"Extracted {len(tables)} tables from query: {list(tables)}")
        return sorted(list(tables))

    except Exception as e:
        logger.warning(f"Failed to parse SQL query for table extraction: {e}")
        return []


def normalize_table_name(
    table_name: str,
    default_catalog: str = "main",
    default_schema: str = "default"
) -> Optional[str]:
    """Normalize table name to fully qualified format (catalog.schema.table).

    Args:
        table_name: Table name in any format (table, schema.table, catalog.schema.table)
        default_catalog: Default catalog to use if not specified
        default_schema: Default schema to use if not specified

    Returns:
        Fully qualified table name or None if invalid

    Example:
        >>> normalize_table_name("customers")
        'main.default.customers'
        >>> normalize_table_name("sales.orders")
        'main.sales.orders'
        >>> normalize_table_name("prod.analytics.revenue")
        'prod.analytics.revenue'
    """
    parts = table_name.split(".")

    if len(parts) == 3:
        return table_name
    elif len(parts) == 2:
        return f"{default_catalog}.{table_name}"
    elif len(parts) == 1:
        return f"{default_catalog}.{default_schema}.{table_name}"
    else:
        logger.warning(f"Invalid table name format: {table_name}")
        return None


def is_system_table(table_name: str) -> bool:
    """Check if table is a Databricks system table.

    System tables should be excluded from schema enrichment.

    Args:
        table_name: Fully qualified table name

    Returns:
        True if table is a system table

    Example:
        >>> is_system_table("system.information_schema.tables")
        True
        >>> is_system_table("main.sales.customers")
        False
    """
    return table_name.startswith("system.")
