"""Spark session management utilities for Databricks integration.

This module provides shared utilities for managing Spark sessions across
the cloud-ceo-dbx codebase. It prioritizes reusing active Databricks sessions
while gracefully falling back to creating new sessions for local development.
"""

from typing import TYPE_CHECKING, Optional

import structlog

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = structlog.get_logger(__name__)


def get_spark_session(app_name: str = "CloudCEO") -> Optional["SparkSession"]:
    """Get active Spark session or create one.

    Prioritizes active session (for Databricks notebooks),
    falls back to creating new session (for local testing).

    This function is designed to work seamlessly in both:
    - Databricks notebooks (reuses pre-configured session)
    - Local development (creates new session)
    - CI/CD tests (creates test session)

    Args:
        app_name: Application name for new sessions (default: "CloudCEO")

    Returns:
        SparkSession instance, or None if PySpark is not available

    Example:
        >>> from cloud_ceo.databricks.spark_utils import get_spark_session
        >>> spark = get_spark_session()
        >>> if spark:
        ...     df = spark.sql("SELECT * FROM my_table")

    Note:
        In Databricks notebooks, this will reuse the existing 'spark' session
        rather than creating a new one, which is more efficient and leverages
        the pre-configured Databricks environment.

    Best Practices:
        - Always check if the returned session is None before using it
        - Use dependency injection to pass sessions to classes when possible
        - Avoid creating multiple sessions - reuse the returned instance
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.debug("PySpark not available - cannot create Spark session")
        return None

    # Try to get active session first (Databricks notebooks)
    spark = SparkSession.getActiveSession()

    if spark is not None:
        logger.debug("Using active Spark session from Databricks notebook")
        return spark

    # Fallback: create session (local testing, standalone scripts)
    logger.debug(f"No active session found - creating new session: {app_name}")
    spark = SparkSession.builder.appName(app_name).getOrCreate()

    return spark
