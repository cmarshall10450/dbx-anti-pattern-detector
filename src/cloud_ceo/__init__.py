"""Cloud CEO - SQL anti-pattern detection and query optimization for Databricks.

Clean, simple API with 6 core functions:
1. analyze() - Analyze any SQL query (pre or post-execution)
2. get_analysis() - Retrieve cached analysis
3. clear_cache() - Clear analysis cache
4. AnalysisResult - Complete analysis results
5. QueryMetrics - Execution metrics model
6. ClusterInfo - Cluster configuration model

Example Usage:
    >>> from cloud_ceo import analyze
    >>>
    >>> # Pre-execution: Analyze query before running
    >>> result = analyze("SELECT * FROM table")
    >>> print(result.summary())
    >>>
    >>> # Post-execution: Analyze with performance data
    >>> result = analyze(
    ...     "SELECT * FROM table",
    ...     execution_metrics={"duration_ms": 45000, ...},
    ...     cluster_config={"node_type": "i3.2xlarge", ...},
    ...     enable_llm=True
    ... )
    >>> print(result.summary())

For Databricks-specific features, import from cloud_ceo.databricks:
    >>> from cloud_ceo.databricks import analyze_databricks_query, analyze_workspace
"""

__version__ = "0.1.0"

# Core API (6 exports)
from cloud_ceo.core import analyze, clear_cache, get_analysis
from cloud_ceo.models import AnalysisResult, ClusterInfo, QueryMetrics

__all__ = [
    # Core functions
    "analyze",
    "get_analysis",
    "clear_cache",
    # Data models
    "AnalysisResult",
    "QueryMetrics",
    "ClusterInfo",
    # Version
    "__version__",
]
