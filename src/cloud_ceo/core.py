"""Core analysis functions for Cloud CEO.

This module provides the main entry points for SQL analysis:
- analyze(): Unified function for pre/post-execution analysis
- get_analysis(): Retrieve cached analysis results
- clear_cache(): Clear analysis cache

All analysis logic is consolidated here for maintainability.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING

import structlog

# Lazy imports: Only load heavy dependencies when needed
if TYPE_CHECKING:
    from cloud_ceo.llm import LLMOrchestrator
    from cloud_ceo.llm.context import ClusterConfig, ExecutionMetrics
    from cloud_ceo.llm.workflow import analyze_violation
    from cloud_ceo.rule_engine.detector import RuleDetector
    from cloud_ceo.rule_engine.shared import DetectionContext, Violation

# Always-needed imports (lightweight)
from cloud_ceo.models import AnalysisResult, ClusterInfo, QueryMetrics

logger = structlog.get_logger(__name__)

# Global caches
_detector_cache: dict[str, Any] = {}  # RuleDetector instances
_analysis_cache: dict[str, AnalysisResult] = {}
_config_cache: Any = None


def analyze(
    query: str | Path,
    *,
    # Execution context (for post-execution analysis)
    query_id: Optional[str] = None,
    execution_metrics: Optional[dict[str, Any] | QueryMetrics] = None,
    cluster_config: Optional[dict[str, Any] | ClusterInfo] = None,

    # LLM options
    enable_llm: bool = False,

    # Databricks metadata
    databricks_query_id: Optional[str] = None,
    user: Optional[str] = None,
    warehouse_id: Optional[str] = None,

    # Configuration
    config_path: Optional[Path] = None,
) -> AnalysisResult:
    """Unified analysis function for pre-execution and post-execution scenarios.

    PERFORMANCE NOTE: Heavy dependencies (LangChain, LLM providers) are loaded
    lazily only when enable_llm=True to minimize startup time.

    This single function intelligently handles both:
    - **Pre-execution**: Analyzing queries before running (no metrics)
    - **Post-execution**: Analyzing executed queries with performance data

    Args:
        query: SQL query string or path to .sql file

        query_id: Custom query ID (auto-generated if not provided)
        execution_metrics: Query execution metrics (dict or QueryMetrics object)
            Required keys: duration_ms, bytes_scanned, rows_scanned
            Optional keys: read_rows, spilled_bytes, shuffle_read_bytes, cost_usd, timestamp
        cluster_config: Cluster configuration (dict or ClusterInfo object)
            Required keys: node_type, executor_count, executor_memory
            Optional keys: cluster_id, cluster_name

        enable_llm: Enable LLM-powered recommendations

        databricks_query_id: Original Databricks query ID
        user: User who executed the query
        warehouse_id: SQL warehouse ID

        config_path: Path to config file

    Returns:
        AnalysisResult with violations and optional LLM analysis

    Examples:
        Pre-execution analysis:
        >>> result = analyze("SELECT * FROM users WHERE status = 'active'")
        >>> print(result.summary())

        Post-execution analysis:
        >>> result = analyze(
        ...     "SELECT * FROM large_table",
        ...     execution_metrics={
        ...         "duration_ms": 45000,
        ...         "bytes_scanned": 5_000_000_000,
        ...         "rows_scanned": 1_000_000,
        ...         "cost_usd": 2.50,
        ...     },
        ...     cluster_config={
        ...         "node_type": "i3.2xlarge",
        ...         "executor_count": 4,
        ...         "executor_memory": "8GB",
        ...     },
        ...     enable_llm=True,
        ... )
        >>> print(result.summary())
    """
    # PERFORMANCE: Log immediately so users see activity
    logger.info("Starting SQL analysis", enable_llm=enable_llm)

    # Lazy import: Only load sqlglot when needed (it's fast, but principle matters)
    import sqlglot

    # Load SQL query
    if isinstance(query, (str, Path)) and Path(query).exists():
        sql_content = Path(query).read_text()
    else:
        sql_content = str(query)

    # Determine analysis type
    analysis_type: Literal["pre_execution", "post_execution"] = (
        "post_execution" if execution_metrics else "pre_execution"
    )

    # Generate or use provided query ID
    if not query_id:
        query_id = _generate_query_id(sql_content)

    logger.info(f"Analyzing query", query_id=query_id, analysis_type=analysis_type)

    # Check cache
    cache_key = f"{query_id}:{analysis_type}"
    if cache_key in _analysis_cache:
        logger.info(f"Returning cached {analysis_type} analysis for query {query_id}")
        return _analysis_cache[cache_key]

    # Load configuration (includes detector imports)
    logger.info("Loading configuration and detectors")
    config = _get_config(config_path)

    # Parse SQL
    logger.info("Parsing SQL query")
    try:
        ast = sqlglot.parse_one(sql_content, dialect="databricks")
    except Exception as e:
        logger.error(f"Failed to parse SQL: {e}")
        result = AnalysisResult(
            query_id=query_id,
            sql_query=sql_content,
            analysis_type=analysis_type,
            violations=[],
            violation_count=0,
            databricks_query_id=databricks_query_id,
            user=user,
            warehouse_id=warehouse_id,
        )
        _analysis_cache[cache_key] = result
        return result

    # Lazy import: Only load detector types when needed
    from cloud_ceo.rule_engine.shared import DetectionContext

    # Run detectors
    logger.info("Running detectors")
    detectors = _get_detectors(config)
    violations: list = []

    for detector in detectors:
        violations.extend(detector.detect(ast, DetectionContext()))

    # Convert metrics and cluster to domain objects
    metrics_obj = _convert_to_metrics(execution_metrics) if execution_metrics else None
    cluster_obj = _convert_to_cluster(cluster_config) if cluster_config else None

    # Build result
    result = AnalysisResult(
        query_id=query_id,
        sql_query=sql_content,
        analysis_type=analysis_type,
        violations=violations,
        violation_count=len(violations),
        llm_enabled=enable_llm,
        execution_metrics=metrics_obj,
        cluster_config=cluster_obj,
        databricks_query_id=databricks_query_id,
        user=user,
        warehouse_id=warehouse_id,
    )

    # LLM enhancement (lazy load LLM modules only when needed)
    if enable_llm and violations and config.llm.enabled:
        logger.info("Enhancing result with LLM (loading LLM dependencies...)")
        result = _enhance_with_llm(result, violations, cluster_obj, metrics_obj, config)
        logger.info(
            f"LLM enhancement complete - "
            f"recommendations: {len(result.llm_recommendations)}, "
            f"severity: {result.severity_assessment is not None}, "
            f"impact: {result.impact_analysis is not None}"
        )

    # Cache result
    _analysis_cache[cache_key] = result

    return result


def get_analysis(query_id: str) -> Optional[AnalysisResult]:
    """Retrieve cached analysis by query ID.

    Args:
        query_id: Query ID from previous analysis

    Returns:
        Cached AnalysisResult or None if not found
    """
    # Try both cache keys
    for analysis_type in ["pre_execution", "post_execution"]:
        cache_key = f"{query_id}:{analysis_type}"
        if cache_key in _analysis_cache:
            return _analysis_cache[cache_key]
    return None


def clear_cache() -> None:
    """Clear all cached analyses."""
    _analysis_cache.clear()
    logger.info("Analysis cache cleared")


def _generate_query_id(sql: str) -> str:
    """Generate unique ID for SQL query."""
    return hashlib.sha256(sql.encode()).hexdigest()[:12]


def _get_config(config_path: Optional[Path] = None) -> Any:
    """Load configuration with caching.

    PERFORMANCE: Config loader is imported on-demand to avoid
    loading YAML parser and Pydantic schemas at import time.
    """
    global _config_cache
    if _config_cache is None:
        from cloud_ceo.config.loader import load_config
        _config_cache = load_config(config_path)
    return _config_cache


def _get_detectors(config: Any) -> list[Any]:
    """Load detectors with caching.

    Returns list of RuleDetector instances (typed as Any to avoid import).
    """
    import importlib

    detectors = []
    for detector_path in config.detectors:
        if detector_path in _detector_cache:
            detectors.append(_detector_cache[detector_path])
            continue

        try:
            if ":" in detector_path:
                module_path, class_name = detector_path.split(":", 1)
            else:
                parts = detector_path.split(".")
                module_path, class_name = ".".join(parts[:-1]), parts[-1]

            module = importlib.import_module(module_path)
            detector = getattr(module, class_name)()
            _detector_cache[detector_path] = detector
            detectors.append(detector)
        except Exception as e:
            logger.warning(f"Could not load detector {detector_path}: {e}")

    return detectors


def _convert_to_metrics(data: dict[str, Any] | QueryMetrics) -> QueryMetrics:
    """Convert dict or QueryMetrics to QueryMetrics object."""
    if isinstance(data, QueryMetrics):
        return data

    return QueryMetrics(
        duration_ms=data["duration_ms"],
        bytes_scanned=data["bytes_scanned"],
        rows_scanned=data["rows_scanned"],
        read_rows=data.get("read_rows"),
        spilled_bytes=data.get("spilled_bytes"),
        shuffle_read_bytes=data.get("shuffle_read_bytes"),
        cost_usd=data.get("cost_usd"),
        timestamp=data.get("timestamp"),
        peak_memory_percent=data.get("peak_memory_percent"),
        avg_memory_percent=data.get("avg_memory_percent"),
        peak_cpu_percent=data.get("peak_cpu_percent"),
        avg_cpu_percent=data.get("avg_cpu_percent"),
        memory_swap_detected=data.get("memory_swap_detected"),
        avg_io_wait_percent=data.get("avg_io_wait_percent"),
    )


def _convert_to_cluster(data: dict[str, Any] | ClusterInfo) -> ClusterInfo:
    """Convert dict or ClusterInfo to ClusterInfo object.

    Supports flexible input formats:
    1. New format: Specify node types + worker count (memory auto-resolved):
       - node_type + worker_count (same node for driver and workers)
       - driver_node_type + worker_node_type + worker_count (different nodes)
    2. Legacy format: Specify explicit memory:
       - node_type + executor_count + executor_memory

    Args:
        data: Dictionary with cluster configuration or ClusterInfo object

    Returns:
        ClusterInfo object with complete configuration

    Raises:
        ValueError: If required fields are missing or invalid

    Examples:
        # New format - same node type for all
        >>> config = {
        ...     "node_type": "i3.2xlarge",
        ...     "worker_count": 4
        ... }

        # New format - different driver and worker nodes
        >>> config = {
        ...     "driver_node_type": "m5.xlarge",
        ...     "worker_node_type": "i3.2xlarge",
        ...     "worker_count": 4
        ... }

        # Legacy format - explicit memory
        >>> config = {
        ...     "node_type": "i3.2xlarge",
        ...     "executor_count": 4,
        ...     "executor_memory": "61GB"
        ... }
    """
    if isinstance(data, ClusterInfo):
        return data

    # Normalize field names for flexibility
    # Support both 'executor_count' and 'worker_count'
    executor_count = data.get("executor_count") or data.get("worker_count")

    # Support both 'node_type' and 'worker_node_type'
    worker_node_type = data.get("worker_node_type") or data.get("node_type")

    # Validate required fields
    if not worker_node_type:
        raise ValueError(
            "Missing required field: 'node_type' or 'worker_node_type' must be specified"
        )

    if not executor_count:
        raise ValueError(
            "Missing required field: 'executor_count' or 'worker_count' must be specified"
        )

    # Get executor memory - explicit or auto-resolved
    executor_memory = data.get("executor_memory")

    if not executor_memory:
        # Auto-resolve memory from node type using Databricks system tables
        logger.debug(f"Auto-resolving memory for node type: {worker_node_type}")

        try:
            from cloud_ceo.databricks.fetch import get_node_type_resolver

            resolver = get_node_type_resolver()
            executor_memory = resolver.get_memory_string(worker_node_type)

            if not executor_memory:
                raise ValueError(
                    f"Could not resolve memory for node type '{worker_node_type}'. "
                    f"Please specify 'executor_memory' explicitly (e.g., '8GB') or "
                    f"ensure PySpark is available and node type exists in system.compute.node_types."
                )

            logger.info(
                f"Auto-resolved {worker_node_type}: {executor_memory} "
                f"(from system.compute.node_types)"
            )

        except ImportError:
            raise ValueError(
                f"Cannot auto-resolve memory for '{worker_node_type}' - "
                f"PySpark integration not available. Please specify 'executor_memory' "
                f"explicitly (e.g., {{'executor_memory': '8GB'}})."
            )

    # Get driver memory - explicit or auto-resolved
    driver_node_type = data.get("driver_node_type")
    driver_memory = data.get("driver_memory")

    if driver_node_type and not driver_memory:
        # Auto-resolve driver memory if driver node type is specified
        try:
            from cloud_ceo.databricks.fetch import get_node_type_resolver

            resolver = get_node_type_resolver()
            driver_memory = resolver.get_memory_string(driver_node_type)

            if driver_memory:
                logger.info(
                    f"Auto-resolved driver {driver_node_type}: {driver_memory}"
                )

        except (ImportError, Exception) as e:
            logger.debug(f"Could not auto-resolve driver memory: {e}")

    return ClusterInfo(
        node_type=worker_node_type,
        executor_count=executor_count,
        executor_memory=executor_memory,
        cluster_id=data.get("cluster_id"),
        cluster_name=data.get("cluster_name"),
        driver_node_type=driver_node_type,
        driver_memory=driver_memory,
        core_count=data.get("core_count"),
        gpu_count=data.get("gpu_count"),
        dbr_version=data.get("dbr_version"),
        owned_by=data.get("owned_by"),
        cluster_source=data.get("cluster_source"),
        data_security_mode=data.get("data_security_mode"),
        spot_instance=data.get("spot_instance"),
    )


def _enhance_with_llm(
    result: AnalysisResult,
    violations: list,
    cluster_obj: Optional[ClusterInfo],
    metrics_obj: Optional[QueryMetrics],
    config: Any,
) -> AnalysisResult:
    """Enhance result with LLM analysis.

    PERFORMANCE: Lazy-loads LLM modules (LangChain, OpenAI, etc.) only when called.
    This avoids loading heavy dependencies during import.
    """
    try:
        # Lazy import: Only load LLM modules when LLM is actually enabled
        from cloud_ceo.llm import LLMOrchestrator
        from cloud_ceo.llm.context import ClusterConfig, ExecutionMetrics, Violation as LLMViolation
        from cloud_ceo.llm.workflow import analyze_violation

        orchestrator = LLMOrchestrator(
            model_name=config.llm.model,
            provider=config.llm.provider or "openai",
        )

        if not orchestrator.is_available():
            logger.warning("LLM not available - skipping enhancement")
            return result

        # Convert to LLM domain objects
        llm_cluster = None
        if cluster_obj:
            llm_cluster = ClusterConfig(
                node_type=cluster_obj.node_type,
                executor_count=cluster_obj.executor_count,
                executor_memory=cluster_obj.executor_memory,
            )

        llm_metrics = None
        if metrics_obj:
            llm_metrics = ExecutionMetrics(
                duration_ms=metrics_obj.duration_ms,
                bytes_scanned=metrics_obj.bytes_scanned,
                rows_scanned=metrics_obj.rows_scanned,
                peak_memory_mb=0,  # Not available in system tables
                shuffle_bytes=metrics_obj.shuffle_read_bytes,
            )

        # Convert violations
        llm_violations = [
            LLMViolation(
                pattern=v.rule_id,
                fragment=str(v.location.get("fragment", "")),
                line_number=v.location.get("line", 0),
                default_message=v.message,
                severity=v.severity.value if hasattr(v.severity, 'value') else str(v.severity),
                fix_suggestion=v.fix_suggestion or "",
            )
            for v in violations
        ]

        # Get Databricks session if available (for schema enrichment)
        databricks_session = None
        try:
            from cloud_ceo.databricks.fetch import DatabricksSystemTables
            # Try to create a session - will fail gracefully if PySpark not available
            databricks_session = DatabricksSystemTables()
        except Exception:
            # PySpark not available or initialization failed - schema enrichment will be skipped
            pass

        # Get industry mode from config (if specified)
        industry_mode = getattr(config, 'industry', None)

        # Analyze with workflow
        if llm_violations:
            workflow_result = analyze_violation(
                violation=llm_violations[0],
                all_violations=llm_violations,
                execution_metrics=llm_metrics,
                cluster_config=llm_cluster,
                orchestrator=orchestrator,
                sql_query=result.sql_query,
                databricks_session=databricks_session,
                industry_mode=industry_mode,
            )

            logger.info(
                f"=== WORKFLOW RESULT RECEIVED ==="
            )
            logger.info(f"  - recommendations: {len(workflow_result.recommendations) if workflow_result.recommendations else 0}")
            logger.info(f"  - severity_assessment: {workflow_result.severity_assessment is not None}")
            logger.info(f"  - impact_analysis: {workflow_result.impact_analysis is not None}")
            logger.info(f"  - violation_explanation: {len(workflow_result.violation_explanation) if workflow_result.violation_explanation else 0} chars")
            logger.info(f"  - error: {workflow_result.error}")

            # Extract results
            if workflow_result.recommendations:
                logger.info(f"=== EXTRACTING {len(workflow_result.recommendations)} RECOMMENDATIONS ===")
                result.llm_recommendations = [
                    {
                        "type": rec.type.value if hasattr(rec.type, "value") else str(rec.type),
                        "description": rec.description,
                        "implementation": rec.implementation,
                        "expected_improvement": rec.expected_improvement,
                        "effort": rec.effort_level.value if hasattr(rec.effort_level, "value") else str(rec.effort_level),
                    }
                    for rec in workflow_result.recommendations
                ]
                logger.info(f"=== SET result.llm_recommendations = {len(result.llm_recommendations)} items ===")
            else:
                logger.warning("=== NO RECOMMENDATIONS IN WORKFLOW RESULT ===")

            if workflow_result.severity_assessment:
                # Extract both overall and per-violation assessments
                sev = workflow_result.severity_assessment
                result.severity_assessment = {
                    # Overall fields (backward compatible)
                    "severity": sev.overall_severity.value,
                    "confidence": sev.overall_confidence,
                    "evidence": sev.overall_evidence,
                    # New per-violation fields
                    "overall_severity": sev.overall_severity.value,
                    "overall_confidence": sev.overall_confidence,
                    "overall_evidence": sev.overall_evidence,
                    "violation_assessments": [
                        {
                            "violation_pattern": va.violation_pattern,
                            "severity": va.severity.value,
                            "confidence": va.confidence,
                            "evidence": va.evidence,
                        }
                        for va in sev.violation_assessments
                    ],
                }

            if workflow_result.impact_analysis:
                # Extract both overall and per-violation impacts
                impact = workflow_result.impact_analysis
                result.impact_analysis = {
                    # Overall fields
                    "root_cause": impact.root_cause,
                    "performance_impact": impact.performance_impact,
                    "cost_impact": impact.cost_impact_usd or 0.0,
                    "cost_impact_usd": impact.cost_impact_usd,
                    "affected_resources": impact.affected_resources,
                    # New per-violation fields
                    "violation_impacts": [
                        {
                            "violation_pattern": vi.violation_pattern,
                            "performance_impact": vi.performance_impact,
                            "cost_contribution": vi.cost_contribution,
                            "affected_resources": vi.affected_resources,
                        }
                        for vi in impact.violation_impacts
                    ],
                }

            if workflow_result.violation_explanation:
                result.violation_explanation = workflow_result.violation_explanation
                logger.info(f"=== EXTRACTED violation_explanation: {len(result.violation_explanation)} chars ===")

    except Exception as e:
        logger.warning(f"LLM enhancement failed: {e}")

    return result
