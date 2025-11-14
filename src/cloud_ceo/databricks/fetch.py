"""Automatic fetching of query data from Databricks system tables.

This module handles all interaction with Databricks system tables to automatically
retrieve query text, execution metrics, and cluster configuration.

Users only need to provide a Databricks query ID - everything else is fetched automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from cloud_ceo.models import ClusterInfo, QueryMetrics

logger = structlog.get_logger(__name__)


class NodeTypeResolver:
    """Resolve node types to hardware specifications from Databricks.

    This class queries system.compute.node_types to fetch hardware specs
    (memory, cores, GPUs) for given node type IDs. Results are cached for
    the duration of the session to avoid repeated queries.

    Example:
        >>> resolver = NodeTypeResolver()
        >>> specs = resolver.get_node_specs("i3.2xlarge")
        >>> print(specs["memory_gb"])
        61
    """

    def __init__(self, spark_session: Optional[Any] = None):
        """Initialize resolver with optional Spark session.

        Args:
            spark_session: PySpark session (auto-created if not provided)
        """
        from cloud_ceo.databricks.spark_utils import get_spark_session

        self._spark = spark_session or get_spark_session()
        self._cache: dict[str, dict[str, Any]] = {}

    def get_node_specs(self, node_type_id: str) -> Optional[dict[str, Any]]:
        """Fetch hardware specifications for a node type.

        Args:
            node_type_id: Node type identifier (e.g., "i3.2xlarge", "m5.xlarge")

        Returns:
            Dict with keys: memory_gb, memory_mb, num_cores, num_gpus, category
            Returns None if node type not found or Spark unavailable

        Example:
            >>> resolver = NodeTypeResolver()
            >>> specs = resolver.get_node_specs("i3.2xlarge")
            >>> print(f"Memory: {specs['memory_gb']}GB, Cores: {specs['num_cores']}")
            Memory: 61GB, Cores: 8
        """
        if not self._spark:
            logger.debug(f"Cannot resolve {node_type_id} - PySpark unavailable")
            return None

        # Check cache
        if node_type_id in self._cache:
            logger.debug(f"Cache hit for node type: {node_type_id}")
            return self._cache[node_type_id]

        # Query system.compute.node_types (using parameterized query for security)
        try:
            sql = """
            SELECT
                node_type_id,
                memory_mb,
                num_cores,
                num_gpus,
                category
            FROM system.compute.node_types
            WHERE node_type_id = :node_type_id
            LIMIT 1
            """

            df = self._spark.sql(sql, args={"node_type_id": node_type_id})
            rows = df.collect()

            if not rows:
                logger.warning(f"Node type '{node_type_id}' not found in system.compute.node_types")
                return None

            row = rows[0]
            specs = {
                "node_type_id": row.node_type_id,
                "memory_mb": row.memory_mb or 0,
                "memory_gb": (row.memory_mb or 0) // 1024,
                "num_cores": row.num_cores or 0,
                "num_gpus": row.num_gpus or 0,
                "category": row.category or "Unknown",
            }

            # Cache result
            self._cache[node_type_id] = specs
            logger.debug(f"Resolved {node_type_id}: {specs['memory_gb']}GB, {specs['num_cores']} cores")

            return specs

        except Exception as e:
            logger.warning(f"Failed to resolve node type '{node_type_id}': {e}")
            return None

    def get_memory_gb(self, node_type_id: str) -> Optional[int]:
        """Get memory in GB for a node type.

        Convenience method that returns only the memory value.

        Args:
            node_type_id: Node type identifier

        Returns:
            Memory in GB, or None if unavailable
        """
        specs = self.get_node_specs(node_type_id)
        return specs["memory_gb"] if specs else None

    def get_memory_string(self, node_type_id: str) -> Optional[str]:
        """Get memory as formatted string (e.g., "61GB").

        Args:
            node_type_id: Node type identifier

        Returns:
            Memory string like "8GB" or None if unavailable
        """
        memory_gb = self.get_memory_gb(node_type_id)
        return f"{memory_gb}GB" if memory_gb else None


# Global resolver instance (lazily initialized)
_global_resolver: Optional[NodeTypeResolver] = None


def get_node_type_resolver() -> NodeTypeResolver:
    """Get or create global NodeTypeResolver instance.

    This ensures we have a single shared resolver with unified caching
    across all calls within a session.

    Returns:
        Shared NodeTypeResolver instance
    """
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = NodeTypeResolver()
    return _global_resolver


@dataclass
class DatabricksQueryData:
    """Complete query data fetched from Databricks system tables.

    Attributes:
        statement_id: Databricks statement/query ID
        query_text: SQL query text
        execution_metrics: Performance metrics
        cluster_config: Cluster configuration
        user: User who executed the query
        warehouse_id: SQL warehouse ID (if applicable)
        cluster_id: Cluster ID (if applicable)
        execution_status: Query status (FINISHED, FAILED, CANCELED)
        error_message: Error message if query failed
    """

    statement_id: str
    query_text: str
    execution_metrics: QueryMetrics
    cluster_config: Optional[ClusterInfo]
    user: str
    warehouse_id: Optional[str]
    cluster_id: Optional[str]
    execution_status: str
    error_message: Optional[str]


class DatabricksSystemTables:
    """Fetch query data from Databricks system tables.

    This class provides high-level methods to retrieve complete query information
    from system.query.history, system.compute.clusters, and system.compute.node_types.
    """

    def __init__(self, spark_session: Optional[Any] = None):
        """Initialize system tables accessor.

        Args:
            spark_session: PySpark session (auto-created if not provided)

        Raises:
            RuntimeError: If PySpark is not available and session not provided
        """
        from cloud_ceo.databricks.spark_utils import get_spark_session

        self._spark = spark_session or get_spark_session()

        if self._spark is None:
            logger.error(
                "PySpark not available - install with: pip install cloud-ceo-dbx[databricks]"
            )
            raise RuntimeError(
                "PySpark required for Databricks system table access. "
                "Install with: pip install cloud-ceo-dbx[databricks]"
            )

    def fetch_query_by_id(self, statement_id: str) -> Optional[DatabricksQueryData]:
        """Fetch complete query data by statement ID.

        Automatically retrieves:
        - Query text
        - Execution metrics (duration, bytes scanned, memory, etc.)
        - Cluster configuration (node type, worker count, memory, runtime version)
        - Runtime utilization metrics (memory %, CPU %, I/O wait - if available)
        - User and warehouse information

        Args:
            statement_id: Databricks statement ID

        Returns:
            DatabricksQueryData with all information, or None if not found

        Example:
            >>> tables = DatabricksSystemTables()
            >>> data = tables.fetch_query_by_id("01HF6K2T3E8M9N7P6Q5R4S")
            >>> print(data.query_text)
            >>> print(f"Duration: {data.execution_metrics.duration_ms}ms")
            >>> if data.execution_metrics.peak_memory_percent:
            ...     print(f"Peak Memory: {data.execution_metrics.peak_memory_percent:.1f}%")
        """
        if not self._spark:
            raise RuntimeError("PySpark session not initialized")

        logger.info(f"Fetching query data for statement_id: {statement_id}")

        try:
            # Fetch query history record
            query_data = self._fetch_query_history(statement_id)
            if not query_data:
                logger.warning(f"Query {statement_id} not found in system.query.history")
                return None

            # Fetch cluster configuration if cluster_id is available
            cluster_config = None
            if query_data.get("cluster_id"):
                cluster_config = self._fetch_cluster_config(
                    query_data["cluster_id"],
                    query_data.get("start_time"),
                )

                # Fetch runtime utilization metrics from node_timeline
                if query_data.get("start_time") and query_data.get("total_duration_ms"):
                    runtime_metrics = self._fetch_runtime_utilization(
                        query_data["cluster_id"],
                        query_data["start_time"],
                        query_data["total_duration_ms"],
                    )
                    if runtime_metrics:
                        # Merge runtime metrics into query_data
                        query_data.update(runtime_metrics)
                        logger.debug(
                            f"Fetched runtime utilization: "
                            f"peak_mem={runtime_metrics.get('peak_memory_percent'):.1f}%, "
                            f"avg_cpu={runtime_metrics.get('avg_cpu_percent'):.1f}%"
                        )

            # Build complete query data object
            return self._build_query_data(query_data, cluster_config)

        except Exception as e:
            logger.error(f"Failed to fetch query {statement_id}: {e}", exc_info=True)
            return None

    def _fetch_query_history(self, statement_id: str) -> Optional[dict]:
        """Fetch query history record from system.query.history.

        Returns dict with query text, metrics, user info, and cluster/warehouse IDs.
        """
        sql = """
        SELECT
            -- Identity
            statement_id,
            statement_text,
            statement_type,
            execution_status,
            error_message,

            -- User & Compute
            executed_by as user,
            compute.warehouse_id,
            compute.compute_id as cluster_id,
            compute.compute_type,

            -- Timing
            start_time,
            end_time,
            total_duration_ms,
            execution_duration_ms,
            compilation_duration_ms,
            total_task_duration_ms,

            -- Data Access (using documented column names)
            read_bytes,
            read_files,
            read_partitions,
            read_rows,
            produced_rows,

            -- Resources (using documented column names)
            spilled_local_bytes,
            shuffle_read_bytes,

            -- Cost estimation (rough)
            total_task_duration_ms * 0.00001 as estimated_cost_usd

        FROM system.query.history
        WHERE statement_id = :statement_id
        LIMIT 1
        """

        try:
            df = self._spark.sql(sql, args={"statement_id": statement_id})
            rows = df.collect()

            if not rows:
                return None

            row = rows[0]

            # Convert to dict (using documented column names)
            return {
                "statement_id": row.statement_id,
                "statement_text": row.statement_text,
                "statement_type": row.statement_type or "UNKNOWN",
                "execution_status": row.execution_status,
                "error_message": row.error_message,
                "user": row.user,
                "warehouse_id": row.warehouse_id,
                "cluster_id": row.cluster_id,
                "compute_type": row.compute_type,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "total_duration_ms": row.total_duration_ms or 0,
                "execution_duration_ms": row.execution_duration_ms or 0,
                "compilation_duration_ms": row.compilation_duration_ms or 0,
                "total_task_duration_ms": row.total_task_duration_ms or 0,
                "read_bytes": row.read_bytes or 0,
                "read_files": row.read_files or 0,
                "read_partitions": row.read_partitions or 0,
                "read_rows": row.read_rows or 0,
                "produced_rows": row.produced_rows or 0,
                "spilled_local_bytes": row.spilled_local_bytes or 0,
                "shuffle_read_bytes": row.shuffle_read_bytes or 0,
                "estimated_cost_usd": row.estimated_cost_usd,
            }

        except Exception as e:
            logger.error(f"Error fetching query history: {e}")
            return None

    def _fetch_cluster_config(
        self, cluster_id: str, query_start_time: Optional[datetime] = None
    ) -> Optional[ClusterInfo]:
        """Fetch cluster configuration from system.compute.clusters.

        Enhanced to include driver specs, runtime version, ownership, and hardware details.
        Uses temporal join to get the cluster config that was active when the query ran.

        Args:
            cluster_id: Cluster ID
            query_start_time: When the query started (for temporal matching)

        Returns:
            ClusterInfo with comprehensive cluster configuration
        """
        # Build parameterized query with temporal filter if we have a start time
        params = {"cluster_id": cluster_id}

        if query_start_time:
            query_ts = query_start_time.strftime("%Y-%m-%d %H:%M:%S")
            params["query_start_time"] = query_ts
            time_filter = """
            AND change_time <= :query_start_time
            AND (delete_time IS NULL OR delete_time > :query_start_time)
            """
        else:
            time_filter = ""

        sql = f"""
        WITH cluster_config AS (
            SELECT
                cluster_id,
                cluster_name,
                worker_node_type,
                driver_node_type,
                worker_count,
                min_autoscale_workers,
                max_autoscale_workers,
                dbr_version,
                owned_by,
                cluster_source,
                data_security_mode,
                aws_attributes.availability as spot_availability,
                change_time
            FROM system.compute.clusters
            WHERE cluster_id = :cluster_id
            {time_filter}
            ORDER BY change_time DESC
            LIMIT 1
        ),
        node_specs AS (
            SELECT
                node_type,
                memory_mb,
                core_count,
                gpu_count
            FROM system.compute.node_types
        )
        SELECT
            c.cluster_id,
            c.cluster_name,
            c.worker_node_type as node_type,
            c.driver_node_type,
            c.worker_count,
            c.min_autoscale_workers,
            c.max_autoscale_workers,
            c.dbr_version,
            c.owned_by,
            c.cluster_source,
            c.data_security_mode,
            c.spot_availability,
            wn.memory_mb as worker_memory_mb,
            wn.core_count as worker_core_count,
            wn.gpu_count as worker_gpu_count,
            dn.memory_mb as driver_memory_mb
        FROM cluster_config c
        LEFT JOIN node_specs wn ON c.worker_node_type = wn.node_type
        LEFT JOIN node_specs dn ON c.driver_node_type = dn.node_type
        """

        try:
            df = self._spark.sql(sql, args=params)
            rows = df.collect()

            if not rows:
                logger.warning(f"Cluster {cluster_id} not found in system.compute.clusters")
                return None

            row = rows[0]

            # Determine worker count (fixed or autoscale)
            worker_count = row.worker_count
            if worker_count is None and row.min_autoscale_workers:
                # Use max for autoscale clusters (conservative estimate)
                worker_count = row.max_autoscale_workers

            # Convert worker memory to GB string
            worker_memory_mb = row.worker_memory_mb or 8192  # Default to 8GB
            worker_memory_gb = f"{worker_memory_mb // 1024}GB"

            # Convert driver memory to GB string
            driver_memory_mb = row.driver_memory_mb or worker_memory_mb
            driver_memory_gb = f"{driver_memory_mb // 1024}GB"

            # Detect spot instances from AWS attributes
            spot_instance = None
            if row.spot_availability:
                spot_instance = "SPOT" in row.spot_availability.upper()

            return ClusterInfo(
                node_type=row.node_type or "unknown",
                executor_count=worker_count or 1,
                executor_memory=worker_memory_gb,
                cluster_id=cluster_id,
                cluster_name=row.cluster_name,
                driver_node_type=row.driver_node_type,
                driver_memory=driver_memory_gb,
                core_count=row.worker_core_count,
                gpu_count=row.worker_gpu_count,
                dbr_version=row.dbr_version,
                owned_by=row.owned_by,
                cluster_source=row.cluster_source,
                data_security_mode=row.data_security_mode,
                spot_instance=spot_instance,
            )

        except Exception as e:
            logger.error(f"Error fetching cluster config: {e}")
            return None

    def _fetch_runtime_utilization(
        self,
        cluster_id: str,
        query_start_time: datetime,
        query_duration_ms: int,
    ) -> Optional[dict]:
        """Fetch runtime utilization metrics from system.compute.node_timeline.

        Returns aggregate metrics across all nodes during query execution window.

        Args:
            cluster_id: Cluster ID
            query_start_time: When query started
            query_duration_ms: Query duration in milliseconds

        Returns:
            Dict with peak/avg memory, CPU, and I/O metrics, or None if unavailable

        Note:
            Nodes that ran for less than 10 minutes may not appear in node_timeline.
        """
        from datetime import timedelta

        # Calculate query execution window
        query_end_time = query_start_time + timedelta(milliseconds=query_duration_ms)

        # Format timestamps
        start_ts = query_start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_ts = query_end_time.strftime("%Y-%m-%d %H:%M:%S")

        sql = """
        SELECT
            -- Memory metrics (critical for memory pressure detection)
            MAX(mem_used_percent) as peak_memory_percent,
            AVG(mem_used_percent) as avg_memory_percent,
            MAX(mem_swap_percent) as peak_swap_percent,

            -- CPU metrics
            MAX(cpu_user_percent + cpu_system_percent) as peak_cpu_percent,
            AVG(cpu_user_percent + cpu_system_percent) as avg_cpu_percent,

            -- I/O metrics
            AVG(cpu_wait_percent) as avg_io_wait_percent,
            SUM(network_sent_bytes) as total_network_sent,
            SUM(network_received_bytes) as total_network_received,

            -- Metadata
            COUNT(DISTINCT instance_id) as node_count,
            COUNT(*) as sample_count

        FROM system.compute.node_timeline
        WHERE cluster_id = :cluster_id
          AND start_time >= :start_time
          AND end_time <= :end_time
        """

        try:
            df = self._spark.sql(
                sql,
                args={
                    "cluster_id": cluster_id,
                    "start_time": start_ts,
                    "end_time": end_ts,
                },
            )
            rows = df.collect()

            if not rows or rows[0].sample_count == 0:
                logger.debug(
                    f"No node timeline data for cluster {cluster_id} "
                    f"(query may have run < 10 minutes)"
                )
                return None

            row = rows[0]

            return {
                "peak_memory_percent": row.peak_memory_percent,
                "avg_memory_percent": row.avg_memory_percent,
                "peak_cpu_percent": row.peak_cpu_percent,
                "avg_cpu_percent": row.avg_cpu_percent,
                "memory_swap_detected": (row.peak_swap_percent or 0) > 0,
                "avg_io_wait_percent": row.avg_io_wait_percent,
                "total_network_sent": row.total_network_sent,
                "total_network_received": row.total_network_received,
                "node_count": row.node_count,
                "sample_count": row.sample_count,
            }

        except Exception as e:
            logger.debug(f"Could not fetch runtime utilization: {e}")
            return None

    def _build_query_data(
        self, query_data: dict, cluster_config: Optional[ClusterInfo]
    ) -> DatabricksQueryData:
        """Build DatabricksQueryData from raw query and cluster data.

        Args:
            query_data: Dict from _fetch_query_history (may include runtime metrics)
            cluster_config: ClusterInfo from _fetch_cluster_config

        Returns:
            Complete DatabricksQueryData object
        """
        # Build execution metrics (using documented columns + runtime utilization)
        metrics = QueryMetrics(
            duration_ms=query_data["execution_duration_ms"],
            bytes_scanned=query_data["read_bytes"],
            rows_scanned=query_data["produced_rows"],
            read_rows=query_data.get("read_rows"),
            spilled_bytes=query_data.get("spilled_local_bytes"),
            shuffle_read_bytes=query_data.get("shuffle_read_bytes"),
            cost_usd=query_data.get("estimated_cost_usd"),
            timestamp=query_data.get("start_time"),
            # Runtime utilization from node_timeline (if available)
            peak_memory_percent=query_data.get("peak_memory_percent"),
            avg_memory_percent=query_data.get("avg_memory_percent"),
            peak_cpu_percent=query_data.get("peak_cpu_percent"),
            avg_cpu_percent=query_data.get("avg_cpu_percent"),
            memory_swap_detected=query_data.get("memory_swap_detected"),
            avg_io_wait_percent=query_data.get("avg_io_wait_percent"),
        )

        return DatabricksQueryData(
            statement_id=query_data["statement_id"],
            query_text=query_data["statement_text"],
            execution_metrics=metrics,
            cluster_config=cluster_config,
            user=query_data["user"],
            warehouse_id=query_data.get("warehouse_id"),
            cluster_id=query_data.get("cluster_id"),
            execution_status=query_data["execution_status"],
            error_message=query_data.get("error_message"),
        )

    def fetch_recent_queries(
        self,
        hours: int = 24,
        min_duration_ms: Optional[int] = None,
        user: Optional[str] = None,
        limit: int = 100,
    ) -> list[DatabricksQueryData]:
        """Fetch recent queries from system tables.

        Args:
            hours: Number of hours to look back
            min_duration_ms: Minimum query duration filter
            user: Filter by specific user
            limit: Maximum number of queries to return

        Returns:
            List of DatabricksQueryData objects
        """
        # Build parameterized query with dynamic filters
        params = {"hours": hours, "limit": limit}
        filters = ["start_time >= current_timestamp() - INTERVAL :hours HOURS"]

        if min_duration_ms:
            filters.append("execution_duration_ms >= :min_duration_ms")
            params["min_duration_ms"] = min_duration_ms

        if user:
            filters.append("executed_by = :user")
            params["user"] = user

        where_clause = " AND ".join(filters)

        sql = f"""
        SELECT statement_id
        FROM system.query.history
        WHERE {where_clause}
        AND execution_status = 'FINISHED'
        ORDER BY execution_duration_ms DESC
        LIMIT :limit
        """

        try:
            df = self._spark.sql(sql, args=params)
            statement_ids = [row.statement_id for row in df.collect()]

            logger.info(f"Fetching {len(statement_ids)} recent queries")

            # Fetch complete data for each query
            queries = []
            for statement_id in statement_ids:
                query_data = self.fetch_query_by_id(statement_id)
                if query_data:
                    queries.append(query_data)

            return queries

        except Exception as e:
            logger.error(f"Error fetching recent queries: {e}")
            return []


    def fetch_table_schema(self, table_name: str) -> Optional[dict[str, Any]]:
        """Fetch complete table schema from information_schema.

        Queries Databricks system.information_schema to retrieve:
        - Column metadata (names, types, comments, nullable flags)
        - Table-level description
        - PII indicators (detected from column names and comments)
        - Security/compliance tags (from table comments)
        - Unity Catalog tags (table-level and column-level)

        Args:
            table_name: Fully qualified table name (e.g., "catalog.schema.table")

        Returns:
            Dict with:
                - table_name: str
                - table_comment: str | None
                - table_tags: dict[str, str] - Unity Catalog table tags
                - columns: List[dict] with {name, type, comment, nullable, pii_detected, column_tags}
                - pii_indicators: List[str] of detected PII patterns
                - security_tags: List[str] of security/compliance tags
            Returns None if table not found or error occurs

        Example:
            >>> tables = DatabricksSystemTables()
            >>> schema = tables.fetch_table_schema("main.default.customers")
            >>> print(f"Columns: {len(schema['columns'])}")
            >>> print(f"PII detected: {schema['pii_indicators']}")
            >>> print(f"Table tags: {schema['table_tags']}")
        """
        if not self._spark:
            logger.debug(f"Cannot fetch schema for {table_name} - PySpark unavailable")
            return None

        # Parse table name into catalog.schema.table
        try:
            parts = table_name.split(".")
            if len(parts) != 3:
                logger.warning(
                    f"Table name '{table_name}' must be fully qualified as "
                    f"catalog.schema.table"
                )
                return None

            catalog, schema, table = parts

            # Query column metadata from information_schema
            column_sql = """
            SELECT
                column_name,
                data_type,
                comment,
                is_nullable
            FROM system.information_schema.columns
            WHERE table_catalog = :catalog
              AND table_schema = :schema
              AND table_name = :table
            ORDER BY ordinal_position
            """

            # Query table metadata
            table_sql = """
            SELECT
                table_comment
            FROM system.information_schema.tables
            WHERE table_catalog = :catalog
              AND table_schema = :schema
              AND table_name = :table
            LIMIT 1
            """

            # Execute queries with parameterized inputs (security best practice)
            params = {"catalog": catalog, "schema": schema, "table": table}

            col_df = self._spark.sql(column_sql, args=params)
            col_rows = col_df.collect()

            if not col_rows:
                logger.warning(f"Table '{table_name}' not found in information_schema")
                return None

            table_df = self._spark.sql(table_sql, args=params)
            table_rows = table_df.collect()
            table_comment = table_rows[0].table_comment if table_rows else None

            # Fetch Unity Catalog tags (gracefully handle if not available)
            table_tags = self._fetch_table_tags(catalog, schema, table)
            column_tags = self._fetch_column_tags(catalog, schema, table)

            # Build column list with PII detection
            columns = []
            pii_indicators = set()

            # Common PII patterns (UK-focused + global pharma)
            PII_PATTERNS = {
                "email": ["email", "e_mail", "mail_address"],
                "phone": ["phone", "mobile", "telephone", "tel_number"],
                "ssn": ["ssn", "national_insurance", "ni_number", "tax_id"],
                "patient": ["patient_id", "patient_name", "medical_record", "mrn", "nhs_number"],
                "name": ["first_name", "last_name", "full_name", "surname", "given_name"],
                "address": ["address", "street", "postcode", "postal_code", "zip_code"],
                "dob": ["date_of_birth", "dob", "birth_date", "birthdate"],
                "financial": ["account_number", "sort_code", "iban", "card_number"],
                "clinical": ["diagnosis", "medication", "treatment", "clinical_trial"],
                "ip_address": ["ip_address", "ip_addr", "client_ip"],
            }

            for row in col_rows:
                col_name_lower = row.column_name.lower()
                comment_lower = (row.comment or "").lower()

                # Detect PII patterns
                detected_pii = []
                for pii_type, patterns in PII_PATTERNS.items():
                    if any(pattern in col_name_lower for pattern in patterns):
                        detected_pii.append(pii_type)
                        pii_indicators.add(pii_type)
                    # Also check comments
                    if row.comment and any(pattern in comment_lower for pattern in patterns):
                        if pii_type not in detected_pii:
                            detected_pii.append(pii_type)
                            pii_indicators.add(pii_type)

                # Get Unity Catalog tags for this column
                col_tags = column_tags.get(row.column_name, {})

                columns.append({
                    "name": row.column_name,
                    "type": row.data_type,
                    "comment": row.comment,
                    "nullable": row.is_nullable == "YES",
                    "pii_detected": detected_pii if detected_pii else None,
                    "column_tags": col_tags if col_tags else {},
                })

            # Extract security tags from table comment
            security_tags = []
            if table_comment:
                comment_lower = table_comment.lower()
                if "gdpr" in comment_lower or "personal data" in comment_lower:
                    security_tags.append("GDPR")
                if "pci" in comment_lower or "payment" in comment_lower:
                    security_tags.append("PCI-DSS")
                if "hipaa" in comment_lower or "phi" in comment_lower:
                    security_tags.append("HIPAA")
                if "gxp" in comment_lower or "21 cfr" in comment_lower:
                    security_tags.append("GxP")

            logger.debug(
                f"Fetched schema for {table_name}: {len(columns)} columns, "
                f"PII indicators: {list(pii_indicators)}, "
                f"table tags: {len(table_tags)}, column tags: {len(column_tags)}"
            )

            return {
                "table_name": table_name,
                "table_comment": table_comment,
                "table_tags": table_tags,
                "columns": columns,
                "pii_indicators": sorted(list(pii_indicators)),
                "security_tags": security_tags,
            }

        except Exception as e:
            logger.warning(f"Failed to fetch schema for '{table_name}': {e}")
            return None

    def _fetch_table_tags(
        self, catalog: str, schema: str, table: str
    ) -> dict[str, str]:
        """Fetch Unity Catalog table-level tags from information_schema.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name

        Returns:
            Dict mapping tag_name to tag_value (empty dict if tags unavailable)
        """
        if not self._spark:
            return {}

        try:
            sql = """
            SELECT
                tag_name,
                tag_value
            FROM system.information_schema.table_tags
            WHERE catalog_name = :catalog
              AND schema_name = :schema
              AND table_name = :table
            """

            params = {"catalog": catalog, "schema": schema, "table": table}
            df = self._spark.sql(sql, args=params)
            rows = df.collect()

            # Build tag dictionary
            tags = {row.tag_name: row.tag_value for row in rows}

            if tags:
                logger.debug(
                    f"Fetched {len(tags)} table tags for {catalog}.{schema}.{table}"
                )

            return tags

        except Exception as e:
            # Tags might not be available (pre-Unity Catalog or DBR < 13.3)
            logger.debug(
                f"Unable to fetch table tags for {catalog}.{schema}.{table}: {e}. "
                f"This is expected if Unity Catalog tags are not enabled (requires DBR 13.3+)."
            )
            return {}

    def _fetch_column_tags(
        self, catalog: str, schema: str, table: str
    ) -> dict[str, dict[str, str]]:
        """Fetch Unity Catalog column-level tags from information_schema.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name

        Returns:
            Dict mapping column_name to dict of {tag_name: tag_value}
            Returns empty dict if tags unavailable
        """
        if not self._spark:
            return {}

        try:
            sql = """
            SELECT
                column_name,
                tag_name,
                tag_value
            FROM system.information_schema.column_tags
            WHERE catalog_name = :catalog
              AND schema_name = :schema
              AND table_name = :table
            """

            params = {"catalog": catalog, "schema": schema, "table": table}
            df = self._spark.sql(sql, args=params)
            rows = df.collect()

            # Build nested dictionary: {column_name: {tag_name: tag_value}}
            column_tags: dict[str, dict[str, str]] = {}
            for row in rows:
                if row.column_name not in column_tags:
                    column_tags[row.column_name] = {}
                column_tags[row.column_name][row.tag_name] = row.tag_value

            if column_tags:
                total_tags = sum(len(tags) for tags in column_tags.values())
                logger.debug(
                    f"Fetched {total_tags} column tags across "
                    f"{len(column_tags)} columns for {catalog}.{schema}.{table}"
                )

            return column_tags

        except Exception as e:
            # Tags might not be available (pre-Unity Catalog or DBR < 13.3)
            logger.debug(
                f"Unable to fetch column tags for {catalog}.{schema}.{table}: {e}. "
                f"This is expected if Unity Catalog tags are not enabled (requires DBR 13.3+)."
            )
            return {}


def fetch_databricks_query(statement_id: str) -> Optional[DatabricksQueryData]:
    """Fetch complete query data from Databricks by statement ID.

    This is a convenience function that creates a DatabricksSystemTables instance
    and fetches the query data in one call.

    Args:
        statement_id: Databricks statement/query ID

    Returns:
        DatabricksQueryData with query text, metrics, and cluster config

    Example:
        >>> from cloud_ceo.databricks import fetch_databricks_query
        >>> data = fetch_databricks_query("01HF6K2T3E8M9N7P6Q5R4S")
        >>> print(f"Query: {data.query_text}")
        >>> print(f"Duration: {data.execution_metrics.duration_ms}ms")
        >>> print(f"Cluster: {data.cluster_config.node_type}")
    """
    tables = DatabricksSystemTables()
    return tables.fetch_query_by_id(statement_id)


def analyze_databricks_query(
    statement_id: str,
    *,
    enable_llm: bool = False,
    config_path: Optional[Path] = None,
):
    """Analyze a Databricks query by statement ID with automatic data fetching.

    This convenience function automatically fetches:
    - Query text from system.query.history
    - Execution metrics (duration, bytes scanned, memory, etc.)
    - Cluster configuration (node type, workers, memory)
    - User and warehouse information

    All you need is the Databricks statement ID - everything else is retrieved automatically!

    Args:
        statement_id: Databricks statement/query ID
        enable_llm: Enable LLM-powered recommendations
        config_path: Optional path to config file

    Returns:
        AnalysisResult with violations and optional LLM analysis

    Raises:
        RuntimeError: If PySpark is not available or query not found

    Example:
        >>> from cloud_ceo.databricks import analyze_databricks_query
        >>>
        >>> # Just pass the statement ID - everything else is automatic!
        >>> result = analyze_databricks_query(
        ...     "01HF6K2T3E8M9N7P6Q5R4S",
        ...     enable_llm=True
        ... )
        >>> print(result.summary())
        >>> print(f"Cost: ${result.cost_usd:.2f}")
    """
    from cloud_ceo.core import analyze

    logger.info(f"Fetching Databricks query data for {statement_id}")

    # Fetch complete query data from system tables
    query_data = fetch_databricks_query(statement_id)

    if not query_data:
        raise ValueError(f"Query {statement_id} not found in system.query.history")

    # Check if query failed
    if query_data.execution_status != "FINISHED":
        logger.warning(
            f"Query {statement_id} status: {query_data.execution_status}"
        )
        if query_data.error_message:
            logger.warning(f"Error: {query_data.error_message}")

    # Analyze using standard analyze() function
    return analyze(
        query_data.query_text,
        query_id=statement_id,
        execution_metrics=query_data.execution_metrics,
        cluster_config=query_data.cluster_config,
        enable_llm=enable_llm,
        databricks_query_id=statement_id,
        user=query_data.user,
        warehouse_id=query_data.warehouse_id,
        config_path=config_path,
    )
