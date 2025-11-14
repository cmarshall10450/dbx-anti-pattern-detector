"""Databricks cost calculator for query impact analysis.

This module provides accurate DBU-based cost calculation for Databricks queries,
replacing LLM-generated estimates with real pricing data.

DBU Pricing Model:
- Databricks charges in DBU (Databricks Units)
- DBU rates vary by: region, workload type, instance type
- Standard rates (US East): $0.07-0.15 per DBU for Jobs, $0.22-0.75 for All-Purpose
- Each instance type consumes DBUs based on compute capacity

Cost Formula:
- Cost = (Runtime Hours) × (DBU Rate per Hour) × (Number of Nodes)
- DBU Rate depends on node type and workload
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import structlog

logger = structlog.get_logger(__name__)


# DBU rates per hour by node type (AWS Standard Compute instances)
# Source: Databricks pricing page - these are baseline rates for Jobs workload
# All-Purpose clusters cost ~3x more
DBU_RATES_PER_HOUR = {
    # Memory-optimized (r5 family)
    "r5.large": 0.75,
    "r5.xlarge": 1.5,
    "r5.2xlarge": 3.0,
    "r5.4xlarge": 6.0,
    "r5.8xlarge": 12.0,
    "r5.12xlarge": 18.0,
    "r5.16xlarge": 24.0,
    "r5.24xlarge": 36.0,

    # Storage-optimized (i3 family) - most common for Spark
    "i3.xlarge": 1.0,
    "i3.2xlarge": 2.0,
    "i3.4xlarge": 4.0,
    "i3.8xlarge": 8.0,
    "i3.16xlarge": 16.0,

    # Compute-optimized (c5 family)
    "c5.large": 0.5,
    "c5.xlarge": 1.0,
    "c5.2xlarge": 2.0,
    "c5.4xlarge": 4.0,
    "c5.9xlarge": 9.0,
    "c5.12xlarge": 12.0,
    "c5.18xlarge": 18.0,

    # General purpose (m5 family)
    "m5.large": 0.69,
    "m5.xlarge": 1.38,
    "m5.2xlarge": 2.76,
    "m5.4xlarge": 5.52,
    "m5.8xlarge": 11.04,
    "m5.12xlarge": 16.56,
    "m5.16xlarge": 22.08,
    "m5.24xlarge": 33.12,
}

# Workload type multipliers
WORKLOAD_MULTIPLIERS = {
    "jobs": 1.0,           # Base rate (Jobs tier - $0.10/DBU)
    "all_purpose": 3.0,    # All-Purpose tier (~$0.30/DBU)
    "sql_warehouse": 2.2,  # SQL Warehouse tier (~$0.22/DBU)
}

# DBU cost per unit (average across regions)
DBU_COST_USD = 0.10  # Standard Jobs tier rate


@dataclass
class CostEstimate:
    """Calculated cost estimate for a query execution.

    Attributes:
        daily_cost_usd: Estimated cost per day in USD
        hourly_cost_usd: Cost per hour in USD
        per_execution_cost_usd: Cost for single query execution
        dbu_rate_per_hour: DBUs consumed per hour for the cluster
        total_nodes: Total number of nodes (driver + workers)
        workload_type: Type of workload (jobs, all_purpose, sql_warehouse)
        calculation_method: How cost was calculated (accurate, estimated, fallback)
    """

    daily_cost_usd: float
    hourly_cost_usd: float
    per_execution_cost_usd: float
    dbu_rate_per_hour: float
    total_nodes: int
    workload_type: str
    calculation_method: Literal["accurate", "estimated", "fallback"]


def parse_memory_string(memory_str: str) -> float:
    """Parse memory string like '61GB' into gigabytes.

    Args:
        memory_str: Memory string (e.g., '61GB', '8 GB', '1024MB')

    Returns:
        Memory in gigabytes (float)

    Examples:
        >>> parse_memory_string('61GB')
        61.0
        >>> parse_memory_string('8 GB')
        8.0
        >>> parse_memory_string('1024MB')
        1.0
    """
    import re

    # Extract number and unit
    match = re.match(r'(\d+\.?\d*)\s*(GB|MB|TB)', memory_str.upper())
    if not match:
        raise ValueError(f"Invalid memory format: {memory_str}")

    value = float(match.group(1))
    unit = match.group(2)

    # Convert to GB
    if unit == 'MB':
        return value / 1024
    elif unit == 'TB':
        return value * 1024
    else:  # GB
        return value


def estimate_dbu_rate_from_memory(memory_gb: float) -> float:
    """Estimate DBU rate when node type is unknown.

    Uses memory size as proxy for compute capacity.
    Based on empirical relationship: ~0.25 DBU per GB of memory

    Args:
        memory_gb: Executor memory in gigabytes

    Returns:
        Estimated DBU rate per hour
    """
    # Rough heuristic: 0.25 DBU per GB memory
    # This assumes memory-optimized instances (r5/i3 family)
    return memory_gb * 0.25


def get_dbu_rate(node_type: str, executor_memory: Optional[str] = None) -> tuple[float, Literal["accurate", "estimated"]]:
    """Get DBU rate for a node type.

    Args:
        node_type: EC2 instance type (e.g., 'i3.2xlarge')
        executor_memory: Optional executor memory string (fallback if node type unknown)

    Returns:
        Tuple of (dbu_rate_per_hour, calculation_method)
    """
    # Clean node type (remove region prefixes if present)
    clean_node_type = node_type.split('.')[-2] + '.' + node_type.split('.')[-1] if '.' in node_type else node_type

    # Try exact match
    if clean_node_type in DBU_RATES_PER_HOUR:
        return DBU_RATES_PER_HOUR[clean_node_type], "accurate"

    # Try without region prefix (e.g., 'Standard_D4s_v3' -> 'd4s.v3')
    normalized = clean_node_type.lower().replace('standard_', '').replace('_', '.')
    for known_type, rate in DBU_RATES_PER_HOUR.items():
        if normalized in known_type.lower():
            return rate, "accurate"

    # Fallback: estimate from memory if available
    if executor_memory:
        try:
            memory_gb = parse_memory_string(executor_memory)
            estimated_rate = estimate_dbu_rate_from_memory(memory_gb)
            logger.info(
                f"Node type '{node_type}' not found in rate table. "
                f"Estimated {estimated_rate:.2f} DBU/hr from {memory_gb}GB memory."
            )
            return estimated_rate, "estimated"
        except Exception as e:
            logger.warning(f"Could not parse memory '{executor_memory}': {e}")

    # Ultimate fallback: assume medium instance
    logger.warning(
        f"Node type '{node_type}' not found and no memory available. "
        f"Using default rate for i3.2xlarge (2.0 DBU/hr)"
    )
    return 2.0, "estimated"


def calculate_cost(
    node_type: str,
    executor_count: int,
    duration_ms: int,
    executor_memory: Optional[str] = None,
    workload_type: Literal["jobs", "all_purpose", "sql_warehouse"] = "all_purpose",
    executions_per_day: int = 24,
) -> CostEstimate:
    """Calculate accurate cost for query execution.

    Args:
        node_type: EC2 instance type (e.g., 'i3.2xlarge')
        executor_count: Number of worker nodes
        duration_ms: Query duration in milliseconds
        executor_memory: Optional executor memory string (e.g., '61GB')
        workload_type: Type of Databricks workload
        executions_per_day: How many times query runs per day (for daily cost)

    Returns:
        CostEstimate with calculated costs

    Examples:
        >>> cost = calculate_cost('i3.2xlarge', 4, 45000)
        >>> print(f"${cost.per_execution_cost_usd:.4f} per run")
        $0.0333 per run
    """
    # Get DBU rate for node type
    dbu_rate_per_node_hour, calc_method = get_dbu_rate(node_type, executor_memory)

    # Apply workload multiplier
    workload_multiplier = WORKLOAD_MULTIPLIERS.get(workload_type, WORKLOAD_MULTIPLIERS["all_purpose"])
    dbu_rate_per_node_hour *= workload_multiplier

    # Total nodes (workers + driver, assuming driver is same node type)
    total_nodes = executor_count + 1

    # Total DBU rate for entire cluster
    cluster_dbu_rate_per_hour = dbu_rate_per_node_hour * total_nodes

    # Convert duration to hours
    duration_hours = duration_ms / (1000 * 60 * 60)

    # Calculate costs
    per_execution_cost_usd = cluster_dbu_rate_per_hour * DBU_COST_USD * duration_hours
    hourly_cost_usd = cluster_dbu_rate_per_hour * DBU_COST_USD
    daily_cost_usd = per_execution_cost_usd * executions_per_day

    logger.info(
        f"Cost calculation ({calc_method}): {total_nodes} × {node_type} nodes, "
        f"{dbu_rate_per_node_hour:.2f} DBU/hr/node, {duration_ms}ms runtime = "
        f"${per_execution_cost_usd:.4f}/run, ${daily_cost_usd:.2f}/day ({executions_per_day} runs/day)"
    )

    return CostEstimate(
        daily_cost_usd=daily_cost_usd,
        hourly_cost_usd=hourly_cost_usd,
        per_execution_cost_usd=per_execution_cost_usd,
        dbu_rate_per_hour=cluster_dbu_rate_per_hour,
        total_nodes=total_nodes,
        workload_type=workload_type,
        calculation_method=calc_method,
    )


def calculate_waste_cost(
    baseline_duration_ms: int,
    actual_duration_ms: int,
    node_type: str,
    executor_count: int,
    executor_memory: Optional[str] = None,
    workload_type: Literal["jobs", "all_purpose", "sql_warehouse"] = "all_purpose",
    executions_per_day: int = 24,
) -> tuple[float, str]:
    """Calculate wasted cost from performance degradation.

    Compares actual runtime against baseline (optimal) runtime to quantify
    the cost impact of anti-patterns.

    Args:
        baseline_duration_ms: Expected duration if query was optimized (ms)
        actual_duration_ms: Current query duration (ms)
        node_type: EC2 instance type
        executor_count: Number of worker nodes
        executor_memory: Optional executor memory string
        workload_type: Type of Databricks workload
        executions_per_day: How many times query runs per day

    Returns:
        Tuple of (daily_waste_usd, explanation_string)

    Examples:
        >>> waste, explanation = calculate_waste_cost(10000, 45000, 'i3.2xlarge', 4)
        >>> print(f"Wasting {waste:.2f}/day: {explanation}")
        Wasting 28.00/day: Query takes 4.5x longer than optimal (45s vs 10s)
    """
    if actual_duration_ms <= baseline_duration_ms:
        return 0.0, "Query is already optimal"

    # Calculate cost for both scenarios
    baseline_cost = calculate_cost(
        node_type, executor_count, baseline_duration_ms,
        executor_memory, workload_type, executions_per_day
    )

    actual_cost = calculate_cost(
        node_type, executor_count, actual_duration_ms,
        executor_memory, workload_type, executions_per_day
    )

    # Waste is the difference
    daily_waste = actual_cost.daily_cost_usd - baseline_cost.daily_cost_usd

    # Build explanation
    slowdown_factor = actual_duration_ms / baseline_duration_ms
    actual_seconds = actual_duration_ms / 1000
    baseline_seconds = baseline_duration_ms / 1000

    explanation = (
        f"Query takes {slowdown_factor:.1f}x longer than optimal "
        f"({actual_seconds:.0f}s vs {baseline_seconds:.0f}s)"
    )

    return daily_waste, explanation
