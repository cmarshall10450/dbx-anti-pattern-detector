"""Sample SQL Queries for Cloud CEO DBX Demonstrations

This module contains a comprehensive library of good and bad SQL queries
organized by industry and anti-pattern type. Each query includes metadata
about expected violations, cost impact, and recommended fixes.

Author: Cloud CEO Team
Version: 1.0.0
"""

from typing import Dict, List

# ============================================================================
# Financial Services Queries
# ============================================================================

FINANCIAL_BAD_QUERIES = [
    {
        "id": "FIN_BAD_001",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "cartesian_join",
        "query": """
SELECT *
FROM financial_customers c, financial_transactions t, financial_fraud_alerts f
WHERE t.amount > 1000
  AND f.severity = 'critical'
""",
        "description": "Cartesian join between three large tables without proper join conditions",
        "expected_violations": ["SPARK_004", "SPARK_001"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add explicit JOIN conditions: c.customer_id = t.customer_id AND t.transaction_id = f.transaction_id",
        "persona": "junior_analyst"
    },
    {
        "id": "FIN_BAD_002",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "select_star",
        "query": """
SELECT *
FROM financial_transactions
WHERE transaction_date >= '2024-01-01'
""",
        "description": "SELECT * on wide transaction table with 50+ columns",
        "expected_violations": ["SPARK_001"],
        "expected_cost_impact": "high",
        "recommended_fix": "Select only needed columns: transaction_id, customer_id, amount, transaction_date, merchant_category",
        "persona": "data_analyst"
    },
    {
        "id": "FIN_BAD_003",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "non_sargable_predicate",
        "query": """
SELECT customer_id, name, account_balance
FROM financial_customers
WHERE YEAR(registration_date) = 2024
  AND UPPER(kyc_status) = 'ACTIVE'
""",
        "description": "Functions on indexed columns prevent optimizer from using indexes",
        "expected_violations": ["SPARK_008"],
        "expected_cost_impact": "high",
        "recommended_fix": "Use sargable predicates: registration_date >= '2024-01-01' AND registration_date < '2025-01-01' AND kyc_status = 'ACTIVE'",
        "persona": "data_engineer"
    },
    {
        "id": "FIN_BAD_004",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "missing_partition_filter",
        "query": """
SELECT customer_id, SUM(amount) as total_amount
FROM financial_transactions
WHERE merchant_category = 'Retail'
GROUP BY customer_id
""",
        "description": "Missing partition filter causes full table scan on date-partitioned table",
        "expected_violations": ["SPARK_005"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add partition filter: WHERE transaction_date >= '2024-01-01' AND merchant_category = 'Retail'",
        "persona": "data_analyst"
    },
    {
        "id": "FIN_BAD_005",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "like_without_wildcards",
        "query": """
SELECT customer_id, name, email
FROM financial_customers
WHERE email LIKE 'john.smith@example.com'
""",
        "description": "LIKE without wildcards should use equality operator",
        "expected_violations": ["SPARK_007"],
        "expected_cost_impact": "low",
        "recommended_fix": "Use equality: WHERE email = 'john.smith@example.com'",
        "persona": "junior_analyst"
    },
    {
        "id": "FIN_BAD_006",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "or_to_in",
        "query": """
SELECT *
FROM financial_customers
WHERE risk_rating = 'high'
   OR risk_rating = 'critical'
   OR risk_rating = 'severe'
   OR risk_rating = 'extreme'
""",
        "description": "Multiple OR conditions should use IN clause",
        "expected_violations": ["SPARK_009", "SPARK_001"],
        "expected_cost_impact": "medium",
        "recommended_fix": "Use IN: WHERE risk_rating IN ('high', 'critical', 'severe', 'extreme')",
        "persona": "data_analyst"
    },
    {
        "id": "FIN_BAD_007",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "multiple_count_distinct",
        "query": """
SELECT
    transaction_date,
    COUNT(DISTINCT customer_id) as unique_customers,
    COUNT(DISTINCT merchant_category) as unique_merchants,
    COUNT(DISTINCT country_code) as unique_countries,
    COUNT(DISTINCT transaction_id) as unique_transactions
FROM financial_transactions
WHERE transaction_date >= '2024-01-01'
GROUP BY transaction_date
""",
        "description": "Multiple COUNT(DISTINCT) operations cause expensive shuffles",
        "expected_violations": ["SPARK_010"],
        "expected_cost_impact": "high",
        "recommended_fix": "Use separate queries or approximate_count_distinct() for large datasets",
        "persona": "data_engineer"
    },
    {
        "id": "FIN_BAD_008",
        "industry": "financial",
        "quality": "bad",
        "anti_pattern_type": "group_by_cardinality",
        "query": """
SELECT
    account_number,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount
FROM financial_customers c
JOIN financial_transactions t ON c.customer_id = t.customer_id
WHERE transaction_date >= '2024-01-01'
GROUP BY account_number
""",
        "description": "GROUP BY on high cardinality column (account_number) causes excessive shuffling",
        "expected_violations": ["SPARK_011"],
        "expected_cost_impact": "high",
        "recommended_fix": "GROUP BY customer_id instead (lower cardinality), join with customers later if account_number display is needed",
        "persona": "data_engineer"
    }
]

FINANCIAL_GOOD_QUERIES = [
    {
        "id": "FIN_GOOD_001",
        "industry": "financial",
        "quality": "good",
        "query": """
SELECT
    c.customer_id,
    c.name,
    c.account_balance,
    t.transaction_id,
    t.amount,
    t.transaction_date
FROM financial_customers c
INNER JOIN financial_transactions t
    ON c.customer_id = t.customer_id
WHERE t.transaction_date >= '2024-01-01'
  AND t.transaction_date < '2024-02-01'
  AND t.amount > 1000
  AND c.risk_rating IN ('high', 'critical')
""",
        "description": "Well-optimized query with explicit columns, proper joins, and partition filtering",
        "expected_violations": [],
        "expected_cost_impact": "optimal",
        "persona": "senior_engineer"
    },
    {
        "id": "FIN_GOOD_002",
        "industry": "financial",
        "quality": "good",
        "query": """
SELECT
    customer_id,
    COUNT(*) as transaction_count,
    SUM(amount) as total_amount,
    AVG(amount) as avg_amount
FROM financial_transactions
WHERE transaction_date >= '2024-01-01'
  AND transaction_date < '2024-02-01'
  AND merchant_category = 'Retail'
GROUP BY customer_id
HAVING COUNT(*) > 10
ORDER BY total_amount DESC
LIMIT 100
""",
        "description": "Efficient aggregation with partition filter and explicit columns",
        "expected_violations": [],
        "expected_cost_impact": "optimal",
        "persona": "data_analyst"
    }
]

# ============================================================================
# Healthcare Queries
# ============================================================================

HEALTHCARE_BAD_QUERIES = [
    {
        "id": "HC_BAD_001",
        "industry": "healthcare",
        "quality": "bad",
        "anti_pattern_type": "select_star",
        "query": """
SELECT *
FROM healthcare_patients
WHERE UPPER(name) LIKE '%SMITH%'
""",
        "description": "SELECT * on PHI table with non-sargable predicate",
        "expected_violations": ["SPARK_001", "SPARK_008"],
        "expected_cost_impact": "high",
        "recommended_fix": "Select specific columns and use sargable predicate: WHERE name LIKE '%Smith%' (case-insensitive comparison at index level)",
        "persona": "junior_analyst",
        "compliance_note": "HIPAA: Violates minimum necessary principle"
    },
    {
        "id": "HC_BAD_002",
        "industry": "healthcare",
        "quality": "bad",
        "anti_pattern_type": "cartesian_join",
        "query": """
SELECT p.patient_id, p.name, a.appointment_date, pr.medication_name
FROM healthcare_patients p, healthcare_appointments a, healthcare_prescriptions pr
WHERE a.department = 'Cardiology'
""",
        "description": "Cartesian join across three healthcare tables",
        "expected_violations": ["SPARK_004"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add proper joins: JOIN appointments a ON p.patient_id = a.patient_id JOIN prescriptions pr ON a.patient_id = pr.patient_id",
        "persona": "junior_analyst",
        "compliance_note": "HIPAA: May expose unauthorized PHI combinations"
    },
    {
        "id": "HC_BAD_003",
        "industry": "healthcare",
        "quality": "bad",
        "anti_pattern_type": "missing_partition_filter",
        "query": """
SELECT
    patient_id,
    COUNT(*) as visit_count,
    SUM(cost_gbp) as total_cost
FROM healthcare_appointments
WHERE department = 'Emergency'
GROUP BY patient_id
""",
        "description": "Missing date partition filter on appointments table",
        "expected_violations": ["SPARK_005"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add partition filter: WHERE appointment_date >= '2024-01-01' AND department = 'Emergency'",
        "persona": "data_analyst",
        "compliance_note": "NHS Data Toolkit: Inefficient data access"
    },
    {
        "id": "HC_BAD_004",
        "industry": "healthcare",
        "quality": "bad",
        "anti_pattern_type": "non_sargable_predicate",
        "query": """
SELECT patient_id, medication_name, dosage
FROM healthcare_prescriptions
WHERE YEAR(prescribed_date) = 2024
  AND MONTH(prescribed_date) IN (1, 2, 3)
""",
        "description": "Date functions prevent partition pruning",
        "expected_violations": ["SPARK_008"],
        "expected_cost_impact": "high",
        "recommended_fix": "Use date range: WHERE prescribed_date >= '2024-01-01' AND prescribed_date < '2024-04-01'",
        "persona": "data_engineer"
    }
]

HEALTHCARE_GOOD_QUERIES = [
    {
        "id": "HC_GOOD_001",
        "industry": "healthcare",
        "quality": "good",
        "query": """
SELECT
    p.patient_id,
    p.condition_category,
    COUNT(a.appointment_id) as appointment_count,
    SUM(a.cost_gbp) as total_cost
FROM healthcare_patients p
INNER JOIN healthcare_appointments a
    ON p.patient_id = a.patient_id
WHERE a.appointment_date >= '2024-01-01'
  AND a.appointment_date < '2024-02-01'
  AND a.department = 'Cardiology'
GROUP BY p.patient_id, p.condition_category
""",
        "description": "HIPAA-compliant query with minimum necessary columns and proper filtering",
        "expected_violations": [],
        "expected_cost_impact": "optimal",
        "persona": "senior_engineer",
        "compliance_note": "HIPAA: Adheres to minimum necessary principle"
    }
]

# ============================================================================
# Energy/Utilities Queries
# ============================================================================

ENERGY_BAD_QUERIES = [
    {
        "id": "ENERGY_BAD_001",
        "industry": "energy",
        "quality": "bad",
        "anti_pattern_type": "missing_partition_filter",
        "query": """
SELECT
    customer_id,
    meter_id,
    AVG(kwh_consumed) as avg_consumption
FROM energy_smart_meter_readings
WHERE kwh_consumed > 5.0
GROUP BY customer_id, meter_id
""",
        "description": "Missing date filter on 50M+ row time-series table",
        "expected_violations": ["SPARK_005"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add partition filter: WHERE date(reading_datetime) >= '2024-01-01' AND kwh_consumed > 5.0",
        "persona": "data_analyst",
        "compliance_note": "Ofgem: Inefficient processing of behavioral data"
    },
    {
        "id": "ENERGY_BAD_002",
        "industry": "energy",
        "quality": "bad",
        "anti_pattern_type": "select_star",
        "query": """
SELECT *
FROM energy_smart_meter_readings
WHERE reading_datetime >= '2024-01-01'
""",
        "description": "SELECT * on behavioral data table (GDPR profiling risk)",
        "expected_violations": ["SPARK_001"],
        "expected_cost_impact": "high",
        "recommended_fix": "Select only required columns to minimize behavioral data exposure",
        "persona": "junior_analyst",
        "compliance_note": "GDPR Art. 22: Data minimization for profiling prevention"
    },
    {
        "id": "ENERGY_BAD_003",
        "industry": "energy",
        "quality": "bad",
        "anti_pattern_type": "cartesian_join",
        "query": """
SELECT c.customer_id, c.tariff_type, r.kwh_consumed
FROM energy_customers c, energy_smart_meter_readings r
WHERE c.vulnerability_flag = true
""",
        "description": "Cartesian join between customers and meter readings",
        "expected_violations": ["SPARK_004"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add join condition: JOIN energy_smart_meter_readings r ON c.customer_id = r.customer_id",
        "persona": "junior_analyst",
        "compliance_note": "ComCoP: Unauthorized exposure of vulnerable customer data"
    }
]

ENERGY_GOOD_QUERIES = [
    {
        "id": "ENERGY_GOOD_001",
        "industry": "energy",
        "quality": "good",
        "query": """
SELECT
    c.customer_id,
    c.tariff_type,
    AVG(r.kwh_consumed) as avg_daily_consumption,
    SUM(r.kwh_consumed * r.tariff_rate) as estimated_cost
FROM energy_customers c
INNER JOIN energy_smart_meter_readings r
    ON c.customer_id = r.customer_id
WHERE date(r.reading_datetime) >= '2024-01-01'
  AND date(r.reading_datetime) < '2024-02-01'
  AND c.vulnerability_flag = false
GROUP BY c.customer_id, c.tariff_type
""",
        "description": "Compliant query with partition filter and data minimization",
        "expected_violations": [],
        "expected_cost_impact": "optimal",
        "persona": "senior_engineer",
        "compliance_note": "Ofgem & GDPR compliant with proper filtering"
    }
]

# ============================================================================
# General Enterprise Queries
# ============================================================================

ENTERPRISE_BAD_QUERIES = [
    {
        "id": "ENT_BAD_001",
        "industry": "enterprise",
        "quality": "bad",
        "anti_pattern_type": "missing_partition_filter",
        "query": """
SELECT
    user_id,
    event_type,
    COUNT(*) as event_count
FROM enterprise_events
WHERE event_type IN ('purchase', 'signup')
GROUP BY user_id, event_type
""",
        "description": "Missing partition filter on 100M+ event table",
        "expected_violations": ["SPARK_005"],
        "expected_cost_impact": "critical",
        "recommended_fix": "Add partition filter: WHERE date(timestamp) >= '2024-01-01' AND event_type IN ('purchase', 'signup')",
        "persona": "data_analyst"
    },
    {
        "id": "ENT_BAD_002",
        "industry": "enterprise",
        "quality": "bad",
        "anti_pattern_type": "distinct_on_unique",
        "query": """
SELECT DISTINCT user_id, email
FROM enterprise_users
WHERE subscription_tier = 'premium'
""",
        "description": "DISTINCT on unique key (user_id) is unnecessary",
        "expected_violations": ["SPARK_012"],
        "expected_cost_impact": "medium",
        "recommended_fix": "Remove DISTINCT since user_id is already unique",
        "persona": "junior_analyst"
    },
    {
        "id": "ENT_BAD_003",
        "industry": "enterprise",
        "quality": "bad",
        "anti_pattern_type": "cartesian_join",
        "query": """
SELECT u.user_id, u.email, p.product_id, p.name
FROM enterprise_users u, enterprise_products p
WHERE u.subscription_tier = 'enterprise'
  AND p.category = 'Electronics'
""",
        "description": "Unintentional cartesian product between users and products",
        "expected_violations": ["SPARK_004"],
        "expected_cost_impact": "critical",
        "recommended_fix": "If showing product catalog, clarify intent with CROSS JOIN. Otherwise, add proper join condition via intermediate table.",
        "persona": "junior_analyst"
    }
]

ENTERPRISE_GOOD_QUERIES = [
    {
        "id": "ENT_GOOD_001",
        "industry": "enterprise",
        "quality": "good",
        "query": """
SELECT
    e.event_type,
    COUNT(*) as event_count,
    COUNT(DISTINCT e.user_id) as unique_users,
    COUNT(DISTINCT e.session_id) as unique_sessions
FROM enterprise_events e
WHERE date(e.timestamp) >= '2024-01-01'
  AND date(e.timestamp) < '2024-02-01'
  AND e.event_type IN ('purchase', 'signup', 'click')
GROUP BY e.event_type
ORDER BY event_count DESC
""",
        "description": "Well-optimized analytics query with partition filtering",
        "expected_violations": [],
        "expected_cost_impact": "optimal",
        "persona": "data_analyst"
    }
]

# ============================================================================
# Helper Functions
# ============================================================================

def get_all_sample_queries() -> List[Dict]:
    """Get all sample queries across all industries."""
    return (
        FINANCIAL_BAD_QUERIES + FINANCIAL_GOOD_QUERIES +
        HEALTHCARE_BAD_QUERIES + HEALTHCARE_GOOD_QUERIES +
        ENERGY_BAD_QUERIES + ENERGY_GOOD_QUERIES +
        ENTERPRISE_BAD_QUERIES + ENTERPRISE_GOOD_QUERIES
    )


def get_queries_by_industry(industry: str) -> List[Dict]:
    """Get queries for specific industry."""
    all_queries = get_all_sample_queries()
    return [q for q in all_queries if q["industry"] == industry]


def get_bad_queries() -> List[Dict]:
    """Get all bad example queries."""
    all_queries = get_all_sample_queries()
    return [q for q in all_queries if q["quality"] == "bad"]


def get_good_queries() -> List[Dict]:
    """Get all good example queries."""
    all_queries = get_all_sample_queries()
    return [q for q in all_queries if q["quality"] == "good"]


def get_queries_by_anti_pattern(anti_pattern_type: str) -> List[Dict]:
    """Get queries demonstrating specific anti-pattern."""
    all_queries = get_all_sample_queries()
    return [q for q in all_queries if q.get("anti_pattern_type") == anti_pattern_type]


def get_critical_cost_queries() -> List[Dict]:
    """Get queries with critical cost impact."""
    all_queries = get_all_sample_queries()
    return [q for q in all_queries if q.get("expected_cost_impact") == "critical"]


def save_queries_to_files(output_dir: str = ".") -> None:
    """Save queries to SQL files organized by industry and quality.

    Args:
        output_dir: Directory to save SQL files
    """
    import os

    query_dir = os.path.join(output_dir, "query_data")
    os.makedirs(query_dir, exist_ok=True)

    industries = ["financial", "healthcare", "energy", "enterprise"]
    qualities = ["bad", "good"]

    for industry in industries:
        for quality in qualities:
            filename = f"{industry}_{quality}.sql"
            filepath = os.path.join(query_dir, filename)

            queries = [
                q for q in get_all_sample_queries()
                if q["industry"] == industry and q["quality"] == quality
            ]

            with open(filepath, "w") as f:
                f.write(f"-- {industry.upper()} {quality.upper()} QUERIES\n")
                f.write(f"-- Generated by Cloud CEO DBX Demo Setup\n")
                f.write(f"-- Count: {len(queries)} queries\n\n")

                for query in queries:
                    f.write(f"-- {query['id']}: {query['description']}\n")
                    if query.get("expected_violations"):
                        f.write(f"-- Expected violations: {', '.join(query['expected_violations'])}\n")
                    if query.get("expected_cost_impact"):
                        f.write(f"-- Cost impact: {query['expected_cost_impact']}\n")
                    if query.get("recommended_fix"):
                        f.write(f"-- Fix: {query['recommended_fix']}\n")
                    if query.get("compliance_note"):
                        f.write(f"-- Compliance: {query['compliance_note']}\n")
                    f.write(f"\n{query['query'].strip()}\n")
                    f.write("\n" + "="*80 + "\n\n")

    print(f"✅ Saved {len(industries) * len(qualities)} SQL files to {query_dir}")


# ============================================================================
# Cost Impact Simulation
# ============================================================================

def generate_execution_metrics(anti_pattern_type: str, scale: str = "medium") -> Dict:
    """Generate realistic execution metrics for query simulation.

    Args:
        anti_pattern_type: Type of anti-pattern (affects cost multiplier)
        scale: Data scale (small/medium/large)

    Returns:
        Dictionary with execution metrics
    """
    import random

    # Base metrics
    base_duration_ms = 5000  # 5 seconds
    base_bytes_scanned = 1_000_000_000  # 1 GB
    base_rows_scanned = 1_000_000  # 1M rows
    base_cost_usd = 0.50  # $0.50

    # Cost multipliers by anti-pattern
    multipliers = {
        "cartesian_join": 50.0,
        "missing_partition_filter": 10.0,
        "select_star": 3.0,
        "non_sargable_predicate": 5.0,
        "multiple_count_distinct": 8.0,
        "group_by_cardinality": 6.0,
        "or_to_in": 2.0,
        "like_without_wildcards": 1.2,
        "distinct_on_unique": 1.5,
        "optimal": 1.0
    }

    # Scale multipliers
    scale_multipliers = {
        "small": 0.1,
        "medium": 1.0,
        "large": 10.0
    }

    multiplier = multipliers.get(anti_pattern_type, 2.0) * scale_multipliers.get(scale, 1.0)

    # Add some randomness
    multiplier *= random.uniform(0.8, 1.2)

    return {
        "duration_ms": int(base_duration_ms * multiplier),
        "bytes_scanned": int(base_bytes_scanned * multiplier),
        "rows_scanned": int(base_rows_scanned * multiplier),
        "peak_memory_mb": int(4096 * (multiplier ** 0.5)),
        "cost_usd": round(base_cost_usd * multiplier, 2),
        "shuffle_read_bytes": int(base_bytes_scanned * multiplier * 0.3) if multiplier > 2 else 0,
        "shuffle_write_bytes": int(base_bytes_scanned * multiplier * 0.2) if multiplier > 2 else 0,
        "query_start_time": "2024-01-15T10:30:00Z",
        "warehouse_id": f"wh_{random.randint(1000, 9999)}",
        "user_email": random.choice([
            "analyst@company.com",
            "engineer@company.com",
            "scientist@company.com"
        ]),
        "cluster_config": {
            "node_type": "i3.2xlarge",
            "executor_count": min(20, max(2, int(multiplier / 5))),
            "executor_memory": "61GB"
        }
    }


if __name__ == "__main__":
    # When run directly, save queries to files
    print("Saving sample queries to SQL files...")
    save_queries_to_files()

    # Print summary
    all_queries = get_all_sample_queries()
    print(f"\n📊 Sample Query Library Summary:")
    print(f"   Total queries: {len(all_queries)}")
    print(f"   Bad queries: {len(get_bad_queries())}")
    print(f"   Good queries: {len(get_good_queries())}")
    print(f"   Critical cost impact: {len(get_critical_cost_queries())}")
    print(f"\nIndustries:")
    for industry in ["financial", "healthcare", "energy", "enterprise"]:
        count = len(get_queries_by_industry(industry))
        print(f"   {industry}: {count} queries")
