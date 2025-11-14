# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Cloud CEO DBX Demo Environment Setup
# MAGIC
# MAGIC This notebook creates comprehensive sample datasets and queries for demonstrating Cloud CEO's anti-pattern detection and optimization capabilities.
# MAGIC
# MAGIC **Features:**
# MAGIC - 🏦 Financial Services (customers, transactions, fraud alerts)
# MAGIC - 🏥 Healthcare (patients, appointments, prescriptions)
# MAGIC - ⚡ Energy/Utilities (customers, smart meter readings, outages)
# MAGIC - 🏢 General Enterprise (users, events, products)
# MAGIC - 📝 Sample queries (good and bad examples)
# MAGIC - 🏷️ Unity Catalog tags for compliance
# MAGIC
# MAGIC **Data Generator:** dbldatagen with Faker integration for realistic, scalable data generation
# MAGIC
# MAGIC **Author:** Cloud CEO Team
# MAGIC **Version:** 2.0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📦 Installation
# MAGIC
# MAGIC Install required dependencies:
# MAGIC - **dbldatagen** - High-performance synthetic data generation
# MAGIC - **faker** - Realistic personal/business data generation

# COMMAND ----------

# Install dependencies
%pip install dbldatagen faker

# COMMAND ----------

# Restart Python to use newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 Imports

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, TimestampType
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Configuration Constants

# COMMAND ----------

DEMO_CATALOG = "demo"
DEMO_SCHEMA = "cloud_ceo_demo"
DEMO_FULL_NAME = f"{DEMO_CATALOG}.{DEMO_SCHEMA}"

# Scale factors for different demo sizes (updated for dbldatagen performance)
SCALE_FACTORS = {
    "small": {
        "multiplier": 0.001,
        "setup_time": "< 30 sec",
        "transactions": 10_000,
        "customers": 1_000,
        "description": "Quick demo with small datasets"
    },
    "medium": {
        "multiplier": 0.01,
        "setup_time": "1-2 min",
        "transactions": 1_000_000,
        "customers": 100_000,
        "description": "Balanced demo with realistic data volumes"
    },
    "large": {
        "multiplier": 0.1,
        "setup_time": "2-4 min",
        "transactions": 100_000_000,
        "customers": 10_000_000,
        "description": "Full-scale demo with production-like volumes"
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🛠️ Helper Functions
# MAGIC
# MAGIC Utility functions for Spark configuration and dependency validation.

# COMMAND ----------

def configure_spark_for_datagen(spark: SparkSession) -> None:
    """Configure Spark for optimal data generation performance."""
    spark.conf.set("spark.sql.shuffle.partitions", "128")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# COMMAND ----------

def validate_dependencies() -> Tuple[bool, str]:
    """Validate that required libraries are installed.

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        import dbldatagen as dg
        from dbldatagen import fakerText
        from faker import Faker
        return True, ""
    except ImportError as e:
        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    MISSING REQUIRED DEPENDENCIES                         ║
╚══════════════════════════════════════════════════════════════════════════╝

The demo setup requires dbldatagen and faker libraries.

In Databricks, run these commands in separate cells:

  Cell 1:
    %pip install dbldatagen faker

  Cell 2:
    dbutils.library.restartPython()

  Cell 3:
    Then run setup again

Technical error: {str(e)}
"""
        return False, error_msg

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏦 Financial Services Table Creation

# COMMAND ----------

def create_financial_tables(
    spark: SparkSession,
    schema: str,
    multiplier: float
) -> Dict[str, Dict]:
    """Create financial services sample tables using dbldatagen."""
    import dbldatagen as dg
    from dbldatagen import fakerText

    tables = {}

    # Calculate row counts
    num_customers = int(1_000_000 * multiplier)
    num_transactions = int(10_000_000 * multiplier)
    num_fraud_alerts = int(50_000 * multiplier)

    # Determine partitions based on row count
    customer_partitions = max(4, min(32, num_customers // 25000))
    transaction_partitions = max(8, min(128, num_transactions // 50000))
    fraud_partitions = max(2, min(16, num_fraud_alerts // 5000))

    # 1. Customers table with realistic personal data
    print(f"  Creating customers table ({num_customers:,} rows)...")

    customer_spec = (
        dg.DataGenerator(spark, name="customers", rows=num_customers,
                        partitions=customer_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("customer_id", "long", minValue=1000000, maxValue=9999999,
                   uniqueValues=num_customers)

        .withColumn("account_number", "string",
                   template=r"ACCT\d{10}",
                   baseColumn="customer_id")

        .withColumn("name", "string",
                   text=fakerText("name"),
                   baseColumn="customer_id")
        .withColumn("email", "string",
                   text=fakerText("ascii_company_email"),
                   baseColumn="customer_id")

        .withColumn("kyc_status", "string",
                   values=["active", "pending", "suspended", "closed"],
                   weights=[70, 15, 10, 5],
                   random=True)

        .withColumn("risk_rating", "string",
                   values=["low", "medium", "high", "critical"],
                   weights=[40, 35, 20, 5],
                   random=True)

        .withColumn("registration_date", "date",
                   begin="2020-01-01", end="2024-12-31",
                   random=True)

        .withColumn("account_balance", "decimal(12,2)",
                   minValue=0, maxValue=1000000,
                   distribution=dg.distributions.Normal(mean=50000, stddev=30000),
                   random=True)
    )

    customers_df = customer_spec.build()
    customers_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.customers")
    tables["customers"] = {"row_count": num_customers, "industry": "financial"}

    # 2. Transactions table (partitioned by date)
    print(f"  Creating transactions table ({num_transactions:,} rows)...")

    transaction_spec = (
        dg.DataGenerator(spark, name="transactions", rows=num_transactions,
                        partitions=transaction_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("transaction_id", "string",
                   template=r"TXN\w{16}")

        .withColumn("customer_id", "long",
                   minValue=1000000, maxValue=1000000 + num_customers - 1,
                   random=True)

        .withColumn("amount", "decimal(10,2)",
                   minValue=0.01, maxValue=50000,
                   distribution=dg.distributions.Gamma(shape=1.5, scale=2.0),
                   random=True)

        .withColumn("transaction_date", "date",
                   begin="2024-01-01", end="2024-12-31",
                   random=True)

        .withColumn("merchant_category", "string",
                   values=["Retail", "Restaurant", "Gas Station", "Grocery", "Online",
                          "Travel", "Entertainment", "Healthcare", "Utilities", "Other"],
                   weights=[25, 20, 10, 15, 15, 5, 5, 2, 2, 1],
                   random=True)

        .withColumn("country_code", "string",
                   values=["US", "GB", "CA", "DE", "FR", "AU", "JP", "BR"],
                   weights=[40, 25, 10, 8, 7, 5, 3, 2],
                   random=True)

        .withColumn("is_suspicious", "boolean",
                   expr="amount > 10000",
                   baseColumn="amount")

        .withColumn("created_at", "timestamp",
                   expr="current_timestamp()")
    )

    transactions_df = transaction_spec.build()
    transactions_df.write.mode("overwrite").format("delta") \
        .partitionBy("transaction_date") \
        .saveAsTable(f"{schema}.transactions")
    tables["transactions"] = {"row_count": num_transactions, "industry": "financial", "partitioned": True}

    # 3. Fraud alerts table
    print(f"  Creating fraud_alerts table ({num_fraud_alerts:,} rows)...")

    fraud_spec = (
        dg.DataGenerator(spark, name="fraud_alerts", rows=num_fraud_alerts,
                        partitions=fraud_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("alert_id", "long", minValue=1, uniqueValues=num_fraud_alerts)

        .withColumn("transaction_id", "string",
                   template=r"TXN\w{16}")

        .withColumn("alert_type", "string",
                   values=["unusual_amount", "velocity", "location", "merchant_risk"],
                   weights=[30, 25, 25, 20],
                   random=True)

        .withColumn("severity", "string",
                   values=["low", "medium", "high", "critical"],
                   weights=[20, 35, 30, 15],
                   random=True)

        .withColumn("investigation_status", "string",
                   values=["open", "investigating", "resolved", "false_positive"],
                   weights=[25, 30, 30, 15],
                   random=True)

        .withColumn("created_at", "timestamp",
                   expr="current_timestamp()")
    )

    fraud_df = fraud_spec.build()
    fraud_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.fraud_alerts")
    tables["fraud_alerts"] = {"row_count": num_fraud_alerts, "industry": "financial"}

    return tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏥 Healthcare Table Creation

# COMMAND ----------

def create_healthcare_tables(
    spark: SparkSession,
    schema: str,
    multiplier: float
) -> Dict[str, Dict]:
    """Create healthcare sample tables using dbldatagen."""
    import dbldatagen as dg
    from dbldatagen import fakerText, FakerTextFactory

    tables = {}

    num_patients = int(500_000 * multiplier)
    num_appointments = int(5_000_000 * multiplier)
    num_prescriptions = int(10_000_000 * multiplier)

    patient_partitions = max(4, min(32, num_patients // 15000))
    appointment_partitions = max(8, min(64, num_appointments // 50000))
    prescription_partitions = max(8, min(64, num_prescriptions // 100000))

    FakerUK = FakerTextFactory(locale=['en_GB'])

    # 1. Patients table (with PHI/PII data)
    print(f"  Creating patients table ({num_patients:,} rows)...")

    patient_spec = (
        dg.DataGenerator(spark, name="patients", rows=num_patients,
                        partitions=patient_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("patient_id", "long", minValue=1000000, uniqueValues=num_patients)

        .withColumn("nhs_number", "string",
                   template=r"\d{3}-\d{3}-\d{4}",
                   baseColumn="patient_id")

        .withColumn("name", "string",
                   text=fakerText("name"),
                   baseColumn="patient_id")

        .withColumn("date_of_birth", "date",
                   begin="1940-01-01", end="2020-12-31",
                   random=True,
                   baseColumn="patient_id")

        .withColumn("postcode", "string",
                   text=FakerUK("postcode"))

        .withColumn("condition_category", "string",
                   values=["Cardiovascular", "Respiratory", "Diabetes", "Cancer",
                          "Mental Health", "None"],
                   weights=[25, 20, 20, 15, 15, 5],
                   random=True)
    )

    patients_df = patient_spec.build()
    patients_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.patients")
    tables["patients"] = {"row_count": num_patients, "industry": "healthcare"}

    # 2. Appointments table (partitioned by date)
    print(f"  Creating appointments table ({num_appointments:,} rows)...")

    appointment_spec = (
        dg.DataGenerator(spark, name="appointments", rows=num_appointments,
                        partitions=appointment_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("appointment_id", "long", minValue=1, uniqueValues=num_appointments)

        .withColumn("patient_id", "long",
                   minValue=1000000, maxValue=1000000 + num_patients - 1,
                   random=True)
        .withColumn("doctor_id", "long",
                   minValue=1, maxValue=1000,
                   random=True)

        .withColumn("appointment_date", "date",
                   begin="2024-01-01", end="2024-12-31",
                   random=True)

        .withColumn("department", "string",
                   values=["Emergency", "Cardiology", "Orthopedics", "Pediatrics",
                          "Oncology", "Neurology", "Radiology", "Surgery",
                          "Primary Care", "Mental Health"],
                   weights=[15, 12, 10, 10, 8, 8, 8, 8, 15, 6],
                   random=True)

        .withColumn("duration_mins", "int",
                   minValue=15, maxValue=120,
                   distribution=dg.distributions.Normal(mean=45, stddev=15),
                   random=True)

        .withColumn("cost_gbp", "decimal(8,2)",
                   minValue=50, maxValue=2000,
                   distribution=dg.distributions.Gamma(shape=2.0, scale=2.0),
                   random=True)
    )

    appointments_df = appointment_spec.build()
    appointments_df.write.mode("overwrite").format("delta") \
        .partitionBy("appointment_date") \
        .saveAsTable(f"{schema}.appointments")
    tables["appointments"] = {"row_count": num_appointments, "industry": "healthcare", "partitioned": True}

    # 3. Prescriptions table (partitioned by date)
    print(f"  Creating prescriptions table ({num_prescriptions:,} rows)...")

    prescription_spec = (
        dg.DataGenerator(spark, name="prescriptions", rows=num_prescriptions,
                        partitions=prescription_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("prescription_id", "long", minValue=1, uniqueValues=num_prescriptions)

        .withColumn("patient_id", "long",
                   minValue=1000000, maxValue=1000000 + num_patients - 1,
                   random=True)

        .withColumn("medication_name", "string",
                   values=["Aspirin", "Ibuprofen", "Paracetamol", "Amoxicillin",
                          "Metformin", "Lisinopril", "Atorvastatin", "Omeprazole",
                          "Simvastatin", "Levothyroxine"],
                   weights=[15, 12, 15, 10, 8, 8, 10, 10, 7, 5],
                   random=True)

        .withColumn("dosage", "string",
                   values=["10mg", "20mg", "50mg", "100mg", "250mg", "500mg"],
                   weights=[10, 20, 25, 25, 15, 5],
                   random=True)

        .withColumn("prescribed_date", "date",
                   begin="2024-01-01", end="2024-12-31",
                   random=True)
    )

    prescriptions_df = prescription_spec.build()
    prescriptions_df.write.mode("overwrite").format("delta") \
        .partitionBy("prescribed_date") \
        .saveAsTable(f"{schema}.prescriptions")
    tables["prescriptions"] = {"row_count": num_prescriptions, "industry": "healthcare", "partitioned": True}

    return tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚡ Energy/Utilities Table Creation

# COMMAND ----------

def create_energy_tables(
    spark: SparkSession,
    schema: str,
    multiplier: float
) -> Dict[str, Dict]:
    """Create energy/utilities sample tables using dbldatagen."""
    import dbldatagen as dg
    from dbldatagen import fakerText, FakerTextFactory

    tables = {}

    num_customers = int(2_000_000 * multiplier)
    num_readings = int(50_000_000 * multiplier)
    num_outages = int(10_000 * multiplier)

    customer_partitions = max(4, min(32, num_customers // 50000))
    reading_partitions = max(16, min(256, num_readings // 100000))
    outage_partitions = max(2, min(8, num_outages // 1000))

    FakerUK = FakerTextFactory(locale=['en_GB'])

    # 1. Energy customers table
    print(f"  Creating energy_customers table ({num_customers:,} rows)...")

    energy_customer_spec = (
        dg.DataGenerator(spark, name="energy_customers", rows=num_customers,
                        partitions=customer_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("customer_id", "long", minValue=1000000, uniqueValues=num_customers)

        .withColumn("account_number", "string",
                   template=r"EGY\d{10}",
                   baseColumn="customer_id")

        .withColumn("postcode", "string",
                   text=FakerUK("postcode"))

        .withColumn("tariff_type", "string",
                   values=["Standard", "Economy 7", "Fixed Rate", "Variable", "Green Energy"],
                   weights=[30, 25, 20, 15, 10],
                   random=True)

        .withColumn("vulnerability_flag", "boolean",
                   values=[True, False],
                   weights=[10, 90],
                   random=True)
    )

    energy_customers_df = energy_customer_spec.build()
    energy_customers_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.energy_customers")
    tables["energy_customers"] = {"row_count": num_customers, "industry": "energy"}

    # 2. Smart meter readings table (large-scale time series)
    print(f"  Creating smart_meter_readings table ({num_readings:,} rows)...")

    reading_spec = (
        dg.DataGenerator(spark, name="smart_meter_readings", rows=num_readings,
                        partitions=reading_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("reading_id", "string",
                   template=r"RD\w{14}")

        .withColumn("meter_id", "string",
                   template=r"M\d{10}")

        .withColumn("customer_id", "long",
                   minValue=1000000, maxValue=1000000 + num_customers - 1,
                   random=True)

        .withColumn("reading_datetime", "timestamp",
                   begin="2024-01-01 00:00:00", end="2024-12-31 23:59:59",
                   interval="30 minutes",
                   random=True)

        .withColumn("kwh_consumed", "decimal(8,3)",
                   minValue=0.1, maxValue=50.0,
                   distribution=dg.distributions.Normal(mean=3.5, stddev=2.0),
                   random=True)

        .withColumn("tariff_rate", "decimal(6,4)",
                   values=[0.15, 0.20, 0.28],
                   weights=[40, 40, 20],
                   random=True)
    )

    readings_df = reading_spec.build()
    readings_df.write.mode("overwrite").format("delta") \
        .partitionBy(expr("date(reading_datetime)")) \
        .saveAsTable(f"{schema}.smart_meter_readings")
    tables["smart_meter_readings"] = {"row_count": num_readings, "industry": "energy", "partitioned": True}

    # 3. Outages table
    print(f"  Creating outages table ({num_outages:,} rows)...")

    outage_spec = (
        dg.DataGenerator(spark, name="outages", rows=num_outages,
                        partitions=outage_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("outage_id", "long", minValue=1, uniqueValues=num_outages)

        .withColumn("region", "string",
                   values=["North", "South", "East", "West", "Central"],
                   weights=[20, 20, 20, 20, 20],
                   random=True)

        .withColumn("start_time", "timestamp",
                   begin="2024-01-01 00:00:00", end="2024-12-31 23:59:59",
                   random=True)

        .withColumn("duration_mins", "int",
                   minValue=5, maxValue=7200,
                   distribution=dg.distributions.Exponential(rate=0.01),
                   random=True)

        .withColumn("end_time", "timestamp",
                   expr="start_time + INTERVAL duration_mins MINUTES",
                   baseColumn=["start_time", "duration_mins"])

        .withColumn("customers_affected", "int",
                   minValue=100, maxValue=50000,
                   distribution=dg.distributions.Gamma(shape=2.0, scale=1.5),
                   random=True)
    )

    outages_df = outage_spec.build()
    outages_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.outages")
    tables["outages"] = {"row_count": num_outages, "industry": "energy"}

    return tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏢 General Enterprise Table Creation

# COMMAND ----------

def create_enterprise_tables(
    spark: SparkSession,
    schema: str,
    multiplier: float
) -> Dict[str, Dict]:
    """Create general enterprise sample tables using dbldatagen."""
    import dbldatagen as dg
    from dbldatagen import fakerText

    tables = {}

    num_users = int(5_000_000 * multiplier)
    num_events = int(100_000_000 * multiplier)
    num_products = int(100_000 * multiplier)

    user_partitions = max(4, min(64, num_users // 50000))
    event_partitions = max(16, min(256, num_events // 200000))
    product_partitions = max(2, min(16, num_products // 5000))

    # 1. Users table
    print(f"  Creating users table ({num_users:,} rows)...")

    user_spec = (
        dg.DataGenerator(spark, name="users", rows=num_users,
                        partitions=user_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("user_id", "long", minValue=1, uniqueValues=num_users)

        .withColumn("email", "string",
                   text=fakerText("ascii_company_email"),
                   baseColumn="user_id")

        .withColumn("registration_date", "date",
                   begin="2020-01-01", end="2024-12-31",
                   random=True)

        .withColumn("country", "string",
                   values=["US", "GB", "CA", "DE", "FR", "AU", "JP", "BR"],
                   weights=[35, 20, 10, 10, 10, 5, 5, 5],
                   random=True)

        .withColumn("subscription_tier", "string",
                   values=["free", "basic", "premium", "enterprise"],
                   weights=[60, 25, 12, 3],
                   random=True)
    )

    users_df = user_spec.build()
    users_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.users")
    tables["users"] = {"row_count": num_users, "industry": "enterprise"}

    # 2. Events table (massive scale, partitioned)
    print(f"  Creating events table ({num_events:,} rows)...")

    event_spec = (
        dg.DataGenerator(spark, name="events", rows=num_events,
                        partitions=event_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("event_id", "string",
                   template=r"EVT\w{16}")

        .withColumn("user_id", "long",
                   minValue=1, maxValue=num_users,
                   random=True)

        .withColumn("event_type", "string",
                   values=["page_view", "click", "purchase", "signup", "logout"],
                   weights=[50, 30, 10, 5, 5],
                   random=True)

        .withColumn("timestamp", "timestamp",
                   begin="2024-01-01 00:00:00", end="2024-12-31 23:59:59",
                   interval="1 second",
                   random=True)

        .withColumn("session_id", "string",
                   template=r"SES\w{12}")

        .withColumn("properties", "string",
                   values=[
                       '{"page": "/home", "duration": 45}',
                       '{"page": "/products", "duration": 120}',
                       '{"page": "/checkout", "duration": 180}',
                       '{"page": "/account", "duration": 60}'
                   ],
                   random=True)
    )

    events_df = event_spec.build()
    events_df.write.mode("overwrite").format("delta") \
        .partitionBy(expr("date(timestamp)")) \
        .saveAsTable(f"{schema}.events")
    tables["events"] = {"row_count": num_events, "industry": "enterprise", "partitioned": True}

    # 3. Products table
    print(f"  Creating products table ({num_products:,} rows)...")

    product_spec = (
        dg.DataGenerator(spark, name="products", rows=num_products,
                        partitions=product_partitions,
                        randomSeedMethod='hash_fieldname')

        .withColumn("product_id", "long", minValue=1, uniqueValues=num_products)

        .withColumn("name", "string",
                   template=r"Product \w{8}")

        .withColumn("category", "string",
                   values=["Electronics", "Clothing", "Books", "Home", "Sports", "Toys"],
                   weights=[25, 20, 15, 20, 10, 10],
                   random=True)

        .withColumn("price", "decimal(10,2)",
                   minValue=10, maxValue=5000,
                   distribution=dg.distributions.Gamma(shape=2.0, scale=1.5),
                   random=True)

        .withColumn("stock_level", "int",
                   minValue=0, maxValue=10000,
                   distribution=dg.distributions.Normal(mean=500, stddev=300),
                   random=True)
    )

    products_df = product_spec.build()
    products_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.products")
    tables["products"] = {"row_count": num_products, "industry": "enterprise"}

    return tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏷️ Unity Catalog Tags

# COMMAND ----------

def apply_compliance_tags(
    spark: SparkSession,
    schema: str,
    tables: Dict[str, Dict]
) -> None:
    """Apply Unity Catalog compliance tags to tables and columns."""

    if "customers" in tables:
        print("  Tagging financial services tables...")
        spark.sql(f"""
            ALTER TABLE {schema}.customers SET TAGS (
                'sensitivity' = 'high',
                'fca_regulated' = 'true',
                'gdpr_relevant' = 'true',
                'retention_years' = '7',
                'consumer_duty_scope' = 'true'
            )
        """)
        spark.sql(f"""
            ALTER TABLE {schema}.customers
            ALTER COLUMN account_number SET TAGS ('pii' = 'true', 'financial_data' = 'true')
        """)
        spark.sql(f"""
            ALTER TABLE {schema}.customers
            ALTER COLUMN email SET TAGS ('pii' = 'true')
        """)

        spark.sql(f"""
            ALTER TABLE {schema}.transactions SET TAGS (
                'sensitivity' = 'high',
                'fca_regulated' = 'true',
                'retention_years' = '7'
            )
        """)

    if "patients" in tables:
        print("  Tagging healthcare tables...")
        spark.sql(f"""
            ALTER TABLE {schema}.patients SET TAGS (
                'sensitivity' = 'critical',
                'hipaa_phi' = 'true',
                'nhs_data_toolkit' = 'applicable',
                'special_category_data' = 'health'
            )
        """)
        spark.sql(f"""
            ALTER TABLE {schema}.patients
            ALTER COLUMN nhs_number SET TAGS ('pii' = 'true', 'patient_identifier' = 'true')
        """)
        spark.sql(f"""
            ALTER TABLE {schema}.patients
            ALTER COLUMN name SET TAGS ('pii' = 'true')
        """)
        spark.sql(f"""
            ALTER TABLE {schema}.patients
            ALTER COLUMN date_of_birth SET TAGS ('pii' = 'true')
        """)

    if "energy_customers" in tables:
        print("  Tagging energy/utilities tables...")
        spark.sql(f"""
            ALTER TABLE {schema}.smart_meter_readings SET TAGS (
                'sensitivity' = 'critical',
                'behavioral_data' = 'true',
                'ofgem_regulated' = 'true',
                'gdpr_profiling_risk' = 'high',
                'data_minimization_required' = 'true'
            )
        """)

    print("  ✅ Tags applied successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Sample Queries Table

# COMMAND ----------

def create_sample_queries_table(
    spark: SparkSession,
    schema: str,
    tables: Dict[str, Dict]
) -> None:
    """Create table with sample good and bad queries."""

    from sample_queries import get_all_sample_queries

    queries = get_all_sample_queries()

    query_data = []
    for query_info in queries:
        query_data.append({
            "query_id": query_info["id"],
            "industry": query_info["industry"],
            "quality": query_info["quality"],
            "anti_pattern_type": query_info.get("anti_pattern_type", None),
            "query_text": query_info["query"],
            "description": query_info["description"],
            "expected_violations": ",".join(query_info.get("expected_violations", [])),
            "expected_cost_impact": query_info.get("expected_cost_impact", "unknown"),
            "recommended_fix": query_info.get("recommended_fix", None),
            "persona": query_info.get("persona", "data_engineer")
        })

    queries_df = spark.createDataFrame(query_data)
    queries_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.sample_queries")

    print(f"  ✅ Created sample_queries table with {len(query_data)} queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Metadata Tables

# COMMAND ----------

def create_metadata_tables(spark: SparkSession, schema: str) -> None:
    """Create metadata tracking tables for demo management."""

    demo_runs_schema = StructType([
        StructField("run_id", StringType(), False),
        StructField("setup_timestamp", TimestampType(), False),
        StructField("scale", StringType(), False),
        StructField("tables_created", IntegerType(), False),
        StructField("unity_catalog_used", BooleanType(), False),
        StructField("tags_applied", BooleanType(), False)
    ])

    demo_runs_df = spark.createDataFrame([], demo_runs_schema)
    demo_runs_df.write.mode("overwrite").format("delta").saveAsTable(f"{schema}.demo_runs")

    print(f"  ✅ Created metadata tracking tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Validation Function

# COMMAND ----------

def validate_demo_environment(
    catalog: str = DEMO_CATALOG,
    schema: str = DEMO_SCHEMA,
    use_unity_catalog: bool = True
) -> Dict[str, any]:
    """Validate that demo environment was created correctly."""
    spark = SparkSession.builder.getOrCreate()

    full_name = f"{catalog}.{schema}" if use_unity_catalog else schema

    print(f"\n🔍 Validating demo environment: {full_name}\n")

    validation_results = {
        "schema_exists": False,
        "tables": {},
        "errors": []
    }

    try:
        schemas = [row.databaseName for row in spark.sql("SHOW SCHEMAS").collect()]
        if schema in schemas or full_name in schemas:
            validation_results["schema_exists"] = True
            print(f"✅ Schema exists: {full_name}")
        else:
            validation_results["errors"].append(f"Schema not found: {full_name}")
            print(f"❌ Schema not found: {full_name}")
            return validation_results

        expected_tables = [
            "customers", "transactions", "fraud_alerts",
            "patients", "appointments", "prescriptions",
            "energy_customers", "smart_meter_readings", "outages",
            "users", "events", "products",
            "sample_queries", "demo_runs"
        ]

        for table_name in expected_tables:
            try:
                full_table_name = f"{full_name}.{table_name}"
                count = spark.table(full_table_name).count()
                validation_results["tables"][table_name] = {
                    "exists": True,
                    "row_count": count
                }
                print(f"✅ {table_name}: {count:,} rows")
            except Exception as e:
                validation_results["tables"][table_name] = {
                    "exists": False,
                    "error": str(e)
                }
                validation_results["errors"].append(f"Table {table_name}: {str(e)}")
                print(f"❌ {table_name}: Error - {str(e)}")

        print(f"\n{'='*80}")
        if len(validation_results["errors"]) == 0:
            print("✅ All validation checks passed")
        else:
            print(f"⚠️  {len(validation_results['errors'])} validation error(s) found")

    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")
        print(f"❌ Validation error: {str(e)}")

    return validation_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🗑️ Teardown Function

# COMMAND ----------

def teardown_demo_environment(
    catalog: str = DEMO_CATALOG,
    schema: str = DEMO_SCHEMA,
    use_unity_catalog: bool = True
) -> None:
    """Remove all demo objects from the workspace."""
    spark = SparkSession.builder.getOrCreate()

    full_name = f"{catalog}.{schema}" if use_unity_catalog else schema

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    Cloud CEO DBX Demo Teardown                           ║
╚══════════════════════════════════════════════════════════════════════════╝

Removing: {full_name}
""")

    response = input("Are you sure you want to delete all demo data? (yes/no): ")

    if response.lower() != "yes":
        print("❌ Teardown cancelled")
        return

    try:
        spark.sql(f"DROP SCHEMA IF EXISTS {full_name} CASCADE")
        print(f"✅ Removed schema: {full_name}")

        if use_unity_catalog:
            drop_catalog = input(f"Also drop catalog '{catalog}'? (yes/no): ")
            if drop_catalog.lower() == "yes":
                spark.sql(f"DROP CATALOG IF EXISTS {catalog} CASCADE")
                print(f"✅ Removed catalog: {catalog}")

        print("\n✅ Demo environment teardown complete")

    except Exception as e:
        print(f"❌ Error during teardown: {str(e)}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Main Setup Function

# COMMAND ----------

def setup_demo_environment(
    scale: str = "medium",
    use_unity_catalog: bool = True,
    apply_tags: bool = True,
    create_queries: bool = True,
    catalog: str = DEMO_CATALOG,
    schema: str = DEMO_SCHEMA
) -> Dict[str, any]:
    """Setup complete demo environment with sample data using dbldatagen."""

    deps_ok, error_msg = validate_dependencies()
    if not deps_ok:
        print(error_msg)
        raise RuntimeError("Missing required dependencies: dbldatagen and faker")

    spark = SparkSession.builder.getOrCreate()

    import dbldatagen as dg
    from dbldatagen import fakerText

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║           Cloud CEO DBX Demo Environment Setup (dbldatagen)              ║
╚══════════════════════════════════════════════════════════════════════════╝

Scale: {scale.upper()} ({SCALE_FACTORS[scale]['description']})
Estimated time: {SCALE_FACTORS[scale]['setup_time']}
Unity Catalog: {use_unity_catalog}
Apply tags: {apply_tags}
Data generator: dbldatagen v{dg.__version__} with Faker integration

Starting setup...
""")

    start_time = datetime.now()

    configure_spark_for_datagen(spark)

    if use_unity_catalog:
        full_name = f"{catalog}.{schema}"
        print(f"📦 Creating Unity Catalog objects: {full_name}")
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        spark.sql(f"USE CATALOG {catalog}")
    else:
        full_name = schema
        print(f"📦 Creating Hive schema: {schema}")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    spark.sql(f"USE {schema}")

    multiplier = SCALE_FACTORS[scale]["multiplier"]

    tables_created = {}

    print("\n🏦 Creating Financial Services tables...")
    financial_tables = create_financial_tables(spark, full_name, multiplier)
    tables_created.update(financial_tables)

    print("\n🏥 Creating Healthcare tables...")
    healthcare_tables = create_healthcare_tables(spark, full_name, multiplier)
    tables_created.update(healthcare_tables)

    print("\n⚡ Creating Energy/Utilities tables...")
    energy_tables = create_energy_tables(spark, full_name, multiplier)
    tables_created.update(energy_tables)

    print("\n🏢 Creating General Enterprise tables...")
    enterprise_tables = create_enterprise_tables(spark, full_name, multiplier)
    tables_created.update(enterprise_tables)

    if apply_tags and use_unity_catalog:
        print("\n🏷️  Applying Unity Catalog tags...")
        apply_compliance_tags(spark, full_name, tables_created)

    if create_queries:
        print("\n📝 Generating sample queries...")
        create_sample_queries_table(spark, full_name, tables_created)

    print("\n📊 Creating metadata tables...")
    create_metadata_tables(spark, full_name)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    summary = {
        "catalog": catalog if use_unity_catalog else None,
        "schema": schema,
        "full_name": full_name,
        "scale": scale,
        "duration_seconds": duration,
        "tables_created": len(tables_created),
        "table_names": list(tables_created.keys()),
        "unity_catalog": use_unity_catalog,
        "tags_applied": apply_tags,
        "queries_created": create_queries,
        "data_generator": "dbldatagen + Faker"
    }

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                           Setup Complete!                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

✅ Created {len(tables_created)} tables in {duration:.1f} seconds
📍 Location: {full_name}
🚀 Data generator: dbldatagen with Faker integration

Tables created:
""")

    for table_name, info in tables_created.items():
        print(f"  • {table_name}: {info['row_count']:,} rows")

    print(f"""
Next steps:
1. Explore sample queries: SELECT * FROM {full_name}.sample_queries LIMIT 10
2. View bad query examples: SELECT * FROM {full_name}.sample_queries WHERE quality = 'bad'
3. Run demo notebooks in /demo directory
4. Clean up when done: teardown_demo_environment()
""")

    return summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Interactive Configuration
# MAGIC
# MAGIC Use widgets to configure your demo environment interactively.

# COMMAND ----------

# Create configuration widgets
dbutils.widgets.dropdown("scale", "medium", ["small", "medium", "large"], "Data Scale")
dbutils.widgets.dropdown("catalog", "demo", ["demo", "main", "dev"], "Catalog")
dbutils.widgets.text("schema", "cloud_ceo_demo", "Schema")
dbutils.widgets.dropdown("apply_tags", "true", ["true", "false"], "Apply UC Tags")

# COMMAND ----------

# Get widget values
scale = dbutils.widgets.get("scale")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
apply_tags = dbutils.widgets.get("apply_tags") == "true"

print(f"""
Configuration:
  Scale: {scale}
  Catalog: {catalog}
  Schema: {schema}
  Apply Unity Catalog tags: {apply_tags}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Execute Setup
# MAGIC
# MAGIC Run this cell to create all demo tables with the configuration above.

# COMMAND ----------

displayHTML("""
<div style='padding:20px; background-color:#FFF4E6; border-left:4px solid #FF9800; margin-bottom:20px;'>
<strong>⚠️ Important:</strong> This will create demo tables in your workspace.
<p>Review the configuration above before proceeding.</p>
</div>
""")

# Run setup
summary = setup_demo_environment(
    scale=scale,
    use_unity_catalog=True,
    apply_tags=apply_tags,
    create_queries=True,
    catalog=catalog,
    schema=schema
)

# COMMAND ----------

# Display success message
displayHTML(f"""
<div style='padding:20px; background-color:#DFF0D8; border:1px solid #D6E9C6; border-radius:4px;'>
<h2 style='color:#3C763D; margin-top:0;'>✅ Setup Complete!</h2>
<p><strong>Tables Created:</strong> {summary['tables_created']}</p>
<p><strong>Location:</strong> {summary['full_name']}</p>
<p><strong>Setup Time:</strong> {summary['duration_seconds']:.1f} seconds</p>
<p><strong>Data Generator:</strong> {summary['data_generator']}</p>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Validate Setup
# MAGIC
# MAGIC Run this cell to verify all tables were created successfully.

# COMMAND ----------

validation_results = validate_demo_environment(catalog=catalog, schema=schema, use_unity_catalog=True)

if len(validation_results["errors"]) == 0:
    displayHTML("""
    <div style='padding:20px; background-color:#DFF0D8; border:1px solid #D6E9C6; border-radius:4px;'>
    <h3 style='color:#3C763D; margin-top:0;'>✅ All Validation Checks Passed</h3>
    <p>Your demo environment is ready to use!</p>
    </div>
    """)
else:
    displayHTML(f"""
    <div style='padding:20px; background-color:#F2DEDE; border:1px solid #EBCCD1; border-radius:4px;'>
    <h3 style='color:#A94442; margin-top:0;'>❌ Validation Errors Found</h3>
    <p>{len(validation_results['errors'])} error(s) detected. Check the output above.</p>
    </div>
    """)
