# Cloud CEO

**SQL Anti-Pattern Detection and Query Optimization for Databricks**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-alpha-yellow)](https://github.com/your-org/cloud-ceo-dbx)

---

## What is Cloud CEO?

Cloud CEO (Cloud Efficiency Optimisation) is a comprehensive SQL analysis platform for Databricks that combines static code analysis with AI-powered recommendations to identify performance bottlenecks, cost inefficiencies, and compliance risks in your SQL workloads.

The platform provides two integrated interfaces:

- **Python SDK** for programmatic analysis and CI/CD integration
- **Databricks Integration** for workspace-wide query monitoring

## Key Features

- Detects SQL anti-patterns through AST-based parsing including non-sargable predicates, Cartesian joins, and inefficient operations
- Analyzes post-execution metrics from Databricks system tables to calculate cost impact and resource utilization
- Generates context-aware optimization suggestions using LLM analysis (OpenAI, Anthropic, AWS Bedrock)
- Reads Unity Catalog tags to provide compliance-aware recommendations for regulated industries
- Monitors workspace-wide query patterns to identify worst-performing queries and users
- Integrates with CI/CD pipelines for pre-execution query validation
- Exports results in JSON, Markdown, or human-readable text formats

## Quick Start

### Installation

```bash
# Basic installation
pip install cloud-ceo-dbx

# With Databricks integration
pip install cloud-ceo-dbx[databricks]
```

### Python SDK

```python
from cloud_ceo import analyze

# Analyze a query before execution
result = analyze("""
    SELECT * FROM customers
    WHERE YEAR(created_at) = 2024
""")

print(result.summary())
```

Output:
```
======================================================================
Cloud CEO Analysis - Pre Execution
Query ID: a1b2c3d4e5f6
======================================================================

1. SELECT_STAR [HIGH]
   Using SELECT * can cause performance issues and data leaks
   Line: 2
   Fix: Explicitly list required columns

2. NON_SARGABLE_PREDICATE [MEDIUM]
   YEAR() function prevents index usage
   Line: 3
   Fix: Rewrite as: created_at >= '2024-01-01' AND created_at < '2025-01-01'

----------------------------------------------------------------------
Total: 2 violation(s)
======================================================================
```

## Use Cases

### Pre-Execution Validation

Validate queries in your CI/CD pipeline before they reach production:

```python
from cloud_ceo import analyze

def validate_query(sql: str) -> bool:
    """Validate query before execution."""
    result = analyze(sql)

    if result.get_critical_violations():
        print("Critical issues found - query rejected")
        for violation in result.get_critical_violations():
            print(f"  {violation.message}")
        return False

    return True

# In your deployment pipeline
if validate_query(production_query):
    deploy_to_production(production_query)
```

### Post-Execution Analysis

Debug slow queries with performance metrics:

```python
result = analyze(
    query_text,
    execution_metrics={
        "duration_ms": 120000,
        "bytes_scanned": 50_000_000_000,
        "rows_scanned": 10_000_000,
        "peak_memory_mb": 15000,
        "cost_usd": 15.75
    },
    cluster_config={
        "node_type": "i3.4xlarge",
        "executor_count": 10,
        "executor_memory": "122GB"
    },
    enable_llm=True
)

print(result.summary())
if result.llm_recommendations:
    for rec in result.llm_recommendations:
        print(f"Recommendation: {rec['description']}")
        print(f"Expected improvement: {rec['expected_improvement']}")
```

### Workspace Administration

Monitor query quality across your organization:

```python
from cloud_ceo import analyze_workspace

# Daily report of expensive queries
report = analyze_workspace(
    days=1,
    min_cost_usd=50,
    enable_llm=True
)

# Send alerts for high-cost violations
if report.total_cost_usd > 1000:
    send_alert(f"High daily cost: ${report.total_cost_usd:.2f}")

# Export to dashboard
export_metrics({
    "total_cost": report.total_cost_usd,
    "violation_rate": report.queries_with_violations / report.total_queries,
    "worst_query": report.worst_queries[0].databricks_query_id
})
```

## Data Governance and Tagging

Cloud CEO integrates with Unity Catalog tags to provide compliance-aware recommendations. Tagging is completely optional, but well-structured tags significantly improve the quality and specificity of LLM analysis.

Unity Catalog tags enable Cloud CEO to provide regulatory compliance guidance based on GDPR, FCA, HIPAA, and other frameworks. The LLM uses tags to assess risk based on data sensitivity classifications, target optimization recommendations for high-value or regulated data, and apply industry-specific knowledge for finance, healthcare, and pharma sectors.

### Enhanced Analysis Example

Without tags, Cloud CEO identifies the anti-pattern:
```
SELECT * loads unnecessary columns
```

With Unity Catalog tags, the analysis includes compliance context:
```sql
-- Table tagged as: sensitivity=high, gdpr_relevant=true, retention_years=7
SELECT * loads unnecessary columns

Enhanced Analysis (using tags):
1. Columns marked pii=true should be explicitly selected (minimize exposure)
2. GDPR requires documented justification for PII access
3. Consider if all selected columns are needed for the business purpose
4. Ensure 7-year retention policy is applied to query results

Recommendation: Explicitly select only required columns and document
business justification for PII access.
```

### Tagging Quick Start

Start with basic classification:

```sql
-- Minimal but effective
ALTER TABLE customers SET TAGS ('contains_pii' = 'true');
ALTER TABLE transactions SET TAGS ('financial_data' = 'true', 'retention_years' = '7');
```

### Industry-Specific Tagging

UK Financial Services:
```sql
ALTER TABLE customers SET TAGS (
    'sensitivity' = 'high',
    'fca_regulated' = 'true',
    'gdpr_relevant' = 'true',
    'retention_years' = '7',
    'consumer_duty_scope' = 'true'
);

ALTER TABLE customers ALTER COLUMN account_number SET TAGS (
    'pii' = 'true',
    'financial_data' = 'true'
);
```

Healthcare (NHS):
```sql
ALTER TABLE patients SET TAGS (
    'sensitivity' = 'critical',
    'nhs_data_toolkit' = 'applicable',
    'special_category_data' = 'health',
    'caldicott_principles' = 'apply'
);

ALTER TABLE patients ALTER COLUMN nhs_number SET TAGS (
    'pii' = 'true',
    'patient_identifier' = 'true',
    'de_identification_required' = 'secondary_use'
);
```

Pharma/Biotech:
```sql
ALTER TABLE clinical_trials SET TAGS (
    'sensitivity' = 'critical',
    'gxp_controlled' = 'true',
    '21_cfr_part_11' = 'true',
    'ich_e6_gcp' = 'true',
    'data_integrity' = 'alcoa_plus'
);
```

UK Energy Utilities:
```sql
ALTER TABLE customers SET TAGS (
    'sensitivity' = 'high',
    'ofgem_regulated' = 'true',
    'gdpr_relevant' = 'true',
    'retention_years' = '7',
    'consumer_duty_scope' = 'true'
);

ALTER TABLE smart_meter_readings SET TAGS (
    'sensitivity' = 'critical',
    'behavioral_data' = 'true',
    'comcop_protected' = 'true',
    'gdpr_profiling_risk' = 'high',
    'data_minimization_required' = 'true'
);

ALTER TABLE customers ALTER COLUMN account_number SET TAGS (
    'pii' = 'true',
    'financial_data' = 'true'
);

ALTER TABLE smart_meter_readings ALTER COLUMN kwh_consumed SET TAGS (
    'consumption_data' = 'true',
    'behavioral_indicator' = 'true',
    'consent_required' = 'data_sharing'
);

ALTER TABLE priority_services_register SET TAGS (
    'sensitivity' = 'critical',
    'vulnerability_data' = 'true',
    'ofgem_psr' = 'true',
    'access_justification_required' = 'true'
);
```

Cloud CEO's LLM automatically detects and interprets these tags to provide compliance-aware recommendations. For comprehensive tagging guidance, see [Data Governance Tagging Guide](docs/DATA_GOVERNANCE_TAGGING_GUIDE.md).

## Custom Detector Plugins

Cloud CEO supports custom detector plugins, enabling organizations to create anti-pattern detectors for compliance requirements (GDPR, HIPAA, SOX), organization-specific SQL standards, and industry-specific optimizations without modifying core code.

### Creating a Custom Detector

```python
# ~/my_detectors/company/detectors.py
from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.rule_engine.types import RuleSeverity

class GDPRDetector(RuleDetector):
    """Detect potential GDPR violations in SQL queries."""

    def __init__(self):
        super().__init__(rule_id="GDPR_001", severity=RuleSeverity.CRITICAL)

    def _detect_violations(self, ast, context):
        violations = []
        # Your detection logic using sqlglot AST
        return violations
```

### Configuring Custom Detectors

```yaml
# config.yaml
custom_detector_paths:
  - ~/my_detectors

allowed_detector_modules:
  - cloud_ceo.detectors.  # Built-in detectors
  - company.detectors.    # Your custom detectors

detectors:
  - cloud_ceo.detectors.spark.SelectStarDetector  # Built-in
  - company.detectors.GDPRDetector                # Custom
```

### Use Cases for Custom Detectors

- **Compliance Rules**: GDPR Article 17 (Right to Erasure), HIPAA PHI access controls, SOX segregation of duties
- **Organization Standards**: Table/column naming conventions, approved join patterns, required audit logging
- **Technology-Specific**: Delta Lake optimization patterns, Databricks-specific best practices
- **Temporary Detectors**: Migration helpers to detect deprecated table usage

For complete documentation including validation, testing, and examples, see:
- **Quick Start**: [30-Minute Tutorial](PLUGIN_QUICK_START.md)
- **Developer Guide**: [Plugin Development Guide](docs/PLUGIN_DEVELOPMENT_GUIDE.md)
- **API Reference**: [Detector API Reference](docs/DETECTOR_API_REFERENCE.md)

## Configuration

Cloud CEO uses a YAML configuration file for LLM settings:

```yaml
# config.yaml
llm:
  enabled: true
  provider: openai  # openai, anthropic, or bedrock
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 4096
  timeout: 30

# Optional: Industry-specific compliance mode
industry: uk_finance  # uk_finance, uk_retail, uk_healthcare, pharma_biotech, uk_energy
```

API keys are loaded from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `AWS_ACCESS_KEY_ID`).

## API Reference

### Core Functions

#### `analyze(query, **options) → AnalysisResult`

Unified analysis function for pre-execution and post-execution scenarios.

Parameters:
- `query` (str | Path): SQL query string or path to .sql file
- `execution_metrics` (dict, optional): Performance data including duration_ms, bytes_scanned, rows_scanned, peak_memory_mb
- `cluster_config` (dict, optional): Cluster configuration including node_type, executor_count, executor_memory
- `enable_llm` (bool): Enable AI-powered recommendations
- `databricks_query_id` (str, optional): Original Databricks query ID

Returns: `AnalysisResult` with violations, metrics, and recommendations

#### `analyze_workspace(**options) → WorkspaceReport`

Analyze Databricks workspace for worst performing queries.

Parameters:
- `days` (int): Number of days to analyze (default: 7)
- `min_cost_usd` (float, optional): Minimum query cost filter
- `users` (list[str], optional): Filter by specific users
- `enable_llm` (bool): Enable LLM analysis
- `top_n` (int): Number of top violators (default: 50)

Returns: `WorkspaceReport` with aggregated statistics and worst queries

### Output Formats

Result objects can be formatted in multiple ways:

```python
# Human-readable text
print(result.summary())

# JSON for APIs and automation
print(result.as_json())

# Markdown for documentation
print(result.as_markdown())
```

## Documentation

- [SDK Documentation](SDK_README.md) - Complete API reference and usage patterns
- [Data Governance Tagging Guide](docs/DATA_GOVERNANCE_TAGGING_GUIDE.md) - Comprehensive tagging strategies
- [Databricks Integration](QUICK_START_DATABRICKS.md) - Notebook usage and workspace monitoring
- [Custom Guidelines](docs/CUSTOM_GUIDELINES_GUIDE.md) - Extending with organization-specific rules
- **[Custom Detector Plugins](docs/PLUGIN_DEVELOPMENT_GUIDE.md)** - Create organization-specific anti-pattern detectors
- **[Security Architecture](SECURITY.md)** - Threat model, security controls, and deployment guidelines

## Requirements

Python 3.11 or 3.12 is required. Core dependencies include `sqlglot` for SQL parsing, `pydantic` for data validation, and `langchain` for LLM orchestration. The optional `pyspark` dependency enables Databricks integration.

For Databricks features, Unity Catalog tags are optional but recommended. Workspace analysis requires system tables access. Query execution requires a SQL warehouse or cluster.

## Context Compression

Cloud CEO compresses LLM prompts from 1,500-3,500 tokens to 800-1,500 tokens (47-57% reduction) using keyword and semantic filtering. All processing happens in-memory using BM25-style keyword scoring and optional sentence embeddings.

When industry mode is enabled (`industry: uk_finance`), the compressor filters regulations to relevant sections. Custom guidelines are ranked by similarity to the detected violation. Execution metrics are filtered to show only violation-relevant fields.

```python
from cloud_ceo import analyze

# Compression enabled by default
result = analyze(query_text, enable_llm=True, industry_mode="uk_finance")

# Disable compression to send full context
result = analyze(query_text, enable_llm=True, enable_compression=False)
```

For semantic similarity ranking, install the optional dependency:

```bash
pip install cloud-ceo-dbx[llm]
```

Without `sentence-transformers`, Cloud CEO uses keyword-based filtering with no functionality loss.

### Data Processing

All compression happens locally in-memory. No data is sent to external services. Techniques used:

| Component | Method |
|-----------|--------|
| Violation classification | Rule-based categorization |
| Relevance scoring | BM25 keyword matching |
| Semantic ranking | Sentence embeddings (optional) |
| Caching | In-memory LRU cache |

For regulated industries (FCA, NHS, Ofgem), all data stays within your infrastructure. Compression is deterministic and can be disabled via `enable_compression=False`.

## Performance

Cloud CEO is designed for production use with caching, lazy loading, and batch optimization. Repeated analysis of the same query returns instant cached results. Detectors load once and reuse. Workspace analysis processes queries in batches. Analysis continues gracefully if LLM services are unavailable.

```python
# First call - full analysis
result1 = analyze(query)  # ~100ms

# Second call - from cache
result2 = analyze(query)  # ~1ms
```

## Contributing

Contributions are welcome. Please ensure:

1. Code follows existing patterns and conventions
2. Tests cover new functionality
3. Documentation is updated
4. Commit messages are clear and descriptive

## Support

- **Issues:** Report bugs and request features via GitHub Issues
- **Documentation:** See the [docs/](docs/) directory
- **Examples:** Find sample code in [examples/](examples/)

## License

Cloud CEO is released under the MIT License. See [LICENSE](LICENSE) for details.

---

**Cloud CEO** is maintained by Optima Partners and used in production to optimize Databricks workloads across financial services, energy, and other regulated industries.
