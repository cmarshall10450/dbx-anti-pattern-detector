"""Rule loading system with pluggable source architecture.

This module provides an abstract RuleSource interface and implementations for
loading rules from various sources (built-in YAML, custom YAML, Delta tables).
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml
from pydantic import ValidationError

from cloud_ceo.rule_engine.schema import RuleMetadata, RuleSchema

if TYPE_CHECKING:
    from cloud_ceo.rule_engine.precedence import RulePrecedenceManager

logger = logging.getLogger(__name__)
structured_logger = structlog.get_logger(__name__)


class RuleSource(ABC):
    """Abstract base class for rule sources.

    Rule sources provide a pluggable architecture for loading rules from
    different storage backends (YAML files, Delta tables, etc.).
    """

    @abstractmethod
    def load(self) -> list[RuleSchema]:
        """Load and validate rules from this source.

        Returns:
            List of validated RuleSchema objects

        Raises:
            Exception: If critical errors prevent rule loading
        """
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if this rule source is enabled.

        Returns:
            True if source is enabled, False otherwise
        """
        pass


class BuiltInRuleSource(RuleSource):
    """Loads built-in rules from YAML files in the built_in directory.

    Built-in rules are curated anti-pattern rules shipped with Cloud CEO.
    They provide immediate value without requiring custom rule authoring.

    Args:
        enabled: Whether built-in rules are enabled (from config)
        rules_dir: Path to built_in rules directory (defaults to package location)
    """

    def __init__(self, enabled: bool = True, rules_dir: Path | None = None):
        """Initialize built-in rule source.

        Args:
            enabled: Whether built-in rules are enabled
            rules_dir: Path to built_in rules directory (auto-detected if None)
        """
        self._enabled = enabled

        if rules_dir is None:
            from cloud_ceo.rule_engine import BUILT_IN_RULES_DIR
            self._rules_dir = BUILT_IN_RULES_DIR
        else:
            self._rules_dir = rules_dir

    def is_enabled(self) -> bool:
        """Check if built-in rules are enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled

    def load(self) -> list[RuleSchema]:
        """Load and validate all built-in rules from YAML files.

        Scans the built_in directory for YAML files and loads rules from each.
        Invalid rules are logged as errors but do not prevent loading of other rules.

        Returns:
            List of validated RuleSchema objects

        Raises:
            FileNotFoundError: If built_in directory doesn't exist
        """
        if not self._enabled:
            logger.info("Built-in rules are disabled via configuration")
            return []

        if not self._rules_dir.exists():
            raise FileNotFoundError(
                f"Built-in rules directory not found: {self._rules_dir}. "
                "This is a critical error - the package may be corrupted."
            )

        logger.info(f"Loading built-in rules from: {self._rules_dir}")

        rules: list[RuleSchema] = []
        yaml_files = list(self._rules_dir.glob("*.yaml")) + list(self._rules_dir.glob("*.yml"))

        if not yaml_files:
            logger.warning(
                f"No YAML files found in built-in rules directory: {self._rules_dir}"
            )
            return rules

        for yaml_file in yaml_files:
            try:
                file_rules = self._load_yaml_file(yaml_file)
                rules.extend(file_rules)
                logger.info(
                    f"Loaded {len(file_rules)} rules from {yaml_file.name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load built-in rule file {yaml_file.name}: {e}",
                    exc_info=True
                )
                continue

        logger.info(f"Successfully loaded {len(rules)} built-in rules total")
        return rules

    def _load_yaml_file(self, yaml_file: Path) -> list[RuleSchema]:
        """Load and validate rules from a single YAML file.

        Args:
            yaml_file: Path to YAML file

        Returns:
            List of validated RuleSchema objects from this file

        Raises:
            yaml.YAMLError: If YAML parsing fails
            ValueError: If file format is invalid
        """
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Invalid YAML syntax in {yaml_file.name}: {e}"
            ) from e

        if data is None:
            logger.warning(f"Empty YAML file: {yaml_file.name}")
            return []

        if not isinstance(data, list):
            raise ValueError(
                f"Expected list of rules in {yaml_file.name}, got {type(data).__name__}"
            )

        rules: list[RuleSchema] = []
        for idx, rule_dict in enumerate(data, start=1):
            if not isinstance(rule_dict, dict):
                logger.error(
                    f"Rule #{idx} in {yaml_file.name} is not a dict, got {type(rule_dict).__name__}. Skipping."
                )
                continue

            try:
                rule = self._validate_rule(rule_dict, yaml_file.name, idx)
                rules.append(rule)
            except ValidationError as e:
                logger.error(
                    f"Rule #{idx} in {yaml_file.name} failed validation: {e}",
                    exc_info=True
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error validating rule #{idx} in {yaml_file.name}: {e}",
                    exc_info=True
                )
                continue

        return rules

    def _validate_rule(self, rule_dict: dict[str, Any], filename: str, index: int) -> RuleSchema:
        """Validate a rule dictionary against RuleSchema.

        Args:
            rule_dict: Rule data as dictionary
            filename: Source filename (for error messages)
            index: Rule index in file (for error messages)

        Returns:
            Validated RuleSchema object

        Raises:
            ValidationError: If validation fails
        """
        try:
            return RuleSchema(**rule_dict)
        except ValidationError as e:
            error_details = self._format_validation_errors(e)
            raise ValidationError.from_exception_data(
                title=f"Rule validation failed in {filename} (rule #{index})",
                line_errors=e.errors()
            ) from e

    def _format_validation_errors(self, error: ValidationError) -> str:
        """Format validation errors for logging.

        Args:
            error: ValidationError from Pydantic

        Returns:
            Formatted error message
        """
        errors = []
        for err in error.errors():
            field = ".".join(str(loc) for loc in err['loc'])
            msg = err['msg']
            errors.append(f"  - {field}: {msg}")
        return "\n".join(errors)


class YAMLRuleSource(RuleSource):
    """Load RuleSchema instances from YAML files.

    YAMLRuleSource supports loading rules from user-specified YAML files,
    including glob pattern expansion for loading multiple files.

    Args:
        yaml_paths: List of YAML file paths or glob patterns
        enabled: Whether YAML rules are enabled (from config)
    """

    def __init__(self, yaml_paths: list[str | Path], enabled: bool = True):
        """Initialize YAML rule source.

        Args:
            yaml_paths: List of YAML file paths or glob patterns (e.g., "rules/**/*.yaml")
            enabled: Whether YAML rules are enabled
        """
        self._yaml_paths = yaml_paths
        self._enabled = enabled

    def is_enabled(self) -> bool:
        """Check if YAML rules are enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled

    def load(self) -> list[RuleSchema]:
        """Load all YAML rules with validation.

        Resolves glob patterns, loads YAML files, validates rules against RuleSchema,
        and returns list of validated rules. Invalid rules are logged but don't
        prevent loading of other rules (graceful degradation).

        Returns:
            List of validated RuleSchema objects

        Raises:
            Exception: Only if critical errors prevent any rule loading
        """
        if not self._enabled:
            structured_logger.info("YAML rules are disabled via configuration")
            return []

        resolved_paths = self._resolve_paths()

        if not resolved_paths:
            structured_logger.warning(
                "No YAML rule files found",
                patterns=self._yaml_paths
            )
            return []

        rules: list[RuleSchema] = []
        validation_errors = 0

        for yaml_path in resolved_paths:
            try:
                file_rules = self._load_file(yaml_path)
                rules.extend(file_rules)
                structured_logger.debug(
                    "Loaded rules from YAML file",
                    file=str(yaml_path),
                    count=len(file_rules)
                )
            except Exception as e:
                validation_errors += 1
                structured_logger.error(
                    "YAML load failed",
                    file=str(yaml_path),
                    error=str(e)
                )
                continue

        disabled_count = sum(1 for r in rules if not r.enabled)

        structured_logger.info(
            "YAML rules loaded",
            files=len(resolved_paths),
            rules=len(rules),
            disabled=disabled_count,
            validation_errors=validation_errors
        )

        return rules

    def _resolve_paths(self) -> list[Path]:
        """Resolve glob patterns to sorted file paths.

        Expands glob patterns (e.g., "rules/**/*.yaml") to concrete file paths,
        handles ~ expansion, and returns sorted list for deterministic order.

        Returns:
            Sorted list of resolved Path objects
        """
        resolved: list[Path] = []

        for pattern in self._yaml_paths:
            path = Path(pattern).expanduser()

            if path.exists() and path.is_file():
                resolved.append(path)
            elif any(c in str(path) for c in ['*', '?', '[']):
                matched = self._expand_glob_pattern(path)
                if not matched:
                    structured_logger.warning(
                        "Glob matched no files",
                        pattern=str(pattern)
                    )
                resolved.extend(matched)
            else:
                structured_logger.warning(
                    "File not found",
                    path=str(path)
                )

        return sorted(set(resolved))

    def _expand_glob_pattern(self, pattern_path: Path) -> list[Path]:
        """Expand a glob pattern to matching files.

        Args:
            pattern_path: Path with glob pattern

        Returns:
            List of matching file paths
        """
        matched: list[Path] = []

        if pattern_path.parent.exists():
            parent = pattern_path.parent
            pattern = pattern_path.name
            matched = sorted(parent.glob(pattern))
        else:
            parts = pattern_path.parts
            for i, part in enumerate(parts):
                if any(c in part for c in ['*', '?', '[']):
                    base = Path(*parts[:i]) if i > 0 else Path.cwd()
                    if base.exists():
                        remaining = str(Path(*parts[i:]))
                        matched = sorted(base.glob(remaining))
                    break

        return [p for p in matched if p.is_file()]

    def _load_file(self, yaml_path: Path) -> list[RuleSchema]:
        """Load and validate single YAML file.

        Args:
            yaml_path: Path to YAML file

        Returns:
            List of validated RuleSchema objects from this file

        Raises:
            ValueError: If YAML syntax is invalid or file format is wrong
        """
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            line = 'unknown'
            if hasattr(e, 'problem_mark') and e.problem_mark:
                line = e.problem_mark.line + 1
            problem = e.problem if hasattr(e, 'problem') else str(e)
            raise ValueError(f"YAML syntax error at line {line}: {problem}")

        if data is None:
            structured_logger.warning(
                "Empty YAML file",
                file=str(yaml_path)
            )
            return []

        if not isinstance(data, list):
            raise ValueError(
                f"YAML must contain list of rules, got {type(data).__name__}"
            )

        rules: list[RuleSchema] = []
        for idx, rule_data in enumerate(data, start=1):
            if not isinstance(rule_data, dict):
                structured_logger.error(
                    "Rule is not a dictionary",
                    file=str(yaml_path),
                    index=idx,
                    type=type(rule_data).__name__
                )
                continue

            try:
                rule = RuleSchema(**rule_data)
                rules.append(rule)
            except ValidationError as e:
                structured_logger.error(
                    "Rule validation failed",
                    file=str(yaml_path),
                    index=idx,
                    rule_id=rule_data.get('rule_id', 'unknown'),
                    errors=[f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
                )
                continue

        return rules


DELTA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS {table_path} (
    rule_id STRING NOT NULL,
    name STRING NOT NULL,
    category STRING NOT NULL CHECK (category IN ('universal', 'context-dependent')),
    severity STRING NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    detection_method STRING NOT NULL CHECK (detection_method IN ('ast_analysis', 'pattern', 'llm', 'query_plan')),
    detector_class STRING NOT NULL CHECK (detector_class LIKE '%:%'),
    enabled BOOLEAN NOT NULL DEFAULT true,
    explanation STRING NOT NULL,
    dependencies ARRAY<STRING>,
    custom_thresholds STRING,
    custom_message STRING,
    metadata STRING,
    config_version STRING NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_by STRING,
    updated_by STRING,
    created_at TIMESTAMP DEFAULT current_timestamp(),
    updated_at TIMESTAMP DEFAULT current_timestamp(),
    CONSTRAINT pk_rule_version PRIMARY KEY (rule_id, config_version)
)
USING DELTA
CLUSTER BY (rule_id, is_active, config_version)
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.enablePredictiveOptimization' = 'true',
    'cloud_ceo.schema_version' = '1.0'
);
""".strip()


class DeltaRuleSource(RuleSource):
    """Load RuleSchema instances from Delta Lake tables.

    DeltaRuleSource provides production-ready integration with Delta Lake for
    versioned rule management. It supports automatic table creation and
    comprehensive validation.

    REQUIRES: Databricks environment with active SparkSession.
    Cloud CEO is designed to run exclusively in Databricks (workspace or
    Databricks Connect).

    The Delta table maintains rule versioning with config_version, allowing
    temporal queries and rollback capabilities. Only active rules with the
    latest version are loaded.

    Args:
        table_path: Path to Delta table (e.g., "catalog.schema.rules" or "/path/to/delta")
        enabled: Whether Delta rules are enabled (from config)

    Features:
        - Auto-create table with production-ready schema
        - Window function for latest version selection
        - Comprehensive validation with security checks
        - Structured logging for observability
    """

    def __init__(self, table_path: str, enabled: bool = True):
        """Initialize Delta rule source.

        Args:
            table_path: Path to Delta table in Unity Catalog or filesystem
            enabled: Whether Delta rules are enabled

        Raises:
            RuntimeError: If PySpark is not available or no active SparkSession
        """
        self._table_path = table_path
        self._enabled = enabled
        self._execution_context: str | None = None
        self._spark = self._get_spark_session()

    def is_enabled(self) -> bool:
        """Check if Delta rules are enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled

    def load(self) -> list[RuleSchema]:
        """Load and validate all active rules from Delta table.

        Loads only active rules with the latest config_version using window
        functions. Invalid rules are logged but don't prevent loading of
        other rules (graceful degradation).

        Returns:
            List of validated RuleSchema objects

        Behavior:
            - Returns empty list if not enabled
            - Returns empty list if Spark unavailable (local/test context)
            - Creates table automatically if it doesn't exist
            - Loads only is_active=true rules with latest config_version
            - Continues loading even if individual rules fail validation
        """
        if not self._enabled:
            structured_logger.info(
                "Delta rules are disabled via configuration",
                table_path=self._table_path
            )
            return []

        execution_context = self._detect_execution_context()
        structured_logger.info(
            "Loading Delta rules",
            table_path=self._table_path,
            execution_context=execution_context
        )

        if self._spark is None:
            structured_logger.warning(
                "SparkSession unavailable, skipping Delta rules",
                table_path=self._table_path,
                execution_context=execution_context
            )
            return []

        try:
            self._create_table_if_not_exists()
            rules = self._load_latest_rules()

            disabled_count = sum(1 for r in rules if not r.enabled)
            structured_logger.info(
                "Delta rules loaded successfully",
                table_path=self._table_path,
                rules=len(rules),
                disabled=disabled_count,
                execution_context=execution_context
            )

            return rules

        except Exception as e:
            structured_logger.error(
                "Failed to load Delta rules",
                table_path=self._table_path,
                error=str(e),
                execution_context=execution_context
            )
            return []

    def _detect_execution_context(self) -> str:
        """Detect current execution environment.

        Determines whether code is running in a Databricks notebook,
        Databricks job, or local environment. This helps with graceful
        degradation and appropriate logging.

        Returns:
            One of 'notebook', 'job', or 'local'
        """
        if self._execution_context is not None:
            return self._execution_context

        try:
            import IPython
            shell = IPython.get_ipython()
            if shell is not None and 'DbShell' in str(type(shell)):
                self._execution_context = 'notebook'
            else:
                self._execution_context = 'local'
        except ImportError:
            try:
                import os
                if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
                    self._execution_context = 'job'
                else:
                    self._execution_context = 'local'
            except Exception:
                self._execution_context = 'local'

        return self._execution_context

    def _get_spark_session(self) -> Any | None:
        """Get active Spark session.

        Returns the active SparkSession in the current environment. For testing,
        a SparkSession can be created with local master.

        Returns:
            SparkSession if available, None if not available

        Note:
            In production (Databricks), SparkSession is always available.
            Returns None only in local/test environments for graceful degradation.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            structured_logger.warning(
                "PySpark not installed, Delta rules unavailable",
                table_path=self._table_path
            )
            return None

        try:
            spark = SparkSession.getActiveSession()
            if spark is None:
                structured_logger.info(
                    "No active Spark session",
                    table_path=self._table_path
                )
            return spark

        except Exception as e:
            structured_logger.warning(
                "Failed to get Spark session",
                table_path=self._table_path,
                error=str(e)
            )
            return None

    def _create_table_if_not_exists(self) -> None:
        """Create Delta table if it doesn't exist.

        Uses predefined DDL schema with production-ready settings including
        change data feed, predictive optimization, and clustering.

        Behavior:
            - Creates table only if it doesn't exist
            - Uses DELTA_TABLE_DDL constant for schema
            - Logs info if table created
            - Logs info if table already exists
            - Does not raise exceptions on failure
        """
        try:
            ddl = DELTA_TABLE_DDL.format(table_path=self._table_path)
            self._spark.sql(ddl)
            structured_logger.info(
                "Delta table verified or created",
                table_path=self._table_path
            )
        except Exception as e:
            if "already exists" in str(e).lower() or "table_or_view_already_exists" in str(e).lower():
                structured_logger.debug(
                    "Delta table already exists",
                    table_path=self._table_path
                )
            else:
                error_msg = str(e)
                if "table_or_view_not_found" in error_msg.lower():
                    structured_logger.info(
                        "Delta table does not exist and could not be created",
                        table_path=self._table_path,
                        error=error_msg
                    )
                else:
                    structured_logger.warning(
                        "Failed to create Delta table",
                        table_path=self._table_path,
                        error=error_msg
                    )

    def _load_latest_rules(self) -> list[RuleSchema]:
        """Load latest active rules using PySpark window functions.

        Loads only is_active=true rules and selects the latest version for
        each rule_id using window function over config_version and updated_at.

        Returns:
            List of validated RuleSchema objects

        Behavior:
            - Loads only is_active=true rules
            - Uses window function to get latest version
            - Orders by config_version DESC, updated_at DESC
            - Continues loading even if individual rules fail validation
            - Returns empty list if table doesn't exist
        """
        try:
            from pyspark.sql import Window
            from pyspark.sql.functions import col, row_number

            df = self._spark.table(self._table_path)

            if df.count() == 0:
                structured_logger.info(
                    "Delta table is empty, no rules to load",
                    table_path=self._table_path
                )
                return []

            window_spec = Window.partitionBy("rule_id").orderBy(
                col("config_version").desc(),
                col("updated_at").desc()
            )

            latest_rules_df = df.filter(col("is_active") == True).withColumn(
                "row_num", row_number().over(window_spec)
            ).filter(col("row_num") == 1).drop("row_num")

            rules: list[RuleSchema] = []
            validation_errors = 0

            for row in latest_rules_df.collect():
                row_dict = row.asDict()
                try:
                    rule = self._dict_to_rule_schema(row_dict)
                    rules.append(rule)
                except ValidationError as e:
                    validation_errors += 1
                    structured_logger.error(
                        "Rule validation failed",
                        table_path=self._table_path,
                        rule_id=row_dict.get('rule_id', 'unknown'),
                        errors=[f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()]
                    )
                    continue
                except Exception as e:
                    validation_errors += 1
                    structured_logger.error(
                        "Failed to convert row to rule",
                        table_path=self._table_path,
                        rule_id=row_dict.get('rule_id', 'unknown'),
                        error=str(e)
                    )
                    continue

            if validation_errors > 0:
                structured_logger.warning(
                    "Some rules failed validation",
                    table_path=self._table_path,
                    validation_errors=validation_errors,
                    successful_rules=len(rules)
                )

            return rules

        except Exception as e:
            error_msg = str(e)
            if "table_or_view_not_found" in error_msg.lower() or "table or view not found" in error_msg.lower():
                structured_logger.info(
                    "Delta table does not exist",
                    table_path=self._table_path
                )
            else:
                structured_logger.error(
                    "Failed to load rules from Delta table",
                    table_path=self._table_path,
                    error=error_msg
                )
            return []

    def _dict_to_rule_schema(self, row_dict: dict[str, Any]) -> RuleSchema:
        """Convert Delta Lake row dictionary to RuleSchema with type conversion.

        Handles conversion of Delta Lake types to Pydantic model types,
        including JSON parsing for complex fields and timestamp conversion.

        Args:
            row_dict: Dictionary from Delta table row

        Returns:
            Validated RuleSchema instance

        Raises:
            ValidationError: If Pydantic validation fails
            ValueError: If required fields are missing or invalid

        Type Conversions:
            - custom_thresholds: JSON string → dict
            - metadata: JSON string → dict (merged with audit fields)
            - dependencies: Array<String> → list[str]
            - Timestamps: auto-converted by Pydantic
        """

        custom_thresholds = {}
        if row_dict.get('custom_thresholds'):
            try:
                custom_thresholds = json.loads(row_dict['custom_thresholds'])
                if not isinstance(custom_thresholds, dict):
                    structured_logger.warning(
                        "custom_thresholds is not a dict, using empty dict",
                        rule_id=row_dict.get('rule_id', 'unknown'),
                        type=type(custom_thresholds).__name__
                    )
                    custom_thresholds = {}
            except json.JSONDecodeError as e:
                structured_logger.warning(
                    "Failed to parse custom_thresholds JSON, using empty dict",
                    rule_id=row_dict.get('rule_id', 'unknown'),
                    error=str(e)
                )

        metadata_dict = {}
        if row_dict.get('metadata'):
            try:
                metadata_dict = json.loads(row_dict['metadata'])
                if not isinstance(metadata_dict, dict):
                    structured_logger.warning(
                        "metadata is not a dict, using empty dict",
                        rule_id=row_dict.get('rule_id', 'unknown'),
                        type=type(metadata_dict).__name__
                    )
                    metadata_dict = {}
            except json.JSONDecodeError as e:
                structured_logger.warning(
                    "Failed to parse metadata JSON, using empty dict",
                    rule_id=row_dict.get('rule_id', 'unknown'),
                    error=str(e)
                )

        metadata_dict.setdefault('created_by', row_dict.get('created_by'))
        metadata_dict.setdefault('created_at', row_dict.get('created_at'))
        metadata_dict.setdefault('updated_at', row_dict.get('updated_at'))
        metadata_dict.setdefault('version', row_dict.get('config_version', '1.0'))
        metadata_dict.setdefault('confidence_score', 0.8)

        if 'updated_by' in row_dict and row_dict['updated_by']:
            metadata_dict.setdefault('approved_by', row_dict['updated_by'])

        dependencies = row_dict.get('dependencies', [])
        if dependencies is None:
            dependencies = []

        detection_method_value = row_dict.get('detection_method', '')
        if detection_method_value.lower() == 'ast_analysis':
            detection_method_value = 'AST'

        detector_class = row_dict['detector_class']

        rule_data = {
            'rule_id': row_dict['rule_id'],
            'name': row_dict['name'],
            'category': row_dict['category'],
            'severity': row_dict['severity'],
            'detection_method': detection_method_value,
            'detector_class': detector_class,
            'enabled': row_dict.get('enabled', True),
            'explanation': row_dict['explanation'],
            'dependencies': dependencies,
            'custom_thresholds': custom_thresholds,
            'custom_message': row_dict.get('custom_message'),
            'metadata': metadata_dict,
        }

        return RuleSchema(**rule_data)


def initialize_rules(
    yaml_paths: list[str | Path] | None = None,
    delta_table: str | None = None,
    built_in_enabled: bool = True,
    yaml_enabled: bool = True,
    delta_enabled: bool = True,
) -> tuple[list[RuleSchema], "RulePrecedenceManager"]:
    """Initialize rules with precedence resolution.

    Orchestrates loading rules from multiple sources in precedence order
    and resolves conflicts using last-write-wins strategy.
    Precedence: built-in < YAML < Delta.

    This is the primary entry point for initializing the rule engine.
    Typically called once at application startup.

    Args:
        yaml_paths: List of YAML file paths or glob patterns (optional)
        delta_table: Delta table path (e.g., "catalog.schema.rules") (optional)
        built_in_enabled: Whether to load built-in rules (default: True)
        yaml_enabled: Whether to load YAML rules (default: True)
        delta_enabled: Whether to load Delta rules (default: True)

    Returns:
        Tuple of (rules, precedence_manager) where:
            - rules: Deduplicated list of RuleSchema objects
            - precedence_manager: RulePrecedenceManager instance for querying overrides

    Example:
        >>> from cloud_ceo.rule_engine import initialize_rules
        >>> rules, manager = initialize_rules(
        ...     yaml_paths=["~/rules/*.yaml"],
        ...     delta_table="main.rules.cloud_ceo_rules"
        ... )
        >>> print(f"Loaded {len(rules)} rules")
        >>> source = manager.get_effective_source("SPARK_001")
        >>> print(f"SPARK_001 from: {source}")

    See Also:
        - RulePrecedenceManager: For precedence resolution details
        - BuiltInRuleSource, YAMLRuleSource, DeltaRuleSource: Source implementations
    """
    from cloud_ceo.rule_engine.precedence import RulePrecedenceManager

    precedence_manager = RulePrecedenceManager()

    # Load built-in rules (lowest precedence)
    if built_in_enabled:
        structured_logger.info("Loading built-in rules")
        built_in_source = BuiltInRuleSource(enabled=True)
        if built_in_source.is_enabled():
            built_in_rules = built_in_source.load()
            precedence_manager.add_rules(built_in_rules, "built-in")
            structured_logger.info("Built-in rules loaded", count=len(built_in_rules))

    # Load YAML rules (medium precedence)
    if yaml_enabled and yaml_paths:
        structured_logger.info("Loading YAML rules", paths=yaml_paths)
        yaml_source = YAMLRuleSource(yaml_paths=yaml_paths, enabled=True)
        if yaml_source.is_enabled():
            yaml_rules = yaml_source.load()
            precedence_manager.add_rules(yaml_rules, "yaml")
            structured_logger.info("YAML rules loaded", count=len(yaml_rules))

    # Load Delta rules (highest precedence)
    if delta_enabled and delta_table:
        structured_logger.info("Loading Delta rules", table=delta_table)
        delta_source = DeltaRuleSource(table_path=delta_table, enabled=True)
        if delta_source.is_enabled():
            delta_rules = delta_source.load()
            precedence_manager.add_rules(delta_rules, "delta")
            structured_logger.info("Delta rules loaded", count=len(delta_rules))

    # Get final deduplicated rules
    all_rules = precedence_manager.get_all_rules()

    # Log summary statistics
    stats = precedence_manager.get_stats()
    structured_logger.info(
        "Rule initialization complete",
        total_rules=stats["total_rules"],
        overridden_rules=stats["overridden_rules"],
        sources_used=stats["sources_used"]
    )

    return all_rules, precedence_manager
