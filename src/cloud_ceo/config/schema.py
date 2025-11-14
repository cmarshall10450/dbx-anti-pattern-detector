"""Minimal YAML-only configuration schema for Cloud CEO.

Simple Pydantic models for configuration validation.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo

logger = structlog.get_logger(__name__)

# Security: Allowed module prefixes for detector classes
ALLOWED_DETECTOR_MODULES = ["cloud_ceo.detectors."]

# Available industry modes for compliance-aware analysis
IndustryMode = Literal["uk_finance", "uk_retail", "uk_healthcare", "pharma_biotech", "uk_energy"]


class SecurityConfig(BaseModel):
    """Security configuration for detector loading and execution.

    Controls security features including static analysis, execution timeouts,
    and namespace trust policies for custom detectors.
    """

    trusted_namespaces: list[str] = Field(
        default_factory=list,
        description="Namespaces explicitly trusted by organization"
    )
    allow_cloud_ceo_extension: bool = Field(
        default=False,
        description="Allow extending cloud_ceo namespace for internal development"
    )
    enable_static_analysis: bool = Field(
        default=True,
        description="Run static analysis on custom detectors"
    )
    max_detector_timeout: int = Field(
        default=5,
        gt=0,
        le=60,
        description="Maximum detector execution time in seconds"
    )
    block_high_risk_detectors: bool = Field(
        default=True,
        description="Block detectors with high or critical risk levels"
    )
    enable_prompt_sanitization: bool = Field(
        default=True,
        description="Sanitize LLM inputs to prevent prompt injection"
    )
    audit_injection_attempts: bool = Field(
        default=False,
        description="Log detected injection patterns to audit trail (non-blocking)"
    )
    audit_table: Optional[str] = Field(
        default=None,
        description="Databricks table for injection audit logs (e.g., 'security.prompt_injection_audit')"
    )
    audit_file: Optional[str] = Field(
        default=None,
        description="JSONL file path for local injection audit logs"
    )
    audit_rate_limit_per_minute: int = Field(
        default=100,
        gt=0,
        le=1000,
        description="Maximum audit log entries per minute to prevent spam"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with detailed error messages and stack traces"
    )
    max_plugin_file_size_mb: int = Field(
        default=10,
        gt=0,
        le=100,
        description="Maximum file size in MB for plugin and guideline files"
    )

    @field_validator("max_detector_timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate detector timeout is reasonable."""
        if v > 30:
            logger.warning(
                "high_detector_timeout",
                timeout=v,
                recommendation="Consider values <= 30 seconds for better performance"
            )
        return v

    @field_validator("max_plugin_file_size_mb")
    @classmethod
    def validate_file_size_limit(cls, v: int) -> int:
        """Validate file size limit is reasonable."""
        if v > 50:
            logger.warning(
                "high_file_size_limit",
                limit_mb=v,
                recommendation="Large file size limits may impact performance"
            )
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration.

    API key is loaded from OPENAI_API_KEY environment variable.
    """

    enabled: bool = Field(
        default=False,
        description="Enable LLM-based detection features"
    )
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, bedrock)"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier (e.g., gpt-4o-mini, claude-3-5-sonnet)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens for completion"
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        gt=0,
        le=300,
        description="Maximum LLM API requests per minute"
    )
    rate_limit_cost_per_hour: float = Field(
        default=10.0,
        gt=0.0,
        le=100.0,
        description="Maximum LLM cost per hour (USD)"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported."""
        valid_providers = {"openai", "anthropic", "bedrock"}
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Invalid provider '{v}'. Must be one of: {', '.join(valid_providers)}"
            )
        return v.lower()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str, info: ValidationInfo) -> str:
        """Validate model identifier and provide suggestions for typos."""
        known_models = {
            "openai": {
                "gpt-4o": "High-performance GPT-4 Turbo",
                "gpt-4o-mini": "Faster, cost-effective GPT-4",
                "gpt-4-turbo": "Latest GPT-4 Turbo",
                "gpt-4": "Standard GPT-4",
                "gpt-3.5-turbo": "Fast and efficient GPT-3.5",
            },
            "anthropic": {
                "claude-3-5-sonnet-20241022": "Latest Claude 3.5 Sonnet",
                "claude-3-opus-20240229": "Most capable Claude 3 model",
                "claude-3-sonnet-20240229": "Balanced Claude 3 model",
                "claude-3-haiku-20240307": "Fast Claude 3 model",
            },
            "bedrock": {
                "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet on Bedrock",
                "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku on Bedrock",
            }
        }

        provider = info.data.get("provider", "openai")
        provider_models = known_models.get(provider, {})

        if provider_models and v not in provider_models:
            logger.warning(
                "unknown_llm_model",
                model=v,
                provider=provider,
                known_models=list(provider_models.keys()),
                suggestion=f"Common {provider} models: {', '.join(list(provider_models.keys())[:3])}"
            )

        return v

    @field_validator("timeout")
    @classmethod
    def validate_llm_timeout(cls, v: int) -> int:
        """Validate LLM timeout is reasonable."""
        if v > 120:
            logger.warning(
                "high_llm_timeout",
                timeout=v,
                recommendation="LLM timeouts > 120s may indicate configuration issues"
            )
        if v < 10:
            logger.warning(
                "low_llm_timeout",
                timeout=v,
                recommendation="Very low timeouts may cause frequent failures"
            )
        return v

    @field_validator("rate_limit_requests_per_minute")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate rate limit is reasonable."""
        if v > 200:
            logger.warning(
                "high_rate_limit",
                requests_per_minute=v,
                recommendation="High rate limits may exceed API quotas"
            )
        return v

    @field_validator("rate_limit_cost_per_hour")
    @classmethod
    def validate_cost_limit(cls, v: float) -> float:
        """Validate cost limit is reasonable."""
        if v > 50.0:
            logger.warning(
                "high_cost_limit",
                cost_per_hour=v,
                recommendation="High cost limits may lead to unexpected expenses"
            )
        return v

    def get_api_key(self) -> str:
        """Get API key from environment variable.

        Returns:
            API key string

        Raises:
            ValueError: If API key not found in environment
        """
        env_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
        }.get(self.provider, "OPENAI_API_KEY")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable '{env_var}'. "
                f"Please set this environment variable to use LLM features."
            )
        return api_key


class CloudCEOConfig(BaseModel):
    """Main Cloud CEO configuration."""

    security: SecurityConfig = Field(
        default_factory=SecurityConfig,
        description="Security configuration for plugin system"
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    # IMPORTANT: allowed_detector_modules must come BEFORE detectors
    # so that the validator has access to it when validating detectors
    allowed_detector_modules: list[str] = Field(
        default_factory=lambda: ["cloud_ceo.detectors."],
        description="Module prefixes allowed for detector loading (security control)"
    )
    detectors: list[str] = Field(
        default_factory=lambda: [
            "cloud_ceo.detectors.spark.CartesianJoinDetector",
            "cloud_ceo.detectors.spark.NonSargablePredicateDetector",
            "cloud_ceo.detectors.spark.SelectStarDetector",
        ],
        description="List of detector class names to enable"
    )
    custom_detector_paths: list[str] = Field(
        default_factory=list,
        description="Filesystem paths to search for custom detectors"
    )
    output_format: str = Field(
        default="json",
        description="Output format (json, text, markdown)"
    )
    industry: Optional[IndustryMode] = Field(
        default=None,
        description=(
            "Industry-specific compliance mode for analysis. Options: "
            "uk_finance (FCA/GDPR), uk_retail (GDPR/Consumer Rights), "
            "uk_healthcare (NHS/Caldicott), pharma_biotech (GxP/21 CFR Part 11), "
            "uk_energy (Ofgem/Smart Meters)"
        )
    )

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format is supported."""
        valid_formats = {"json", "text", "markdown"}
        if v.lower() not in valid_formats:
            raise ValueError(
                f"Invalid output_format '{v}'. Must be one of: {', '.join(valid_formats)}"
            )
        return v.lower()

    @field_validator("custom_detector_paths")
    @classmethod
    def validate_custom_paths(cls, v: list[str]) -> list[str]:
        """Basic validation of custom detector paths.

        Deep validation (path traversal, symlinks, etc.) happens in load_custom_detector_paths().
        """
        # Just ensure it's a list of strings - actual security validation happens at load time
        return v

    @field_validator("detectors")
    @classmethod
    def validate_detectors_not_empty(cls, v: list[str]) -> list[str]:
        """Validate at least one detector is enabled."""
        if not v:
            raise ValueError("At least one detector must be enabled")
        return v

    @model_validator(mode='after')
    def validate_detector_modules(self) -> 'CloudCEOConfig':
        """Validate detectors are from allowed modules and check namespace policies.

        This runs after all fields are set, so we have access to allowed_detector_modules
        and security config.
        """
        # Security: Validate each detector is from allowed modules
        for detector_path in self.detectors:
            # Parse module path
            if ":" in detector_path:
                module_path = detector_path.split(":", 1)[0]
            else:
                parts = detector_path.split(".")
                module_path = ".".join(parts[:-1])

            # Check against allowlist
            # Normalize module_path with trailing dot for prefix matching
            # This ensures "myorg.detectors" matches both "myorg.detectors" and "myorg.detectors."
            module_path_normalized = module_path + "."

            # Normalize allowed prefixes to have trailing dots
            allowed_normalized = [
                allowed if allowed.endswith(".") else allowed + "."
                for allowed in self.allowed_detector_modules
            ]

            if not any(module_path_normalized.startswith(allowed) for allowed in allowed_normalized):
                raise ValueError(
                    f"Detector '{detector_path}' is not from an allowed module. "
                    f"Add module prefix to 'allowed_detector_modules' in config. "
                    f"Allowed prefixes: {self.allowed_detector_modules}"
                )

            # SECURITY: Prevent custom detectors from shadowing built-in modules
            # Allow cloud_ceo namespace extension only if explicitly configured
            if module_path.startswith("cloud_ceo."):
                # Built-in detectors are always allowed
                if module_path.startswith("cloud_ceo.detectors."):
                    continue

                # Check if cloud_ceo extension is allowed
                if self.security.allow_cloud_ceo_extension:
                    # Verify namespace is in trusted list
                    # Use normalized comparison to handle trailing dots consistently
                    trusted_normalized = [
                        trusted if trusted.endswith(".") else trusted + "."
                        for trusted in self.security.trusted_namespaces
                    ]
                    is_trusted = any(
                        module_path_normalized.startswith(trusted)
                        for trusted in trusted_normalized
                    )
                    if is_trusted:
                        logger.warning(
                            "allowing_cloud_ceo_extension",
                            detector=detector_path,
                            namespace=module_path
                        )
                        continue

                raise ValueError(
                    f"Security: Custom detectors cannot use 'cloud_ceo.' namespace. "
                    f"Detector '{detector_path}' appears to be attempting namespace collision. "
                    f"To allow internal development, set security.allow_cloud_ceo_extension=true "
                    f"and add namespace to security.trusted_namespaces. "
                    f"Otherwise use a different module prefix (e.g., 'company.detectors', 'myorg.rules')."
                )

        return self

    def load_custom_detector_paths(self) -> None:
        """Add custom paths to Python path for detector discovery.

        This method should be called before loading detectors to ensure
        custom detector modules can be imported. Paths are expanded and
        validated before being added to sys.path.

        Security Controls:
        1. Only paths explicitly configured by the user are added
        2. Path traversal attacks (..) are blocked
        3. Paths must be absolute after resolution
        4. Symlink loops are detected and rejected
        5. Paths are appended (not prepended) to prevent shadowing built-in modules
        6. Paths containing 'cloud_ceo' directories are rejected to prevent namespace collision
        7. All operations are logged for audit purposes

        Raises:
            ValueError: If path contains path traversal, cloud_ceo directory, or is invalid
        """
        for path_str in self.custom_detector_paths:
            # SECURITY FIX H-1: Prevent path traversal attacks
            if ".." in path_str:
                raise ValueError(
                    f"Security: Path traversal detected in '{path_str}'. "
                    f"Relative paths with '..' are not allowed for security reasons."
                )

            # Expand and resolve path
            try:
                path = Path(path_str).expanduser().resolve(strict=False)
            except RuntimeError as e:
                raise ValueError(
                    f"Security: Invalid path or symlink loop detected in '{path_str}': {e}"
                )

            # SECURITY FIX H-1: Ensure path is absolute after resolution
            if not path.is_absolute():
                raise ValueError(
                    f"Security: Path must resolve to absolute path: '{path_str}' -> '{path}'"
                )

            # SECURITY FIX H-1: Check for ".." in resolved path as well
            if ".." in str(path):
                raise ValueError(
                    f"Security: Path traversal detected in resolved path '{path}' from '{path_str}'"
                )

            if not path.exists():
                logger.warning(
                    "custom_detector_path_not_found",
                    path=str(path),
                    original=path_str
                )
                continue

            if not path.is_dir():
                logger.warning(
                    "custom_detector_path_not_directory",
                    path=str(path)
                )
                continue

            # SECURITY: Check for potential namespace collision attacks
            # Reject paths that contain cloud_ceo subdirectories
            cloud_ceo_subdir = path / "cloud_ceo"
            if cloud_ceo_subdir.exists():
                raise ValueError(
                    f"Security: Custom detector path '{path}' contains 'cloud_ceo' directory. "
                    f"This could be an attempt to shadow built-in detectors. "
                    f"Custom detectors must use a different namespace (e.g., 'company', 'myorg')."
                )

            path_str_resolved = str(path)
            if path_str_resolved not in sys.path:
                # SECURITY: Append instead of insert to prevent shadowing built-in modules
                sys.path.append(path_str_resolved)
                logger.info(
                    "custom_detector_path_added",
                    path=path_str_resolved,
                    position="append",
                    sys_path_length=len(sys.path)
                )

    def model_dump_yaml(self) -> dict[str, Any]:
        """Export configuration as YAML-compatible dict."""
        config = {
            "security": {
                "trusted_namespaces": self.security.trusted_namespaces,
                "allow_cloud_ceo_extension": self.security.allow_cloud_ceo_extension,
                "enable_static_analysis": self.security.enable_static_analysis,
                "max_detector_timeout": self.security.max_detector_timeout,
                "block_high_risk_detectors": self.security.block_high_risk_detectors,
                "enable_prompt_sanitization": self.security.enable_prompt_sanitization,
                "audit_injection_attempts": self.security.audit_injection_attempts,
                "audit_table": self.security.audit_table,
                "audit_file": self.security.audit_file,
                "audit_rate_limit_per_minute": self.security.audit_rate_limit_per_minute,
                "debug_mode": self.security.debug_mode,
                "max_plugin_file_size_mb": self.security.max_plugin_file_size_mb,
            },
            "llm": {
                "enabled": self.llm.enabled,
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
            },
            "detectors": self.detectors,
            "output_format": self.output_format,
        }

        # Include custom detector configuration if specified
        if self.custom_detector_paths:
            config["custom_detector_paths"] = self.custom_detector_paths

        # Include allowed modules if different from default
        if self.allowed_detector_modules != ["cloud_ceo.detectors."]:
            config["allowed_detector_modules"] = self.allowed_detector_modules

        # Only include industry if specified
        if self.industry:
            config["industry"] = self.industry

        return config
