"""Custom exceptions for rule management operations.

This module defines exception classes for rule CRUD operations, providing
structured error handling with proper context for debugging and monitoring.
"""

import re
import traceback
from pathlib import Path
from typing import Optional


def sanitize_error_message(
    error: Exception,
    debug_mode: bool = False,
    include_type: bool = True
) -> str:
    """Sanitize error messages to prevent information disclosure.

    In production mode, removes stack traces, file paths, and internal details
    that could expose system architecture or sensitive information.
    In debug mode, preserves full error details for troubleshooting.

    Args:
        error: The exception to sanitize
        debug_mode: If True, show full error details. If False, sanitize for production.
        include_type: If True, include exception type in sanitized message

    Returns:
        Sanitized error message safe for user display

    Examples:
        >>> try:
        ...     raise FileNotFoundError("/home/user/secrets/config.yaml not found")
        ... except Exception as e:
        ...     sanitized = sanitize_error_message(e, debug_mode=False)
        >>> # sanitized = "Configuration file not found"
    """
    if debug_mode:
        # Include traceback if available
        tb = traceback.format_exc()
        # format_exc() returns 'NoneType: None\n' when not in exception handler
        if tb and 'NoneType' not in tb:
            return f"{type(error).__name__}: {str(error)}\n{tb}"
        else:
            # No active traceback, create a simple debug format
            return f"Traceback (most recent call last):\n  {type(error).__name__}: {str(error)}"

    error_str = str(error)
    error_type = type(error).__name__

    # Remove file paths (both Unix and Windows style)
    error_str = re.sub(r'(/[\w/.-]+|[A-Z]:\\[\w\\.-]+)', '[PATH]', error_str)

    # Remove line numbers and code locations
    error_str = re.sub(r'line \d+', 'line [N]', error_str, flags=re.IGNORECASE)
    error_str = re.sub(r'at 0x[0-9a-fA-F]+', 'at [ADDRESS]', error_str)

    # Remove internal module references
    error_str = re.sub(r'in module [\'"][\w.]+[\'"]', 'in module [MODULE]', error_str)
    error_str = re.sub(r'module [\'"][\w.]+[\'"] has no', 'module [MODULE] has no', error_str)
    error_str = re.sub(r'named [\'"][\w.]+[\'"]', 'named [MODULE]', error_str)

    # Remove stack trace information
    error_str = re.sub(r'File ".*?"', 'File "[REDACTED]"', error_str)

    # Remove API keys or tokens (if accidentally included)
    error_str = re.sub(r'(sk-|api[_-]?key[_-]?)[a-zA-Z0-9_-]{20,}', r'\1[REDACTED]', error_str, flags=re.IGNORECASE)

    # Provide user-friendly messages for common errors
    friendly_messages = {
        'FileNotFoundError': 'File or directory not found',
        'PermissionError': 'Permission denied to access resource',
        'ValueError': 'Invalid value provided',
        'TypeError': 'Invalid type in operation',
        'ImportError': 'Failed to import required module',
        'ModuleNotFoundError': 'Required module not installed',
        'KeyError': 'Required configuration key missing',
        'AttributeError': 'Invalid attribute access',
        'ConnectionError': 'Network connection failed',
        'TimeoutError': 'Operation timed out',
    }

    base_message = friendly_messages.get(error_type, 'An error occurred')

    # Keep user-friendly context if present
    if error_str and error_str != error_type:
        sanitized = f"{base_message}: {error_str}"
    else:
        sanitized = base_message

    if include_type:
        return f"{error_type}: {sanitized}"

    return sanitized


class RuleManagementError(Exception):
    """Base exception for rule management errors.

    All rule management exceptions inherit from this base class,
    allowing clients to catch all rule-related errors with a single handler.
    """
    pass


class RuleNotFoundError(RuleManagementError):
    """Raised when a rule cannot be found.

    This exception includes the rule_id for proper error context and logging.
    """

    def __init__(self, rule_id: str) -> None:
        """Initialize exception with rule identifier.

        Args:
            rule_id: The rule identifier that was not found
        """
        self.rule_id = rule_id
        super().__init__(f"Rule {rule_id} not found")


class InvalidRuleUpdateError(RuleManagementError):
    """Raised when attempting an invalid rule update.

    Examples of invalid updates:
    - Attempting to change detector_class (should create new rule instead)
    - Attempting to update non-existent fields
    - Violating business rules (e.g., can't modify detection logic)
    """
    pass


class ConcurrentModificationError(RuleManagementError):
    """Raised when concurrent modification detected during update.

    This exception implements optimistic locking validation. When a client
    provides an expected_version that doesn't match the current version,
    it indicates another process has modified the rule since it was read.
    """

    def __init__(self, rule_id: str, expected_version: str, actual_version: str) -> None:
        """Initialize exception with version conflict details.

        Args:
            rule_id: The rule identifier that was being updated
            expected_version: The version the client expected
            actual_version: The actual current version in the database
        """
        self.rule_id = rule_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"Rule {rule_id} was modified by another user. "
            f"Expected version {expected_version}, found {actual_version}"
        )


class RuleValidationError(RuleManagementError):
    """Raised when rule validation fails.

    This covers validation failures including:
    - Invalid detector_class (not in whitelist or not importable)
    - Invalid custom_thresholds (doesn't match detector's schema)
    - Schema validation errors
    - Business rule violations
    """
    pass


class FileSizeLimitExceeded(RuleManagementError):
    """Raised when a file exceeds the maximum allowed size.

    This security control prevents denial-of-service attacks and
    resource exhaustion from processing extremely large files.
    """

    def __init__(self, file_path: str, size_mb: float, limit_mb: int) -> None:
        """Initialize exception with file size details.

        Args:
            file_path: Path to the file that exceeded the limit
            size_mb: Actual file size in megabytes
            limit_mb: Maximum allowed size in megabytes
        """
        self.file_path = file_path
        self.size_mb = size_mb
        self.limit_mb = limit_mb
        super().__init__(
            f"File '{file_path}' ({size_mb:.2f}MB) exceeds maximum size limit of {limit_mb}MB"
        )


class InvalidAPIKeyError(RuleManagementError):
    """Raised when an API key fails validation.

    This provides early validation of API keys before making actual API calls,
    preventing wasted requests and providing clearer error messages.
    """

    def __init__(self, message: str, provider: Optional[str] = None) -> None:
        """Initialize exception with validation details.

        Args:
            message: Description of the validation failure
            provider: Optional provider name (openai, anthropic, etc.)
        """
        self.provider = provider
        if provider:
            super().__init__(f"{provider}: {message}")
        else:
            super().__init__(message)


def validate_file_size(file_path: Path, max_size_mb: int) -> None:
    """Validate that a file does not exceed the maximum allowed size.

    This security control prevents resource exhaustion from processing
    extremely large files. Should be called before reading any user-provided
    or plugin files.

    Args:
        file_path: Path to the file to validate
        max_size_mb: Maximum allowed file size in megabytes

    Raises:
        FileSizeLimitExceeded: If file size exceeds the limit
        FileNotFoundError: If file does not exist

    Examples:
        >>> from pathlib import Path
        >>> validate_file_size(Path("config.yaml"), max_size_mb=10)
        >>> # Raises FileSizeLimitExceeded if file is larger than 10MB
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise FileSizeLimitExceeded(
            file_path=str(file_path),
            size_mb=size_mb,
            limit_mb=max_size_mb
        )


def validate_api_key(api_key: str, provider: str = "openai") -> None:
    """Validate API key format without making API calls.

    This provides early validation to catch configuration errors
    before attempting expensive API operations.

    Args:
        api_key: The API key to validate
        provider: The provider name (openai, anthropic, bedrock)

    Raises:
        InvalidAPIKeyError: If API key format is invalid

    Examples:
        >>> validate_api_key("sk-1234567890abcdef", "openai")
        >>> # Validates format but doesn't check if key actually works
    """
    if not api_key or not api_key.strip():
        raise InvalidAPIKeyError(
            "API key is empty or missing",
            provider=provider
        )

    if len(api_key) < 20:
        raise InvalidAPIKeyError(
            f"API key is too short ({len(api_key)} characters). "
            f"Valid API keys must be at least 20 characters (typically 40+ characters). "
            f"Please check your environment variables.",
            provider=provider
        )

    valid_prefixes = {
        "openai": ["sk-", "sk-proj-"],
        "anthropic": ["sk-ant-"],
        "bedrock": [],  # AWS keys don't have standard prefixes
    }

    provider_lower = provider.lower()
    expected_prefixes = valid_prefixes.get(provider_lower, [])

    if expected_prefixes and not any(api_key.startswith(prefix) for prefix in expected_prefixes):
        raise InvalidAPIKeyError(
            f"API key should start with one of: {', '.join(expected_prefixes)}. "
            f"Got: {api_key[:10]}... "
            f"Please verify you're using the correct API key for {provider}.",
            provider=provider
        )

    invalid_chars = set(api_key) - set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    if invalid_chars:
        raise InvalidAPIKeyError(
            f"API key contains invalid characters: {', '.join(sorted(invalid_chars))}. "
            f"API keys should only contain alphanumeric characters, hyphens, and underscores.",
            provider=provider
        )

    if provider_lower == "openai" and api_key.startswith("sk-proj-"):
        if len(api_key) < 56:
            raise InvalidAPIKeyError(
                f"OpenAI project API key appears truncated. "
                f"Project keys are typically 56+ characters.",
                provider=provider
            )
    elif provider_lower == "anthropic" and api_key.startswith("sk-ant-"):
        if len(api_key) < 40:
            raise InvalidAPIKeyError(
                f"Anthropic API key appears truncated. "
                f"Anthropic keys are typically 100+ characters.",
                provider=provider
            )
