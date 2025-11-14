"""Detector plugin loading and validation.

This module provides functionality for:
- Loading detector classes from module paths
- Validating detector implementations
- Security checks for custom detectors
- Caching loaded detectors
"""

import importlib
from pathlib import Path
from typing import Optional

import structlog

from cloud_ceo.config.schema import CloudCEOConfig
from cloud_ceo.rule_engine.base_detector import BaseDetector
from cloud_ceo.rule_engine.detector import RuleDetector
from cloud_ceo.security import DetectorSecurityAnalyzer, SecurityAuditLogger

logger = structlog.get_logger(__name__)

# Global security components
_security_analyzer: Optional[DetectorSecurityAnalyzer] = None
_security_logger = SecurityAuditLogger()

# Simple detector cache to avoid redundant instantiation
_detector_cache: dict[str, RuleDetector] = {}


class SecurityError(Exception):
    """Raised when a security violation is detected during plugin loading."""
    pass


def load_detectors(config: CloudCEOConfig, verbose: bool = False) -> list[RuleDetector]:
    """Load detector classes from configuration, including custom plugins.

    This method:
    1. Loads custom detector paths into sys.path
    2. Validates and imports each detector class
    3. Caches detectors to avoid redundant instantiation

    Args:
        config: Cloud CEO configuration with detector list
        verbose: Enable detailed logging

    Returns:
        List of instantiated detector objects
    """
    # Load custom detector paths first
    config.load_custom_detector_paths()

    detectors = []

    for detector_path in config.detectors:
        try:
            detector = load_single_detector(detector_path, verbose, config)
            detectors.append(detector)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {detector_path}: {e}")
            logger.warning(
                "detector_load_failed",
                detector_path=detector_path,
                error=str(e)
            )

    return detectors


def load_single_detector(
    detector_path: str,
    verbose: bool = False,
    config: Optional[CloudCEOConfig] = None
) -> RuleDetector:
    """Load a single detector by module path with security analysis.

    Args:
        detector_path: Fully qualified detector path (e.g., "module.path:ClassName")
        verbose: Enable detailed logging
        config: Optional configuration for security settings

    Returns:
        Instantiated detector object

    Raises:
        ImportError: If module cannot be imported
        AttributeError: If class not found in module
        TypeError: If class is not a valid detector
        SecurityError: If security analysis fails for high-risk detectors
    """
    # Check cache first
    if detector_path in _detector_cache:
        if verbose:
            class_name = detector_path.split(".")[-1].split(":")[-1]
            print(f"  Cached: {class_name}")
        return _detector_cache[detector_path]

    # Parse detector path
    if ":" in detector_path:
        module_path, class_name = detector_path.split(":", 1)
    else:
        parts = detector_path.split(".")
        module_path, class_name = ".".join(parts[:-1]), parts[-1]

    # Import module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import detector module '{module_path}': {e}. "
            f"Ensure the module is installed or add its path to custom_detector_paths."
        )

    # SECURITY: Static analysis for custom detectors
    module_file = getattr(module, "__file__", None)
    is_custom_detector = not module_path.startswith("cloud_ceo.detectors.")

    if config and config.security.enable_static_analysis and is_custom_detector and module_file:
        if verbose:
            print(f"  Running security analysis on {class_name}...")

        try:
            global _security_analyzer
            if _security_analyzer is None:
                _security_analyzer = DetectorSecurityAnalyzer(
                    max_file_size_mb=config.security.max_plugin_file_size_mb
                )
            analysis = _security_analyzer.analyze_detector(module_file)
            _security_logger.log_detector_load(detector_path, analysis)

            risk_level = analysis.get('risk_level', 'unknown')

            if risk_level in ['high', 'critical'] and config.security.block_high_risk_detectors:
                raise SecurityError(
                    f"Security: Detector '{detector_path}' has {risk_level} risk level. "
                    f"Issues found: {len(analysis.get('issues', []))}. "
                    f"Dangerous imports: {', '.join(analysis.get('dangerous_imports', []))}. "
                    f"To allow, set security.block_high_risk_detectors=false in config."
                )

            if risk_level in ['medium', 'high', 'critical'] and verbose:
                print(
                    f"  Warning: Detector has {risk_level} risk level. "
                    f"Issues: {len(analysis.get('issues', []))}"
                )

            _security_logger.log_security_validation_passed(detector_path, analysis)

        except SecurityError:
            raise
        except Exception as e:
            if verbose:
                print(f"  Warning: Security analysis failed: {e}")
            logger.warning(
                "security_analysis_failed",
                detector=detector_path,
                error=str(e)
            )

    # SECURITY: Verify built-in detectors come from the actual cloud_ceo package
    # This prevents module shadowing attacks
    if module_path.startswith("cloud_ceo.detectors."):
        if module_file:
            module_file_path = Path(module_file).resolve()
            # Check if module is from site-packages (installed) or src/cloud_ceo (development)
            is_from_site_packages = "site-packages" in str(module_file_path)
            is_from_src = "src/cloud_ceo" in str(module_file_path)

            if not (is_from_site_packages or is_from_src):
                raise SecurityError(
                    f"Security: Built-in detector '{module_path}' is being loaded from "
                    f"an unexpected location: {module_file_path}. "
                    f"This could be a module shadowing attack. "
                    f"Built-in detectors must be loaded from the installed cloud_ceo package."
                )

    # Get detector class
    try:
        detector_class = getattr(module, class_name)
    except AttributeError:
        raise AttributeError(
            f"Detector class '{class_name}' not found in module '{module_path}'. "
            f"Check the class name and ensure it's exported."
        )

    # Validate it's a detector subclass
    if not isinstance(detector_class, type) or not issubclass(detector_class, BaseDetector):
        raise TypeError(
            f"'{class_name}' is not a valid detector class. "
            f"Custom detectors must inherit from BaseDetector or RuleDetector."
        )

    # Instantiate detector
    try:
        detector = detector_class()
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate detector '{class_name}': {e}. "
            f"Check the __init__ method signature."
        )

    # Validate detector has required attributes
    if not hasattr(detector, 'rule_id'):
        raise AttributeError(
            f"Detector '{class_name}' must set 'rule_id' in __init__"
        )
    if not hasattr(detector, 'severity'):
        raise AttributeError(
            f"Detector '{class_name}' must set 'severity' in __init__"
        )

    # Cache and return
    _detector_cache[detector_path] = detector

    if verbose:
        print(f"  Loaded: {class_name} (rule_id={detector.rule_id})")

    logger.info(
        "detector_loaded",
        detector_class=class_name,
        module=module_path,
        rule_id=detector.rule_id
    )

    return detector


def validate_detector_plugin(detector_class: type) -> list[str]:
    """Validate a detector plugin implementation.

    Args:
        detector_class: Detector class to validate

    Returns:
        List of error messages (empty list if valid)
    """
    errors = []

    # Check inheritance
    try:
        if not issubclass(detector_class, BaseDetector):
            errors.append(
                f"Detector must inherit from BaseDetector or RuleDetector"
            )
    except TypeError:
        errors.append(f"'{detector_class}' is not a valid class")
        return errors

    # Try to instantiate
    try:
        instance = detector_class()
    except Exception as e:
        errors.append(f"Failed to instantiate detector: {e}")
        return errors

    # Check required attributes
    if not hasattr(instance, 'rule_id'):
        errors.append("Detector must set 'rule_id' in __init__()")

    if not hasattr(instance, 'severity'):
        errors.append("Detector must set 'severity' in __init__()")

    # Check detect method
    if not hasattr(instance, 'detect'):
        errors.append("Detector must implement detect() method")

    # Warn if RuleDetector overrides detect()
    if isinstance(instance, RuleDetector):
        if 'detect' in detector_class.__dict__:
            errors.append(
                "Warning: RuleDetector subclass should not override detect() method. "
                "Override _detect_violations() instead."
            )

    return errors


# Backwards compatibility: expose old names
_load_detectors = load_detectors
_load_single_detector = load_single_detector
