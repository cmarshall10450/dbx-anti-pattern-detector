"""Simple YAML configuration loader for Cloud CEO.

Loads configuration from YAML file or uses defaults.
"""

from pathlib import Path

import structlog
import yaml

from cloud_ceo.config.schema import CloudCEOConfig

logger = structlog.get_logger(__name__)


def load_config(config_path: str | None = None) -> CloudCEOConfig:
    """Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        CloudCEOConfig: Validated configuration object

    Raises:
        FileNotFoundError: If config_path specified but doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        logger.info("Using default configuration")
        return CloudCEOConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            logger.warning("Empty configuration file, using defaults", path=str(path))
            return CloudCEOConfig()

        config = CloudCEOConfig(**config_dict)
        logger.info("Configuration loaded", path=str(path))
        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")
