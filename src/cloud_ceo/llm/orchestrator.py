"""LLM Orchestrator for multi-provider support.

This module provides a thin wrapper around LangChain for coordinating
LLM operations across multiple providers (OpenAI, Anthropic, Bedrock).
"""

import os
from typing import Optional, Type

import structlog
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError

from cloud_ceo.llm.rate_limiter import LLMRateLimiter, RateLimitExceeded
from cloud_ceo.rule_engine.exceptions import validate_api_key, InvalidAPIKeyError

logger = structlog.get_logger(__name__)


class LLMOrchestrator:
    """Thin wrapper coordinating LangChain components.

    Replaces SimpleLLMClient with multi-provider support and
    enables LangGraph workflows.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        provider: str = "openai",
        max_requests_per_minute: int = 60,
        max_cost_per_hour: float = 10.0
    ) -> None:
        """Initialize with any supported LLM provider.

        Args:
            model_name: Model identifier (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022")
            provider: Provider name ("openai", "anthropic", "bedrock")
            max_requests_per_minute: Maximum API requests per minute (default: 60)
            max_cost_per_hour: Maximum cost per hour in USD (default: 10.0)
        """
        self.model_name = model_name
        self.provider = provider
        self.model: Optional[BaseChatModel] = None
        self._available = False

        # SECURITY FIX H-3: Initialize rate limiter
        self.rate_limiter = LLMRateLimiter(
            max_requests_per_minute=max_requests_per_minute,
            max_cost_per_hour=max_cost_per_hour
        )

        # SECURITY FIX M-4: Validate API key format before initialization
        api_key = self._get_api_key_for_provider(provider)
        if api_key:
            validate_api_key(api_key, provider)

        try:
            # Format: "provider:model" for init_chat_model
            model_identifier = f"{provider}:{model_name}"
            self.model = init_chat_model(model_identifier)
            self._available = self._check_availability()
        except Exception as e:
            logger.error("llm_init_failed", error=str(e), exc_info=True)
            self.model = None
            self._available = False

    def _get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for the specified provider from environment.

        Args:
            provider: Provider name (openai, anthropic, bedrock)

        Returns:
            API key string or None if not found
        """
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",  # Primary key for AWS
        }

        env_var = provider_keys.get(provider.lower())
        if env_var:
            return os.getenv(env_var)
        return None

    def _check_availability(self) -> bool:
        """Verify API keys and model availability.

        Returns:
            True if LLM is configured and API key is present
        """
        if self.model is None:
            return False

        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        }

        key_spec = provider_keys.get(self.provider)
        if isinstance(key_spec, list):
            return all(os.getenv(key) for key in key_spec)
        else:
            return bool(os.getenv(key_spec or ""))

    def is_available(self) -> bool:
        """Check if LLM is configured and available.

        Returns:
            True if LLM client is ready to use
        """
        return self._available

    def invoke_with_schema(
        self,
        prompt: ChatPromptTemplate,
        schema: Type[BaseModel],
        inputs: dict,
        strict: bool = True
    ) -> Optional[BaseModel]:
        """Invoke LLM with STRICT structured output validation and rate limiting.

        Args:
            prompt: ChatPromptTemplate to render
            schema: Pydantic model for response structure
            inputs: Variables for prompt template
            strict: Enforce strict validation (default: True)

        Returns:
            Validated Pydantic model or None if validation fails

        Raises:
            ValidationError: If strict=True and validation fails
            RateLimitExceeded: If rate limit is exceeded
        """
        if not self.is_available() or self.model is None:
            logger.warning("llm_unavailable", schema=schema.__name__)
            return None

        # SECURITY FIX H-3: Check rate limit before invoking
        allowed, reason = self.rate_limiter.check_rate_limit()
        if not allowed:
            logger.error("llm_rate_limit_exceeded", reason=reason, schema=schema.__name__)
            raise RateLimitExceeded(f"LLM rate limit exceeded: {reason}")

        try:
            chain = prompt | self.model.with_structured_output(schema)
            result = chain.invoke(inputs)

            # SECURITY FIX H-3: Record successful request
            self.rate_limiter.record_request()

            if result is not None:
                # SECURITY FIX H-2: Explicit Pydantic validation
                # Check if result is already a validated instance of the schema
                if isinstance(result, schema):
                    # Already validated by LangChain, return as-is
                    logger.info(
                        "llm_invocation_success",
                        schema=schema.__name__,
                        validated=True
                    )
                    return result

                # Result is a dict or other type - validate it
                validated = schema.model_validate(
                    result.model_dump() if isinstance(result, BaseModel) else result,
                    strict=strict
                )

                logger.info(
                    "llm_invocation_success",
                    schema=schema.__name__,
                    validated=True
                )
                return validated

            logger.warning("llm_returned_none", schema=schema.__name__)
            return None

        except RateLimitExceeded:
            raise

        except ValidationError as e:
            logger.error(
                "llm_validation_failed",
                schema=schema.__name__,
                errors=e.errors()
            )
            if strict:
                raise
            return None

        except Exception as e:
            logger.error(
                "llm_invocation_failed",
                schema=schema.__name__,
                error=str(e),
                exc_info=True
            )
            return None

    def get_rate_limit_stats(self) -> dict:
        """Get current rate limiter statistics.

        Returns:
            Dictionary with current usage stats
        """
        return self.rate_limiter.get_stats()
