"""LangChain-based LLM integration for Cloud CEO.

This module provides LLM capabilities for SQL anti-pattern analysis:
1. Multi-provider LLM support (OpenAI, Anthropic, Bedrock)
2. Context-aware analysis with ContextBuilder
3. Structured output with Pydantic schemas

Example usage:
    from cloud_ceo.llm import LLMOrchestrator, ContextBuilder
    from cloud_ceo.llm.prompts import EXPLAIN_PROMPT
    from cloud_ceo.llm.schemas import ViolationExplanation

    # Initialize orchestrator
    orchestrator = LLMOrchestrator("gpt-4o-mini", "openai")

    # Build context once
    builder = ContextBuilder()
    context = builder.build_context(
        violation=violation,
        all_violations=all_violations,
        execution_metrics=metrics,
        cluster_config=cluster
    )

    # Get structured explanation
    explanation = orchestrator.invoke_with_schema(
        prompt=EXPLAIN_PROMPT,
        schema=ViolationExplanation,
        inputs={"context_summary": context.context_summary}
    )

    if explanation:
        print(explanation.explanation)
        print(f"Severity: {explanation.severity}")
"""

from .context import AnalysisContext, ContextBuilder
from .orchestrator import LLMOrchestrator
from .schemas import ClusterRecommendation, QueryRewrite, ViolationExplanation

__all__ = [
    "LLMOrchestrator",
    "ContextBuilder",
    "AnalysisContext",
    "ViolationExplanation",
    "ClusterRecommendation",
    "QueryRewrite",
]
