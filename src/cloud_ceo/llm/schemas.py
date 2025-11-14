"""Pydantic schemas for structured LLM outputs.

This module defines typed response schemas for LLM operations,
enabling type-safe and validated responses.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

try:
    from cloud_ceo.guidelines.models import GuidelineViolation
    GUIDELINES_AVAILABLE = True
except ImportError:
    GUIDELINES_AVAILABLE = False


class ViolationExplanation(BaseModel):
    """Structured explanation of a SQL anti-pattern violation.

    Attributes:
        explanation: Detailed explanation of why this is an anti-pattern
        severity: Severity level (critical, high, medium, low)
        business_impact: Business impact description
        databricks_execution: How Spark/Databricks will execute this query
        custom_guideline_violations: List of custom guideline violations (if any)
        related_guideline_ids: List of related custom guideline IDs
    """

    explanation: str = Field(
        description="Detailed explanation of the violation in 2-3 sentences"
    )
    severity: str = Field(
        description="Severity level: critical, high, medium, low"
    )
    business_impact: str = Field(
        description="Business impact including cost and performance implications"
    )
    databricks_execution: Optional[str] = Field(
        default=None,
        description="How Spark will execute this query on Databricks"
    )
    custom_guideline_violations: Optional[List[str]] = Field(
        default=None,
        description="List of custom guideline IDs this violation relates to"
    )
    related_guideline_ids: Optional[List[str]] = Field(
        default=None,
        description="Custom guidelines related to this violation"
    )


class ClusterRecommendation(BaseModel):
    """Structured cluster configuration recommendation.

    Attributes:
        recommendation: Specific recommendation (cluster change or query rewrite)
        recommendation_type: Type of recommendation (cluster_change or query_rewrite)
        reasoning: Why this recommendation is made
        estimated_improvement: Expected performance improvement
        estimated_cost: Cost implications of the recommendation
    """

    recommendation: str = Field(
        description="Specific cluster configuration recommendation or query optimization advice"
    )
    recommendation_type: str = Field(
        description="Type: cluster_change or query_rewrite"
    )
    reasoning: str = Field(
        description="Why this configuration is recommended"
    )
    estimated_improvement: Optional[str] = Field(
        default=None,
        description="Expected performance improvement (e.g., '2x faster', '50% cost reduction')"
    )
    estimated_cost: Optional[str] = Field(
        default=None,
        description="Cost impact of the recommendation"
    )


class QueryRewrite(BaseModel):
    """Structured query rewrite result.

    Attributes:
        original_query: The original SQL query
        optimized_query: The optimized version
        changes_made: List of optimizations applied
        expected_improvement: Expected performance benefit
        custom_guidelines_applied: Custom guideline IDs enforced in rewrite
        guideline_compliance_notes: Notes on custom guideline compliance
    """

    original_query: str = Field(
        description="The original SQL query with anti-patterns"
    )
    optimized_query: str = Field(
        description="The optimized SQL query with anti-patterns fixed"
    )
    changes_made: List[str] = Field(
        description="List of specific optimizations applied"
    )
    expected_improvement: Optional[str] = Field(
        default=None,
        description="Expected performance improvement description"
    )
    custom_guidelines_applied: Optional[List[str]] = Field(
        default=None,
        description="List of custom guideline IDs applied during rewrite"
    )
    guideline_compliance_notes: Optional[str] = Field(
        default=None,
        description="Explanation of how custom guidelines were enforced"
    )
