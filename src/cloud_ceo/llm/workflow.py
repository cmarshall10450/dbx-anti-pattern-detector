"""LangGraph multi-step analysis workflow.

This module implements a 5-node LangGraph workflow for analyzing SQL anti-patterns
with cluster-aware recommendations and graceful degradation.

Workflow: validate_context → classify_severity → analyze_impact → generate_recommendations → validate_recommendations
"""

import logging
import signal
from contextlib import contextmanager
from typing import List, Literal, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from cloud_ceo.llm.context import (
    AnalysisContext,
    ClusterConfig,
    ContextBuilder,
    ExecutionMetrics,
    Violation,
)
from cloud_ceo.llm.context_validator import ContextValidator
from cloud_ceo.llm.orchestrator import LLMOrchestrator
from cloud_ceo.llm.prompts import (
    IMPACT_PROMPT,
    RECOMMEND_PROMPT,
    REFINEMENT_PROMPT,
    SEVERITY_PROMPT,
    create_severity_prompt,
    create_impact_prompt,
    create_recommend_prompt,
    create_refinement_prompt,
)
from cloud_ceo.llm.state import (
    AnalysisState,
    EffortLevel,
    ImpactAnalysis,
    Recommendation,
    RecommendationList,
    RecommendationType,
    RoutingDecision,
    SeverityAssessment,
    SeverityLevel,
    WorkflowResult,
)
from cloud_ceo.security.prompt_sanitizer import PromptSanitizer

logger = logging.getLogger(__name__)

# Configuration
MAX_REFINEMENT_ATTEMPTS = 1  # Allow 1 refinement = 2 total LLM calls max


@contextmanager
def timeout_context(seconds: int):
    """Context manager for workflow timeout.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        TimeoutError: If workflow exceeds timeout

    Yields:
        None
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Workflow exceeded {seconds}s timeout")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class AnalysisWorkflow:
    """LangGraph workflow for multi-step violation analysis.

    This class encapsulates the 5-node workflow with dependency injection
    for the LLMOrchestrator and ContextValidator.
    """

    def __init__(
        self,
        orchestrator: LLMOrchestrator,
        industry_mode: Optional[str] = None,
        enable_compression: bool = True
    ) -> None:
        """Initialize workflow with LLM orchestrator and optional industry mode.

        Args:
            orchestrator: Configured LLM orchestrator instance
            industry_mode: Optional industry compliance mode (uk_finance, uk_retail, uk_healthcare, pharma_biotech, uk_energy)
            enable_compression: Enable context compression (default: True)
        """
        self.orchestrator = orchestrator
        self.industry_mode = industry_mode
        self.enable_compression = enable_compression

        # Initialize context validator (lightweight, no Redis)
        self.context_validator = ContextValidator() if enable_compression else None

        # Initialize prompt sanitizer for injection protection
        self.prompt_sanitizer = PromptSanitizer()

        # Create industry-aware prompts if industry mode specified
        if industry_mode:
            try:
                from cloud_ceo.llm.industry_prompts import get_industry_prompt
                industry_context = get_industry_prompt(industry_mode)
                self.severity_prompt = create_severity_prompt(industry_context)
                self.impact_prompt = create_impact_prompt(industry_context)
                self.recommend_prompt = create_recommend_prompt(industry_context)
                self.refinement_prompt = create_refinement_prompt(industry_context)
            except Exception:
                # Fall back to default prompts if industry prompts unavailable
                self.severity_prompt = SEVERITY_PROMPT
                self.impact_prompt = IMPACT_PROMPT
                self.recommend_prompt = RECOMMEND_PROMPT
                self.refinement_prompt = REFINEMENT_PROMPT
        else:
            self.severity_prompt = SEVERITY_PROMPT
            self.impact_prompt = IMPACT_PROMPT
            self.recommend_prompt = RECOMMEND_PROMPT
            self.refinement_prompt = REFINEMENT_PROMPT

    def validate_context(self, state: AnalysisState) -> dict:
        """Node 0: Compress large context using proven techniques.

        This node implements the Context Validator pattern using:
        - Hybrid search (BM25 + semantic similarity)
        - Relevance filtering for regulations/guidelines
        - In-memory LRU caching (no Redis dependency)

        CRITICAL: Preserves <all_violations> XML section from context_summary
        to ensure LLM has all violations for recommendation generation.

        Production evidence:
        - LangChain DocumentCompressor (100k+ repos)
        - Semantic search (Pinecone, Weaviate standard)
        - Expected compression: 47-57% (1,500-3,500 → 800-1,500 tokens)

        Args:
            state: Current analysis state

        Returns:
            Partial state update with compressed_context
        """
        # Skip if compression disabled
        if not state.get("compression_enabled", True) or not self.context_validator:
            return {"compressed_context": None}

        try:
            # Extract industry context if available
            industry_context = None
            if self.industry_mode:
                try:
                    from cloud_ceo.llm.industry_prompts import get_industry_prompt
                    industry_context = get_industry_prompt(self.industry_mode)
                except Exception:
                    pass

            # Compress context
            compressed = self.context_validator.compress_context(
                violation_pattern=state["violation"].pattern,
                industry_context=industry_context,
                custom_guidelines=state.get("custom_guidelines_text"),
                metrics_context=state["context"].context_summary if state["context"].has_metrics else None
            )

            # Extract critical sections from original context
            # These sections are CRITICAL for all LLM nodes (severity, impact, recommendations)
            original_context = state["context"].context_summary
            critical_sections = self._extract_critical_sections(original_context)

            # Build compressed context summary with preserved critical sections
            parts = []

            # Add wrapper for valid XML structure
            parts.append("<query_context>")

            # ALWAYS include all critical sections (highest priority)
            # 1. Violations (for recommendations)
            if critical_sections.get('violations'):
                parts.append(critical_sections['violations'])
                logger.info(f"✓ Preserved violations XML ({len(critical_sections['violations'])} chars)")
            else:
                logger.warning("⚠️ No violations XML found in original context!")

            # 2. Current violation context
            if critical_sections.get('current_violation'):
                parts.append(critical_sections['current_violation'])
                logger.info(f"✓ Preserved current violation ({len(critical_sections['current_violation'])} chars)")

            # 3. Performance profile (for severity assessment)
            if critical_sections.get('performance'):
                parts.append(critical_sections['performance'])
                logger.info(f"✓ Preserved performance profile ({len(critical_sections['performance'])} chars)")
            else:
                logger.warning("⚠️ No performance profile found - severity assessment may be limited")

            # 4. Resource utilization (for impact analysis)
            if critical_sections.get('resources'):
                parts.append(critical_sections['resources'])
                logger.info(f"✓ Preserved resource utilization ({len(critical_sections['resources'])} chars)")
            else:
                logger.warning("⚠️ No resource utilization found - impact analysis may be limited")

            # 5. Cluster configuration (for capacity-aware analysis)
            if critical_sections.get('cluster'):
                parts.append(critical_sections['cluster'])
                logger.info(f"✓ Preserved cluster config ({len(critical_sections['cluster'])} chars)")

            # Add compressed supplementary content (regulations, guidelines)
            if compressed.filtered_regulations:
                parts.append(f"  <filtered_regulations>\n{compressed.filtered_regulations}\n  </filtered_regulations>")
            if compressed.filtered_guidelines:
                parts.append(f"  <filtered_guidelines>\n{compressed.filtered_guidelines}\n  </filtered_guidelines>")

            # Close XML wrapper
            parts.append("</query_context>")

            compressed_summary = "\n\n".join(parts) if parts else None

            # Log compression stats
            if compressed_summary:
                logger.info(
                    f"Context compressed: {compressed.stats.original_tokens} → "
                    f"{compressed.stats.compressed_tokens} tokens "
                    f"({compressed.stats.compression_ratio:.0%} ratio, "
                    f"{compressed.stats.processing_time_ms:.1f}ms)"
                )

            return {"compressed_context": compressed_summary}

        except Exception as e:
            logger.warning(f"Context compression failed, using full context: {e}")
            return {"compressed_context": None}

    def _extract_violations_xml(self, context_summary: str) -> Optional[str]:
        """Extract violations XML section from context summary.

        Extracts both <violations_summary> and <all_violations> sections
        which are critical for LLM recommendation generation.

        Args:
            context_summary: Full context summary string

        Returns:
            Extracted XML string or None if not found
        """
        import re

        # Pattern to extract complete violations section
        # Match from <all_violations> to </all_violations> (inclusive)
        # Note: <violations_summary> is INSIDE <all_violations>
        pattern = r'(<all_violations\s+count="\d+">.*?</all_violations>)'

        match = re.search(pattern, context_summary, re.DOTALL)
        if match:
            return match.group(1)

        return None

    def _extract_critical_sections(self, context_summary: str) -> dict:
        """Extract critical XML sections that must be preserved for all nodes.

        Extracts violations, performance profile, resource utilization, and cluster config
        which are essential for severity and impact analysis.

        Args:
            context_summary: Full context summary string

        Returns:
            Dict with extracted sections
        """
        import re

        sections = {}

        # Extract violations (already handled by _extract_violations_xml)
        sections['violations'] = self._extract_violations_xml(context_summary)

        # Extract performance profile (critical for severity assessment)
        perf_pattern = r'(<performance_profile>.*?</performance_profile>)'
        perf_match = re.search(perf_pattern, context_summary, re.DOTALL)
        if perf_match:
            sections['performance'] = perf_match.group(1)

        # Extract resource utilization (critical for impact analysis)
        resource_pattern = r'(<resource_utilization>.*?</resource_utilization>)'
        resource_match = re.search(resource_pattern, context_summary, re.DOTALL)
        if resource_match:
            sections['resources'] = resource_match.group(1)

        # Extract cluster configuration (needed for capacity assessment)
        cluster_pattern = r'(<cluster_configuration>.*?</cluster_configuration>)'
        cluster_match = re.search(cluster_pattern, context_summary, re.DOTALL)
        if cluster_match:
            sections['cluster'] = cluster_match.group(1)

        # Extract current violation (needed for context)
        current_pattern = r'(<current_violation>.*?</current_violation>)'
        current_match = re.search(current_pattern, context_summary, re.DOTALL)
        if current_match:
            sections['current_violation'] = current_match.group(1)

        return sections

    def classify_severity(self, state: AnalysisState) -> dict:
        """Node 1: Cluster-aware severity assessment.

        KEY DIFFERENTIATOR: Same violation gets different severity
        based on cluster resources (e.g., 5GB broadcast = CRITICAL
        on 8GB executors, LOW on 32GB executors).

        Uses compressed_context if available (47-57% token reduction).

        Args:
            state: Current analysis state

        Returns:
            Partial state update with severity_assessment
        """
        if not self.orchestrator.is_available():
            logger.warning("LLM unavailable - using default severity assessment")
            return {"severity_assessment": self._default_severity(state)}

        try:
            # Prefer compressed context if available, otherwise use full context
            context_summary = (
                state.get("compressed_context") or state["context"].context_summary
            )

            # Sanitize context summary for prompt injection protection
            # Pass max_length=None to avoid truncating the structured XML context
            sanitized_context = self.prompt_sanitizer.sanitize_violation_message(
                context_summary, max_length=None
            )

            assessment = self.orchestrator.invoke_with_schema(
                prompt=self.severity_prompt,  # Use industry-aware prompt
                schema=SeverityAssessment,
                inputs={
                    "context_summary": sanitized_context,
                    "confidence_level": state["context"].confidence_level,
                },
            )

            if assessment is None:
                return {"severity_assessment": self._default_severity(state)}

            logger.info(
                f"Severity: {assessment.severity} (confidence: {assessment.confidence:.2f})"
            )

            new_message = HumanMessage(
                content=f"Assessed severity: {assessment.severity} - {assessment.evidence}"
            )
            return {"severity_assessment": assessment, "messages": [new_message]}

        except Exception as e:
            logger.error(f"Severity classification failed: {e}")
            return {"severity_assessment": self._default_severity(state)}

    def analyze_impact(self, state: AnalysisState) -> dict:
        """Node 2: Analyze performance and cost impact.

        Uses derived metrics with business context
        (e.g., "7987MB peak = 97% of 8GB executors - HIGH RISK").

        Uses compressed_context if available (token savings).

        Args:
            state: Current analysis state

        Returns:
            Partial state update with impact_analysis
        """
        if not self.orchestrator.is_available():
            logger.warning("LLM unavailable for impact analysis")
            return {}

        try:
            # Prefer compressed context if available
            context_summary = (
                state.get("compressed_context") or state["context"].context_summary
            )

            # Sanitize context summary for prompt injection protection
            # Pass max_length=None to avoid truncating the structured XML context
            sanitized_context = self.prompt_sanitizer.sanitize_violation_message(
                context_summary, max_length=None
            )

            # Get severity assessment for context
            assessment = state.get("severity_assessment")

            analysis = self.orchestrator.invoke_with_schema(
                prompt=self.impact_prompt,  # Use industry-aware prompt
                schema=ImpactAnalysis,
                inputs={
                    "context_summary": sanitized_context,
                    "severity": (
                        assessment.model_dump_json() if assessment else "unknown"
                    ),
                },
            )

            if analysis is None:
                logger.warning("Impact analysis returned None")
                return {}

            logger.info(f"Impact: {analysis.root_cause[:50]}...")

            new_message = HumanMessage(
                content=f"Impact analysis: {analysis.root_cause}"
            )
            return {"impact_analysis": analysis, "messages": [new_message]}

        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
            return {}

    def generate_recommendations(self, state: AnalysisState) -> dict:
        """Node 3: Generate ranked recommendations.

        MANDATORY: LLM must generate recommendations. No fallback system.

        If no violations exist, returns explicit "no issues found" message.
        If violations exist, LLM must analyze and provide recommendations.

        Args:
            state: Current analysis state

        Returns:
            Partial state update with recommendations

        Raises:
            RuntimeError: If LLM fails to generate recommendations when violations exist
        """
        logger.info("=== GENERATE RECOMMENDATIONS NODE ===")
        logger.info(f"Impact analysis exists: {state.get('impact_analysis') is not None}")
        logger.info(f"LLM orchestrator available: {self.orchestrator.is_available()}")

        # Check if we have any violations at all
        all_violations = state.get("all_violations", [state["violation"]])

        # DEBUG: Log violation details
        logger.info(f"=== SENDING {len(all_violations)} VIOLATIONS TO LLM ===")
        for i, v in enumerate(all_violations, 1):
            logger.info(f"{i}. Pattern={v.pattern}, Line={v.line_number}, Fragment={v.fragment[:50]}...")
            if hasattr(v, 'default_message'):
                logger.info(f"   Message: {v.default_message[:100]}...")

        if not all_violations:
            logger.info("No violations detected - returning 'all clear' recommendation")
            return self._generate_no_issues_response()

        # LLM must be available for recommendation generation
        if not self.orchestrator.is_available():
            error_msg = "LLM is unavailable but required for recommendation generation"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        impact = state.get("impact_analysis")
        if impact is None:
            error_msg = "Impact analysis missing but required for recommendation generation"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Attempt LLM generation with retry
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"LLM recommendation attempt {attempt + 1}/{max_retries}")

                # Prefer compressed context if available
                context_summary = (
                    state.get("compressed_context") or state["context"].context_summary
                )

                # DEBUG: Log what context is being sent
                logger.info("=== CONTEXT SUMMARY BEING SENT TO LLM ===")
                logger.info(f"Context length: {len(context_summary)} chars")
                logger.info("=== BEGIN CONTEXT ===")
                logger.info(context_summary)
                logger.info("=== END CONTEXT ===")

                # Check if all violations are in the context
                violations_in_context = context_summary.count("<violation")
                if "<all_violations" in context_summary:
                    logger.info(f"✓ Found <all_violations> section with {violations_in_context} violations")
                else:
                    logger.warning(f"⚠️ NO <all_violations> section found in context!")

                # Validate XML context before sanitization (validation uses original structure)
                self._validate_xml_context(context_summary, len(all_violations))

                # Sanitize context summary for prompt injection protection
                # Pass max_length=None to avoid truncating the structured XML context
                sanitized_context = self.prompt_sanitizer.sanitize_violation_message(
                    context_summary, max_length=None
                )

                logger.info("Invoking LLM with recommendation prompt...")
                result = self.orchestrator.invoke_with_schema(
                    prompt=self.recommend_prompt,
                    schema=RecommendationList,
                    inputs={
                        "context_summary": sanitized_context,
                        "impact": impact.model_dump_json(),
                    },
                )

                # Validate LLM response
                if result is None:
                    logger.error(f"LLM returned None on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError("LLM returned None after all retries")

                if not result.recommendations or len(result.recommendations) == 0:
                    logger.error(f"LLM returned empty recommendations on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                    raise RuntimeError("LLM returned empty recommendations list after all retries")

                recommendations = result.recommendations
                violation_explanation = result.violation_explanation

                logger.info(f"=== LLM SUCCESS: {len(recommendations)} RECOMMENDATIONS ===")
                logger.info(f"Violation explanation length: {len(violation_explanation) if violation_explanation else 0}")
                for idx, rec in enumerate(recommendations, 1):
                    logger.info(f"  Rec {idx}: {rec.description[:50]}... (type={rec.type}, effort={rec.effort_level})")

                new_message = HumanMessage(
                    content=f"Generated {len(recommendations)} recommendations with violation explanation"
                )
                return {
                    "recommendations": recommendations,
                    "violation_explanation": violation_explanation,
                    "messages": [new_message]
                }

            except ValidationError as e:
                logger.error(f"=== PYDANTIC VALIDATION ERROR (attempt {attempt + 1}) ===")
                logger.error(f"Validation errors: {e.errors()}")
                if hasattr(e, 'model'):
                    logger.error(f"Raw LLM response: {e.model}")

                if attempt < max_retries - 1:
                    logger.info("Retrying LLM invocation...")
                    continue
                raise RuntimeError(f"Schema validation failed after all retries: {e}")

            except Exception as e:
                logger.error(f"LLM invocation failed on attempt {attempt + 1}: {e}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")

                if attempt < max_retries - 1:
                    logger.info("Retrying LLM invocation...")
                    continue
                raise RuntimeError(f"LLM recommendation generation failed after all retries: {e}")

        # Should never reach here due to raises above, but for safety
        raise RuntimeError("Failed to generate recommendations after all attempts")

    def _validate_xml_context(self, context_summary: str, expected_violation_count: int) -> None:
        """Validate XML context contains required violation elements.

        This is a non-blocking validation that logs warnings if XML structure
        is missing expected elements. Does not raise exceptions.

        Validates:
        - Presence of <all_violations> section
        - count attribute matches expected violation count
        - Individual <violation> tags present

        Args:
            context_summary: The XML context string to validate
            expected_violation_count: Number of violations expected in context
        """
        validation_issues = []

        # Check 1: Presence of <all_violations> section
        if "<all_violations" not in context_summary:
            validation_issues.append("Missing <all_violations> section in XML context")
        else:
            # Check 2: Verify count attribute
            import re
            count_match = re.search(r'<all_violations\s+count="(\d+)"', context_summary)
            if count_match:
                context_count = int(count_match.group(1))
                if context_count != expected_violation_count:
                    validation_issues.append(
                        f"Violation count mismatch: expected {expected_violation_count}, "
                        f"found {context_count} in XML"
                    )
            else:
                validation_issues.append("Missing count attribute in <all_violations> tag")

            # Check 3: Count individual <violation> tags
            violation_tags = context_summary.count("<violation")
            if violation_tags < expected_violation_count:
                validation_issues.append(
                    f"Insufficient violation tags: expected {expected_violation_count}, "
                    f"found {violation_tags}"
                )

        # Check 4: Presence of violations_summary section (Phase 1 enhancement)
        if "<violations_summary>" not in context_summary:
            validation_issues.append("Missing <violations_summary> section (Phase 1 enhancement)")

        # Log all validation issues
        if validation_issues:
            logger.warning("=== XML CONTEXT VALIDATION WARNINGS ===")
            for issue in validation_issues:
                logger.warning(f"  - {issue}")
            logger.warning(
                "These warnings indicate potential formatting issues that may affect LLM coverage. "
                "Proceeding with LLM invocation but coverage may be impacted."
            )
        else:
            logger.info(f"✓ XML context validation passed ({expected_violation_count} violations)")

    def _generate_no_issues_response(self) -> dict:
        """Generate response when no violations are detected.

        Returns:
            State update dict with informational message
        """
        logger.info("Generating 'no issues found' response")

        no_issue_recommendation = Recommendation(
            type=RecommendationType.CODE_FIX,
            violations_addressed=[],
            description="No SQL anti-patterns detected - query is well optimized",
            expected_improvement="Query follows best practices",
            effort_level=EffortLevel.LOW,
            implementation="No changes needed",
            user_explanation="Great job! Your query doesn't have any of the common performance anti-patterns we check for."
        )

        violation_explanation = (
            "No issues found! Your query follows Spark SQL best practices and "
            "doesn't exhibit any of the performance anti-patterns we monitor."
        )

        new_message = HumanMessage(
            content="No violations detected - query is well optimized"
        )

        return {
            "recommendations": [no_issue_recommendation],
            "violation_explanation": violation_explanation,
            "messages": [new_message]
        }


    def validate_recommendations(self, state: AnalysisState) -> dict:
        """Node 4: Validate recommendations are actionable.

        Filters out invalid or conflicting recommendations.

        Args:
            state: Current analysis state

        Returns:
            Partial state update with validated recommendations
        """
        recommendations = state.get("recommendations", [])

        if not recommendations:
            logger.info("No recommendations to validate")
            return {}

        valid_recs = [
            rec
            for rec in recommendations
            if self._is_actionable(rec) and self._is_valid_config(rec)
        ]

        filtered_count = len(recommendations) - len(valid_recs)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} invalid recommendations")

        all_violations = state.get("all_violations", [])
        if all_violations and valid_recs:
            valid_recs = self._ensure_all_violations_addressed(valid_recs, all_violations)

        return {"recommendations": valid_recs}

    def _check_violation_coverage(
        self,
        recommendations: List[Recommendation],
        all_violations: List[Violation]
    ) -> List[Violation]:
        """Check which violations are not addressed by current recommendations.

        Uses set for O(1) lookup performance as specified in implementation plan.

        Args:
            recommendations: Current list of recommendations
            all_violations: All detected violations

        Returns:
            List of violations not addressed by any recommendation
        """
        addressed_patterns = set()
        for rec in recommendations:
            addressed_patterns.update(rec.violations_addressed)

        missing_violations = [
            v for v in all_violations
            if v.pattern not in addressed_patterns
        ]

        return missing_violations

    def _create_refinement_feedback(
        self,
        missing_violations: List[Violation],
        existing_recommendations: List[Recommendation],
        all_violations: List[Violation]
    ) -> dict:
        """Create feedback text for refinement prompt.

        Args:
            missing_violations: Violations that need recommendations
            existing_recommendations: Already generated recommendations
            all_violations: All detected violations

        Returns:
            Dict with formatted feedback text for refinement prompt
        """
        addressed_patterns = set()
        for rec in existing_recommendations:
            addressed_patterns.update(rec.violations_addressed)

        missing_violations_text = []
        for v in missing_violations:
            missing_violations_text.append(
                f"- {v.pattern} at line {v.line_number}: {v.fragment[:100]}"
            )

        addressed_violations_text = []
        for v in all_violations:
            if v.pattern in addressed_patterns:
                rec_nums = [
                    i + 1 for i, rec in enumerate(existing_recommendations)
                    if v.pattern in rec.violations_addressed
                ]
                rec_list = ", ".join([f"#{n}" for n in rec_nums])
                addressed_violations_text.append(
                    f"- {v.pattern} (covered by recommendation {rec_list})"
                )

        existing_recommendations_text = []
        for i, rec in enumerate(existing_recommendations, 1):
            existing_recommendations_text.append(
                f"#{i}: {rec.description} (fixes: {', '.join(rec.violations_addressed)})"
            )

        return {
            "missing_violations_text": "\n".join(missing_violations_text) if missing_violations_text else "None",
            "addressed_violations_text": "\n".join(addressed_violations_text) if addressed_violations_text else "None",
            "existing_recommendations_text": "\n".join(existing_recommendations_text) if existing_recommendations_text else "None"
        }

    def check_coverage(self, state: AnalysisState) -> dict:
        """Node 5: Check if all violations are addressed by recommendations.

        This node validates that every violation has at least one recommendation.
        If violations are missing, it updates state with missing_violations for refinement.

        Args:
            state: Current analysis state

        Returns:
            Partial state update with missing_violations
        """
        recommendations = state.get("recommendations", [])
        all_violations = state.get("all_violations", [])

        if not recommendations or not all_violations:
            logger.info("No recommendations or violations to check coverage")
            return {"missing_violations": None}

        missing_violations = self._check_violation_coverage(recommendations, all_violations)

        if missing_violations:
            missing_patterns = [v.pattern for v in missing_violations]
            logger.warning(
                f"Coverage check: {len(missing_violations)} violations not addressed: {missing_patterns}"
            )
        else:
            logger.info("Coverage check: All violations addressed")

        return {"missing_violations": missing_violations if missing_violations else None}

    def generate_missing_recommendations(self, state: AnalysisState) -> dict:
        """Node 6: Generate recommendations for missing violations only.

        This node is called when check_coverage identifies missing violations.
        It uses a focused REFINEMENT_PROMPT to generate recommendations ONLY
        for the missing violations without duplicating existing ones.

        Args:
            state: Current analysis state

        Returns:
            Partial state update with merged recommendations and updated refinement_count
        """
        logger.info("=== GENERATE MISSING RECOMMENDATIONS NODE ===")

        missing_violations = state.get("missing_violations", [])
        existing_recommendations = state.get("recommendations", [])
        refinement_count = state.get("refinement_count", 0)

        if not missing_violations:
            logger.info("No missing violations - skipping refinement")
            return {}

        if refinement_count >= MAX_REFINEMENT_ATTEMPTS:
            logger.warning(
                f"Max refinement attempts ({MAX_REFINEMENT_ATTEMPTS}) reached. "
                f"Accepting partial results for {len(missing_violations)} unaddressed violations."
            )
            return {"refinement_count": refinement_count + 1}

        if not self.orchestrator.is_available():
            error_msg = "LLM unavailable for refinement - accepting partial results"
            logger.error(error_msg)
            return {"refinement_count": refinement_count + 1}

        impact = state.get("impact_analysis")
        if impact is None:
            logger.warning("Impact analysis missing - skipping refinement")
            return {"refinement_count": refinement_count + 1}

        try:
            logger.info(f"Attempting refinement {refinement_count + 1}/{MAX_REFINEMENT_ATTEMPTS}")
            logger.info(f"Missing violations: {[v.pattern for v in missing_violations]}")

            all_violations = state.get("all_violations", [])
            feedback = self._create_refinement_feedback(
                missing_violations,
                existing_recommendations or [],
                all_violations
            )

            logger.info("=== REFINEMENT FEEDBACK ===")
            logger.info(f"Missing: {feedback['missing_violations_text']}")
            logger.info(f"Already addressed: {feedback['addressed_violations_text']}")

            result = self.orchestrator.invoke_with_schema(
                prompt=self.refinement_prompt,
                schema=RecommendationList,
                inputs={
                    "missing_violations_text": feedback["missing_violations_text"],
                    "addressed_violations_text": feedback["addressed_violations_text"],
                    "existing_recommendations_text": feedback["existing_recommendations_text"],
                    "impact": impact.model_dump_json(),
                },
            )

            if result is None or not result.recommendations:
                logger.error("Refinement LLM returned None or empty recommendations")
                return {"refinement_count": refinement_count + 1}

            new_recs = result.recommendations
            logger.info(f"Refinement generated {len(new_recs)} new recommendations")

            merged_recommendations = (existing_recommendations or []) + new_recs

            logger.info(f"Total recommendations after refinement: {len(merged_recommendations)}")

            new_message = HumanMessage(
                content=f"Refinement generated {len(new_recs)} additional recommendations"
            )

            return {
                "recommendations": merged_recommendations,
                "refinement_count": refinement_count + 1,
                "messages": [new_message]
            }

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return {"refinement_count": refinement_count + 1}

    def should_refine(self, state: AnalysisState) -> Literal["generate_missing_recommendations", "END"]:
        """Routing function to determine if refinement is needed.

        Args:
            state: Current analysis state

        Returns:
            "generate_missing_recommendations" if refinement needed, "END" otherwise
        """
        missing_violations = state.get("missing_violations")
        refinement_count = state.get("refinement_count", 0)

        if not missing_violations:
            logger.info("Routing: No missing violations - END")
            return "END"

        if refinement_count >= MAX_REFINEMENT_ATTEMPTS:
            logger.warning(
                f"Routing: Max refinement attempts reached ({refinement_count}/{MAX_REFINEMENT_ATTEMPTS}) - END"
            )
            return "END"

        logger.info(
            f"Routing: Refinement needed for {len(missing_violations)} violations "
            f"(attempt {refinement_count + 1}/{MAX_REFINEMENT_ATTEMPTS})"
        )
        return "generate_missing_recommendations"

    def _default_severity(self, state: AnalysisState) -> SeverityAssessment:
        """Fallback severity assessment when LLM unavailable.

        Args:
            state: Current analysis state

        Returns:
            Rule-based severity assessment
        """
        from cloud_ceo.llm.state import ViolationSeverityAssessment

        all_violations = state.get("all_violations", [state["violation"]])

        severity_map = {
            "CARTESIAN_JOIN": SeverityLevel.CRITICAL,
            "SELECT_STAR": SeverityLevel.HIGH,
            "MISSING_PARTITION_FILTER": SeverityLevel.HIGH,
            "NON_SARGABLE_PREDICATE": SeverityLevel.MEDIUM,
        }

        # Create per-violation assessments
        violation_assessments = []
        max_severity = SeverityLevel.LOW

        for v in all_violations:
            severity = severity_map.get(v.pattern, SeverityLevel.MEDIUM)
            if severity.value == "critical":
                max_severity = SeverityLevel.CRITICAL
            elif severity.value == "high" and max_severity.value != "critical":
                max_severity = SeverityLevel.HIGH
            elif severity.value == "medium" and max_severity.value not in ["critical", "high"]:
                max_severity = SeverityLevel.MEDIUM

            violation_assessments.append(
                ViolationSeverityAssessment(
                    violation_pattern=v.pattern,
                    severity=severity,
                    confidence=0.5,
                    evidence=f"Rule-based assessment for {v.pattern} (LLM unavailable)",
                )
            )

        return SeverityAssessment(
            violation_assessments=violation_assessments,
            overall_severity=max_severity,
            overall_confidence=0.5,
            overall_evidence="Rule-based assessment (LLM unavailable)",
            should_analyze=True,
        )

    def _is_actionable(self, rec: Recommendation) -> bool:
        """Check if recommendation is actionable.

        Args:
            rec: Recommendation to validate

        Returns:
            True if recommendation is actionable
        """
        return (
            len(rec.description) >= 20
            and len(rec.implementation) >= 30
            and rec.expected_improvement
        )

    def _is_valid_config(self, rec: Recommendation) -> bool:
        """Check if cluster config recommendation is valid.

        Args:
            rec: Recommendation to validate

        Returns:
            True if recommendation is valid
        """
        if rec.type != RecommendationType.CLUSTER_CONFIG:
            return True

        valid_node_types = {"i3.xlarge", "i3.2xlarge", "i3.4xlarge", "i3.8xlarge"}
        return any(node_type in rec.description for node_type in valid_node_types)

    def _ensure_all_violations_addressed(
        self,
        recommendations: List[Recommendation],
        all_violations: List[Violation]
    ) -> List[Recommendation]:
        """Validate that all violations have at least one recommendation addressing them.

        This is a validation step - if violations are missing, it logs a warning
        but does NOT generate fallback recommendations. The LLM should have addressed
        all violations.

        Args:
            recommendations: Current list of recommendations
            all_violations: All detected violations

        Returns:
            Original list of recommendations (unchanged)
        """
        addressed_patterns = set()
        for rec in recommendations:
            addressed_patterns.update(rec.violations_addressed)

        missing_violations = [
            v for v in all_violations
            if v.pattern not in addressed_patterns
        ]

        if missing_violations:
            missing_patterns = [v.pattern for v in missing_violations]
            logger.warning(
                f"LLM did not address {len(missing_violations)} violations: {missing_patterns}. "
                "This indicates the LLM prompt may need adjustment to ensure full coverage."
            )

        return recommendations

    def should_continue(self, state: AnalysisState) -> RoutingDecision:
        """Determine whether to continue analysis.

        Always proceeds to impact analysis when violations exist, regardless of severity.
        This ensures the workflow always generates recommendations for detected violations.

        Args:
            state: Current analysis state

        Returns:
            Always returns "analyze_impact" to ensure full workflow execution
        """
        assessment = state.get("severity_assessment")
        logger.info(
            f"Routing decision: Proceeding to impact analysis "
            f"(severity={assessment.severity if assessment else 'unknown'})"
        )
        return "analyze_impact"


def create_analysis_workflow(
    orchestrator: LLMOrchestrator,
    industry_mode: Optional[str] = None,
    enable_compression: bool = True
) -> StateGraph:
    """Build the LangGraph workflow with dependency injection.

    Args:
        orchestrator: Configured LLM orchestrator instance
        industry_mode: Optional industry compliance mode for enhanced analysis
        enable_compression: Enable context compression (default: True)

    Returns:
        Compiled StateGraph ready for execution
    """
    workflow_instance = AnalysisWorkflow(
        orchestrator,
        industry_mode=industry_mode,
        enable_compression=enable_compression
    )
    workflow = StateGraph(AnalysisState)

    # Add Node 0: Context Validator (compression)
    workflow.add_node("validate_context", workflow_instance.validate_context)

    # Add Nodes 1-4: Analysis workflow
    workflow.add_node("classify_severity", workflow_instance.classify_severity)
    workflow.add_node("analyze_impact", workflow_instance.analyze_impact)
    workflow.add_node(
        "generate_recommendations", workflow_instance.generate_recommendations
    )
    workflow.add_node(
        "validate_recommendations", workflow_instance.validate_recommendations
    )

    # Add Nodes 5-6: Refinement workflow
    workflow.add_node("check_coverage", workflow_instance.check_coverage)
    workflow.add_node(
        "generate_missing_recommendations", workflow_instance.generate_missing_recommendations
    )

    # Set Node 0 as entry point
    workflow.set_entry_point("validate_context")

    # Node 0 → Node 1 (always proceed to severity classification)
    workflow.add_edge("validate_context", "classify_severity")

    # Nodes 1-4: Original workflow
    workflow.add_conditional_edges(
        "classify_severity",
        workflow_instance.should_continue,
        {"analyze_impact": "analyze_impact", "END": END},
    )

    workflow.add_edge("analyze_impact", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "validate_recommendations")

    # Node 4 → Node 5: Check coverage after validation
    workflow.add_edge("validate_recommendations", "check_coverage")

    # Node 5 → Conditional: Refinement loop or END
    workflow.add_conditional_edges(
        "check_coverage",
        workflow_instance.should_refine,
        {
            "generate_missing_recommendations": "generate_missing_recommendations",
            "END": END
        },
    )

    # Node 6 → Node 4: Refinement loops back to validate new recommendations
    # Then check_coverage again (allows up to MAX_REFINEMENT_ATTEMPTS loops)
    workflow.add_edge("generate_missing_recommendations", "validate_recommendations")

    return workflow.compile()


def analyze_violation(
    violation: Violation,
    all_violations: List[Violation],
    execution_metrics: Optional[ExecutionMetrics] = None,
    cluster_config: Optional[ClusterConfig] = None,
    timeout: int = 10,
    orchestrator: Optional[LLMOrchestrator] = None,
    sql_query: Optional[str] = None,
    databricks_session: Optional[any] = None,
    industry_mode: Optional[str] = None,
    custom_guidelines_text: Optional[str] = None,
    enable_compression: bool = True,
) -> WorkflowResult:
    """Analyze violation and return recommendations with auto-fetched table schemas and industry-aware guidance.

    Public API for the analysis workflow with timeout handling
    and graceful degradation.

    Args:
        violation: Detected anti-pattern
        all_violations: All detected violations (for cross-reasoning)
        execution_metrics: Query performance data (optional)
        cluster_config: Current cluster config (optional)
        timeout: Max workflow duration in seconds
        orchestrator: Optional orchestrator instance (creates default if None)
        sql_query: Optional full SQL query text (for table schema extraction)
        databricks_session: Optional DatabricksSystemTables instance for schema fetching
        industry_mode: Optional industry compliance mode (uk_finance, uk_retail, uk_healthcare, pharma_biotech, uk_energy)
        custom_guidelines_text: Optional custom SQL guidelines text
        enable_compression: Enable context compression (default: True)

    Returns:
        WorkflowResult with recommendations or error
    """
    if orchestrator is None:
        orchestrator = LLMOrchestrator()

    try:
        context_builder = ContextBuilder()
        context = context_builder.build_context(
            violation=violation,
            all_violations=all_violations,
            execution_metrics=execution_metrics,
            cluster_config=cluster_config,
            sql_query=sql_query,
            databricks_session=databricks_session,
            custom_guidelines_text=custom_guidelines_text,
        )

        workflow = create_analysis_workflow(
            orchestrator,
            industry_mode=industry_mode,
            enable_compression=enable_compression
        )

        initial_state: AnalysisState = {
            "violation": violation,
            "context": context,
            "all_violations": all_violations,
            "severity_assessment": None,
            "impact_analysis": None,
            "recommendations": None,
            "violation_explanation": None,
            "messages": [],
            "compressed_context": None,
            "industry_mode": industry_mode,
            "custom_guidelines_text": custom_guidelines_text,
            "compression_enabled": enable_compression,
            "missing_violations": None,
            "refinement_count": 0,
            "addressed_violations": set(),
        }

        with timeout_context(timeout):
            result = workflow.invoke(initial_state)

        return WorkflowResult(
            recommendations=result.get("recommendations"),
            severity_assessment=result.get("severity_assessment"),
            impact_analysis=result.get("impact_analysis"),
            violation_explanation=result.get("violation_explanation"),
            error=None,
            partial=False,
        )

    except TimeoutError:
        logger.warning(f"Workflow timeout after {timeout}s")
        return WorkflowResult(
            recommendations=None,
            error=f"Workflow timeout after {timeout}s",
            partial=True,
        )
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return WorkflowResult(
            recommendations=None, error=f"Workflow failed: {str(e)}", partial=False,
        )
