"""LangGraph state management for analysis workflow.

This module defines the state schema and Pydantic models for the
multi-step LangGraph analysis workflow.
"""

from enum import Enum
from typing import Annotated, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from cloud_ceo.llm.context import AnalysisContext, Violation

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _is_notebook_environment() -> bool:
    """Detect if running in a notebook environment (Jupyter/Databricks).

    Returns:
        True if running in notebook, False otherwise
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return True  # Likely Databricks or other notebook
    except NameError:
        return False  # Not in IPython environment


def _is_databricks_environment() -> bool:
    """Detect if running specifically in Databricks notebook.

    Returns:
        True if running in Databricks, False otherwise
    """
    try:
        import os
        if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
            return True

        try:
            shell = get_ipython().__class__.__module__  # type: ignore
            if 'databricks' in shell.lower():
                return True
        except (NameError, AttributeError):
            pass

        return False
    except Exception:
        return False


class SeverityLevel(str, Enum):
    """Severity classification levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    """Types of recommendations."""

    CODE_FIX = "code_fix"
    QUERY_REWRITE = "query_rewrite"
    CLUSTER_CONFIG = "cluster_config"


class EffortLevel(str, Enum):
    """Implementation effort levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ViolationSeverityAssessment(BaseModel):
    """Severity assessment for a single violation.

    Attributes:
        violation_pattern: The violation pattern ID (e.g., SPARK_004)
        severity: Severity level for this specific violation
        confidence: Confidence score from 0.0 to 1.0
        evidence: Explanation of why this severity was assigned
    """

    violation_pattern: str = Field(
        description="Violation pattern ID (e.g., SPARK_004, SPARK_001)"
    )
    severity: SeverityLevel = Field(
        description="Severity level for this violation: critical, high, medium, low"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    evidence: str = Field(
        min_length=20, description="Detailed explanation for this violation's severity"
    )


class SeverityAssessment(BaseModel):
    """Output of classify_severity node.

    Provides both per-violation severity assessments and overall severity.

    Attributes:
        violation_assessments: List of per-violation severity assessments
        overall_severity: Overall severity level (highest among all violations)
        overall_confidence: Overall confidence score
        overall_evidence: Summary evidence explaining the overall severity
        should_analyze: Whether to proceed with full analysis
    """

    violation_assessments: List[ViolationSeverityAssessment] = Field(
        description="Per-violation severity assessments",
        min_length=1
    )
    overall_severity: SeverityLevel = Field(
        description="Overall severity level (highest among all violations)"
    )
    overall_confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence score from 0.0 to 1.0"
    )
    overall_evidence: str = Field(
        min_length=10, description="Summary explanation for overall severity assessment"
    )
    should_analyze: bool = Field(
        default=True,
        description="Whether to proceed with full analysis (skip low-severity violations)"
    )

    @property
    def severity(self) -> SeverityLevel:
        """Backward compatibility: return overall_severity."""
        return self.overall_severity

    @property
    def confidence(self) -> float:
        """Backward compatibility: return overall_confidence."""
        return self.overall_confidence

    @property
    def evidence(self) -> str:
        """Backward compatibility: return overall_evidence."""
        return self.overall_evidence

    def get_severity_color(self) -> str:
        """Get color code for severity level.

        Returns:
            Rich color name for the severity level
        """
        color_map = {
            SeverityLevel.CRITICAL: "red",
            SeverityLevel.HIGH: "orange1",
            SeverityLevel.MEDIUM: "yellow",
            SeverityLevel.LOW: "green",
        }
        return color_map.get(self.severity, "white")

    def get_severity_html_color(self) -> str:
        """Get HTML color code for severity level.

        Returns:
            HTML hex color for the severity level
        """
        color_map = {
            SeverityLevel.CRITICAL: "#dc2626",  # red-600
            SeverityLevel.HIGH: "#ea580c",      # orange-600
            SeverityLevel.MEDIUM: "#ca8a04",    # yellow-600
            SeverityLevel.LOW: "#16a34a",       # green-600
        }
        return color_map.get(self.severity, "#6b7280")

    def get_severity_icon(self) -> str:
        """Get icon for severity level.

        Returns:
            Unicode icon representing the severity
        """
        icon_map = {
            SeverityLevel.CRITICAL: "🔴",
            SeverityLevel.HIGH: "🟠",
            SeverityLevel.MEDIUM: "🟡",
            SeverityLevel.LOW: "🟢",
        }
        return icon_map.get(self.severity, "⚪")


class ViolationImpact(BaseModel):
    """Impact analysis for a single violation.

    Attributes:
        violation_pattern: The violation pattern ID (e.g., SPARK_004)
        performance_impact: Specific performance impact for this violation
        cost_contribution: This violation's contribution to total cost (if calculable)
        affected_resources: Resources affected by this specific violation
    """

    violation_pattern: str = Field(
        description="Violation pattern ID (e.g., SPARK_004, SPARK_001)"
    )
    performance_impact: str = Field(
        min_length=20,
        description="Detailed performance impact for this specific violation"
    )
    cost_contribution: Optional[float] = Field(
        default=None,
        description="This violation's contribution to daily cost in USD"
    )
    affected_resources: List[str] = Field(
        description="Resources affected by this violation (e.g., 'executors', 'memory', 'network')"
    )


class ImpactAnalysis(BaseModel):
    """Output of analyze_impact node.

    Provides both per-violation impact analysis and overall impact summary.

    Attributes:
        violation_impacts: List of per-violation impact analyses
        root_cause: Overall summary of primary causes
        cost_impact_usd: Total estimated daily cost in USD (if calculable)
        performance_impact: Overall query runtime impact description
        affected_resources: All affected resources across violations
    """

    violation_impacts: List[ViolationImpact] = Field(
        description="Per-violation impact analyses",
        min_length=1
    )
    root_cause: str = Field(
        min_length=20,
        description="Overall summary explaining primary causes across all violations"
    )
    cost_impact_usd: Optional[float] = Field(
        default=None,
        description="Total estimated daily cost impact in USD (sum of all violations)"
    )
    performance_impact: str = Field(
        min_length=20,
        description="Overall description of cumulative query runtime impact"
    )
    affected_resources: List[str] = Field(
        description="All affected resources across all violations (e.g., 'executors', 'memory', 'network')"
    )


class Recommendation(BaseModel):
    """Single recommendation option.

    Attributes:
        type: Type of recommendation (code_fix, query_rewrite, or cluster_config)
        violations_addressed: List of violation pattern IDs this recommendation fixes
        description: What to change
        expected_improvement: Expected benefit (e.g., '30% faster', '$50/day savings')
        effort_level: Implementation effort (low, medium, high)
        implementation: How to apply this recommendation
        user_explanation: User-friendly explanation of why this matters (optional)
    """

    type: RecommendationType = Field(description="Type of recommendation")
    violations_addressed: List[str] = Field(
        description="List of violation pattern IDs this recommendation addresses",
        default_factory=list
    )
    description: str = Field(
        min_length=20, description="What needs to be changed"
    )
    expected_improvement: str = Field(
        description="Expected benefit (e.g., '30% faster', '$50/day savings')"
    )
    effort_level: EffortLevel = Field(description="Implementation effort level")
    implementation: str = Field(
        min_length=30, description="Step-by-step instructions for applying this recommendation"
    )
    user_explanation: Optional[str] = Field(
        default=None,
        description="Plain-language explanation of why this recommendation matters (for Data Analysts)"
    )

    def model_post_init(self, __context) -> None:
        """Validate that CODE_FIX and QUERY_REWRITE recommendations have violations_addressed.

        Exception: Allow empty violations_addressed for "no issues" recommendations.
        """
        if self.type in [RecommendationType.CODE_FIX, RecommendationType.QUERY_REWRITE]:
            # Allow empty violations_addressed for informational "no issues" recommendations
            if not self.violations_addressed:
                # Check if this is a "no issues" recommendation
                is_no_issues = (
                    "no" in self.description.lower() and
                    ("issue" in self.description.lower() or "optimized" in self.description.lower())
                )
                if not is_no_issues:
                    raise ValueError(
                        f"Recommendation type '{self.type.value}' must have violations_addressed populated"
                    )


class RecommendationList(BaseModel):
    """Wrapper for a list of recommendations with violation explanation.

    This model is used for structured LLM output when generating multiple recommendations.

    Attributes:
        violation_explanation: User-friendly explanation of violations sorted by severity
        recommendations: List of generated recommendations
    """

    violation_explanation: str = Field(
        description="User-friendly explanation of all violations, sorted by severity (CRITICAL → LOW)",
        min_length=50
    )
    recommendations: List[Recommendation] = Field(
        description="List of ranked recommendations for addressing the violation",
        min_length=1,
        max_length=5
    )

    def get_effort_color(self) -> str:
        """Get color code for effort level.

        Returns:
            Rich color name for the effort level
        """
        color_map = {
            EffortLevel.LOW: "green",
            EffortLevel.MEDIUM: "yellow",
            EffortLevel.HIGH: "red",
        }
        return color_map.get(self.effort_level, "white")

    def get_effort_html_color(self) -> str:
        """Get HTML color code for effort level.

        Returns:
            HTML hex color for the effort level
        """
        color_map = {
            EffortLevel.LOW: "#16a34a",      # green-600
            EffortLevel.MEDIUM: "#ca8a04",   # yellow-600
            EffortLevel.HIGH: "#dc2626",     # red-600
        }
        return color_map.get(self.effort_level, "#6b7280")

    def get_type_icon(self) -> str:
        """Get icon for recommendation type.

        Returns:
            Unicode icon representing the recommendation type
        """
        icon_map = {
            RecommendationType.CODE_FIX: "🔧",
            RecommendationType.QUERY_REWRITE: "📝",
            RecommendationType.CLUSTER_CONFIG: "⚙️",
        }
        return icon_map.get(self.type, "💡")

class AnalysisState(TypedDict):
    """State passed between workflow nodes.

    This TypedDict defines the shared state that flows through the
    LangGraph workflow. Each node receives this state and can update
    specific fields by returning a partial dict.

    Attributes:
        violation: The current violation being analyzed
        context: Pre-formatted analysis context from ContextBuilder
        all_violations: All detected violations for cross-reasoning
        severity_assessment: Output from classify_severity node
        impact_analysis: Output from analyze_impact node
        recommendations: Output from generate_recommendations node
        violation_explanation: User-friendly explanation of violations (from LLM)
        messages: Message history for token optimization (uses add_messages reducer)
        compressed_context: Compressed context from Context Validator (if available)
        industry_mode: Optional industry compliance mode (uk_finance, uk_retail, etc.)
        custom_guidelines_text: Optional custom SQL guidelines text
        compression_enabled: Whether context compression is enabled (default: True)
        missing_violations: Violations not addressed by current recommendations
        refinement_count: Number of refinement attempts (max: MAX_REFINEMENT_ATTEMPTS)
        addressed_violations: Set of violation patterns already addressed by recommendations
    """

    violation: Violation
    context: AnalysisContext
    all_violations: List[Violation]
    severity_assessment: Optional[SeverityAssessment]
    impact_analysis: Optional[ImpactAnalysis]
    recommendations: Optional[List[Recommendation]]
    violation_explanation: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    compressed_context: Optional[str]
    industry_mode: Optional[str]
    custom_guidelines_text: Optional[str]
    compression_enabled: bool
    missing_violations: Optional[List[Violation]]
    refinement_count: int
    addressed_violations: set[str]


class WorkflowResult(BaseModel):
    """Result of the analysis workflow.

    This is returned by the public API to provide structured results
    with error handling and partial result support.

    Attributes:
        recommendations: List of recommendations (if analysis succeeded)
        severity_assessment: Severity assessment (if classify node succeeded)
        impact_analysis: Impact analysis (if analyze node succeeded)
        violation_explanation: User-friendly explanation of violations sorted by severity
        error: Error message (if workflow failed)
        partial: Whether this is a partial result from workflow timeout
    """

    recommendations: Optional[List[Recommendation]] = Field(
        default=None, description="Generated recommendations"
    )
    severity_assessment: Optional[SeverityAssessment] = Field(
        default=None, description="Severity classification"
    )
    impact_analysis: Optional[ImpactAnalysis] = Field(
        default=None, description="Impact analysis results"
    )
    violation_explanation: Optional[str] = Field(
        default=None, description="User-friendly explanation of violations (sorted by severity)"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    partial: bool = Field(
        default=False, description="True if this is a partial result due to timeout"
    )

    def __str__(self) -> str:
        """Fallback string representation for environments without rich.

        Display order: Severity → Violation Explanation → Impact → Recommendations

        Returns:
            Plain text summary of the analysis result
        """
        if self.error:
            status = "[PARTIAL] " if self.partial else ""
            return f"{status}ERROR: {self.error}"

        lines = ["=" * 80, "Analysis Result Summary", "=" * 80]

        if self.severity_assessment:
            sev = self.severity_assessment
            lines.append(
                f"\nSeverity: {sev.severity.value.upper()} "
                f"(Confidence: {sev.confidence:.0%})"
            )
            lines.append(f"Evidence: {sev.evidence}")

        if self.violation_explanation:
            lines.append(f"\n--- What's Wrong & Why It Matters ---")
            lines.append(self.violation_explanation)

        if self.impact_analysis:
            impact = self.impact_analysis
            lines.append(f"\n--- Impact Analysis ---")
            lines.append(f"Root Cause: {impact.root_cause}")
            if impact.cost_impact_usd:
                lines.append(f"Cost Impact: ${impact.cost_impact_usd:.2f}/day")
            lines.append(f"Performance Impact: {impact.performance_impact}")
            if impact.affected_resources:
                lines.append(
                    f"Affected Resources: {', '.join(impact.affected_resources)}"
                )

        if self.recommendations:
            lines.append(f"\n--- {len(self.recommendations)} Recommendation(s) ---")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"\n{i}. {rec.description}")
                lines.append(f"   Type: {rec.type.value}")
                if rec.violations_addressed:
                    lines.append(f"   Fixes: {', '.join(rec.violations_addressed)}")
                lines.append(f"   Expected Improvement: {rec.expected_improvement}")
                lines.append(f"   Effort: {rec.effort_level.value}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def print_summary(self, console: Optional["Console"] = None) -> None:
        """Print a quick summary view (severity + recommendation count).

        This is ideal for scanning multiple violations in sequence.
        Automatically detects environment (terminal vs notebook) and uses
        appropriate rendering method.

        Args:
            console: Optional Rich Console instance. Creates default if None.
                    Only used in terminal environments.
        """
        if _is_notebook_environment():
            self._display_summary_html()
        elif RICH_AVAILABLE:
            self._print_summary_rich(console)
        else:
            print(self.__str__())

    def print_detailed(self, console: Optional["Console"] = None) -> None:
        """Print comprehensive analysis results with all details.

        This is ideal for interactive investigation of a single violation.
        Automatically detects environment (terminal vs notebook) and uses
        appropriate rendering method.

        Args:
            console: Optional Rich Console instance. Creates default if None.
                    Only used in terminal environments.
        """
        if _is_notebook_environment():
            self._display_detailed_html()
        elif RICH_AVAILABLE:
            self._print_detailed_rich(console)
        else:
            print(self.__str__())

    def _print_summary_rich(self, console: Optional["Console"] = None) -> None:
        """Print summary using Rich library (terminal environments).

        Args:
            console: Optional Rich Console instance
        """
        if console is None:
            console = Console()

        if self.error:
            console.print(
                Panel(
                    f"[red]ERROR:[/red] {self.error}",
                    title="Analysis Failed" + (" (Partial)" if self.partial else ""),
                    border_style="red",
                )
            )
            return

        summary_parts = []

        if self.severity_assessment:
            sev = self.severity_assessment
            severity_text = Text()
            severity_text.append(f"{sev.get_severity_icon()} ", style="bold")
            severity_text.append(
                sev.severity.value.upper(), style=f"bold {sev.get_severity_color()}"
            )
            severity_text.append(
                f" ({sev.confidence:.0%} confidence)", style="dim"
            )
            summary_parts.append(severity_text)

        if self.impact_analysis and self.impact_analysis.cost_impact_usd:
            cost_text = Text()
            cost_text.append("💰 ", style="bold")
            cost_text.append(
                f"${self.impact_analysis.cost_impact_usd:.2f}/day",
                style="bold yellow",
            )
            summary_parts.append(cost_text)

        if self.recommendations:
            rec_text = Text()
            rec_text.append("💡 ", style="bold")
            rec_text.append(
                f"{len(self.recommendations)} recommendation(s)", style="bold cyan"
            )
            summary_parts.append(rec_text)

        if summary_parts:
            console.print(
                Panel(
                    Text("  |  ").join(summary_parts),
                    title="Quick Summary",
                    border_style="cyan",
                )
            )
        else:
            console.print("[dim]No analysis results available[/dim]")

    def _print_detailed_rich(self, console: Optional["Console"] = None) -> None:
        """Print detailed results using Rich library (terminal environments).

        Display order: Severity → Violation Explanation → Impact → Recommendations

        Args:
            console: Optional Rich Console instance
        """
        if console is None:
            console = Console()

        if self.error:
            console.print(
                Panel(
                    f"[red bold]ERROR:[/red bold] {self.error}",
                    title="❌ Analysis Failed"
                    + (" (Partial Result)" if self.partial else ""),
                    border_style="red",
                )
            )
            return

        if self.severity_assessment:
            self._print_severity_section(console)

        if self.violation_explanation:
            self._print_violation_explanation_section(console)

        if self.impact_analysis:
            self._print_impact_section(console)

        if self.recommendations:
            self._print_recommendations_section(console)
        else:
            console.print(
                Panel(
                    "[dim]No recommendations available[/dim]",
                    title="💡 Recommendations",
                    border_style="dim",
                )
            )

    def _display_summary_html(self) -> None:
        """Display summary using HTML (notebook environments)."""
        try:
            from IPython.display import display, HTML
        except ImportError:
            print(self.__str__())
            return

        if self.error:
            status = "⚠️ PARTIAL RESULT" if self.partial else "❌ ERROR"
            html = f"""
            <div style="border: 2px solid #dc2626; border-radius: 8px; padding: 16px;
                        margin: 10px 0; background-color: #fef2f2;">
                <div style="font-size: 16px; font-weight: bold; color: #dc2626;
                           margin-bottom: 8px;">{status}</div>
                <div style="color: #991b1b;">{self.error}</div>
            </div>
            """
            display(HTML(html))
            return

        parts = []

        if self.severity_assessment:
            sev = self.severity_assessment
            color = sev.get_severity_html_color()
            parts.append(f"""
                <div style="display: inline-block; margin-right: 20px;">
                    <span style="font-size: 16px;">{sev.get_severity_icon()}</span>
                    <span style="font-weight: bold; color: {color};">
                        {sev.severity.value.upper()}
                    </span>
                    <span style="color: #6b7280; font-size: 14px;">
                        ({sev.confidence:.0%} confidence)
                    </span>
                </div>
            """)

        if self.impact_analysis and self.impact_analysis.cost_impact_usd:
            parts.append(f"""
                <div style="display: inline-block; margin-right: 20px;">
                    <span style="font-size: 16px;">💰</span>
                    <span style="font-weight: bold; color: #ca8a04;">
                        ${self.impact_analysis.cost_impact_usd:.2f}/day
                    </span>
                </div>
            """)

        if self.recommendations:
            parts.append(f"""
                <div style="display: inline-block;">
                    <span style="font-size: 16px;">💡</span>
                    <span style="font-weight: bold; color: #0891b2;">
                        {len(self.recommendations)} recommendation(s)
                    </span>
                </div>
            """)

        if parts:
            html = f"""
            <div style="border: 2px solid #0891b2; border-radius: 8px; padding: 16px;
                        margin: 10px 0; background-color: #f0f9ff;">
                <div style="font-size: 14px; font-weight: bold; color: #0891b2;
                           margin-bottom: 12px;">Quick Summary</div>
                <div style="display: flex; flex-wrap: wrap; align-items: center;">
                    {''.join(parts)}
                </div>
            </div>
            """
        else:
            html = """
            <div style="border: 2px solid #d1d5db; border-radius: 8px; padding: 16px;
                        margin: 10px 0; background-color: #f9fafb;">
                <div style="color: #6b7280;">No analysis results available</div>
            </div>
            """

        display(HTML(html))

    def _display_detailed_html(self) -> None:
        """Display detailed results using HTML (notebook environments).

        Display order: Severity → Violation Explanation → Impact → Recommendations
        """
        try:
            from IPython.display import display, HTML
        except ImportError:
            print(self.__str__())
            return

        if self.error:
            status = "⚠️ PARTIAL RESULT" if self.partial else "❌ ERROR"
            html = f"""
            <div style="border: 2px solid #dc2626; border-radius: 8px; padding: 16px;
                        margin: 10px 0; background-color: #fef2f2;">
                <div style="font-size: 18px; font-weight: bold; color: #dc2626;
                           margin-bottom: 8px;">{status}</div>
                <div style="color: #991b1b; font-size: 14px;">{self.error}</div>
            </div>
            """
            display(HTML(html))
            return

        sections = []

        if self.severity_assessment:
            sections.append(self._build_severity_html())

        if self.violation_explanation:
            sections.append(self._build_violation_explanation_html())

        if self.impact_analysis:
            sections.append(self._build_impact_html())

        if self.recommendations:
            sections.extend(self._build_recommendations_html())
        else:
            sections.append("""
                <div style="border: 2px solid #d1d5db; border-radius: 8px; padding: 16px;
                            margin: 10px 0; background-color: #f9fafb;">
                    <div style="font-size: 16px; font-weight: bold; color: #374151;
                               margin-bottom: 8px;">💡 Recommendations</div>
                    <div style="color: #6b7280;">No recommendations available</div>
                </div>
            """)

        display(HTML(''.join(sections)))

    def _build_severity_html(self) -> str:
        """Build HTML for severity section.

        Returns:
            HTML string for severity assessment
        """
        if not self.severity_assessment:
            return ""

        sev = self.severity_assessment
        color = sev.get_severity_html_color()

        return f"""
        <div style="border: 2px solid {color}; border-radius: 8px; padding: 16px;
                    margin: 10px 0; background-color: #fefefe;">
            <div style="font-size: 16px; font-weight: bold; color: {color};
                       margin-bottom: 12px;">
                🎯 Severity Assessment
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-weight: bold;">Severity:</span>
                <span style="font-size: 16px; margin-left: 4px;">{sev.get_severity_icon()}</span>
                <span style="font-weight: bold; color: {color}; margin-left: 4px;">
                    {sev.severity.value.upper()}
                </span>
                <span style="color: #6b7280; font-size: 14px; margin-left: 8px;">
                    (Confidence: {sev.confidence:.0%})
                </span>
            </div>
            <div style="color: #374151; line-height: 1.6;">
                {sev.evidence}
            </div>
        </div>
        """

    def _build_violation_explanation_html(self) -> str:
        """Build HTML for violation explanation section.

        Returns:
            HTML string for violation explanation
        """
        if not self.violation_explanation:
            return ""

        import html
        explanation_html = html.escape(self.violation_explanation).replace('\n', '<br>')

        return f"""
        <div style="border: 2px solid #0891b2; border-radius: 8px; padding: 16px;
                    margin: 10px 0; background-color: #fefefe;">
            <div style="font-size: 16px; font-weight: bold; color: #0891b2;
                       margin-bottom: 12px;">
                📋 What's Wrong & Why It Matters
            </div>
            <div style="color: #374151; line-height: 1.8; white-space: pre-wrap;">
                {explanation_html}
            </div>
        </div>
        """

    def _build_impact_html(self) -> str:
        """Build HTML for impact section.

        Returns:
            HTML string for impact analysis
        """
        if not self.impact_analysis:
            return ""

        impact = self.impact_analysis

        rows = [
            f"""
            <tr>
                <td style="padding: 8px; font-weight: bold; color: #0891b2;
                           white-space: nowrap;">Root Cause</td>
                <td style="padding: 8px; color: #374151;">{impact.root_cause}</td>
            </tr>
            """
        ]

        if impact.cost_impact_usd:
            rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: bold; color: #0891b2;
                           white-space: nowrap;">Cost Impact</td>
                <td style="padding: 8px; font-weight: bold; color: #ca8a04;">
                    ${impact.cost_impact_usd:.2f}/day
                </td>
            </tr>
            """)

        rows.append(f"""
        <tr>
            <td style="padding: 8px; font-weight: bold; color: #0891b2;
                       white-space: nowrap;">Performance</td>
            <td style="padding: 8px; color: #374151;">{impact.performance_impact}</td>
        </tr>
        """)

        if impact.affected_resources:
            rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: bold; color: #0891b2;
                           white-space: nowrap; vertical-align: top;">Affected Resources</td>
                <td style="padding: 8px; color: #374151;">
                    {', '.join(impact.affected_resources)}
                </td>
            </tr>
            """)

        return f"""
        <div style="border: 2px solid #ca8a04; border-radius: 8px; padding: 16px;
                    margin: 10px 0; background-color: #fefefe;">
            <div style="font-size: 16px; font-weight: bold; color: #ca8a04;
                       margin-bottom: 12px;">
                📊 Impact Analysis
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                {''.join(rows)}
            </table>
        </div>
        """

    def _build_recommendations_html(self) -> List[str]:
        """Build HTML for recommendations section.

        Returns:
            List of HTML strings for each recommendation
        """
        if not self.recommendations:
            return []

        html_sections = []

        for i, rec in enumerate(self.recommendations, 1):
            effort_color = rec.get_effort_html_color()

            violations_row = ""
            if rec.violations_addressed:
                violations_list = ", ".join(rec.violations_addressed)
                violations_row = f"""
                    <tr>
                        <td style="padding: 8px; font-weight: bold; color: #0891b2;
                                   white-space: nowrap; vertical-align: top;">Fixes Violations</td>
                        <td style="padding: 8px; color: #374151;">
                            <span style="background-color: #f0f9ff; padding: 4px 8px;
                                       border-radius: 4px; font-family: monospace; font-size: 12px;">
                                {violations_list}
                            </span>
                        </td>
                    </tr>
                """

            user_explanation_row = ""
            if rec.user_explanation:
                import html as html_module
                explanation_html = html_module.escape(rec.user_explanation)
                user_explanation_row = f"""
                    <tr>
                        <td colspan="2" style="padding: 16px 8px 8px 8px;">
                            <div style="font-weight: bold; color: #0891b2; margin-bottom: 8px;">
                                Why This Matters:
                            </div>
                            <div style="color: #374151; line-height: 1.6;
                                       white-space: pre-wrap;">{explanation_html}</div>
                        </td>
                    </tr>
                """

            html = f"""
            <div style="border: 2px solid #16a34a; border-radius: 8px; padding: 16px;
                        margin: 10px 0; background-color: #fefefe;">
                <div style="font-size: 16px; font-weight: bold; color: #16a34a;
                           margin-bottom: 12px;">
                    {rec.get_type_icon()} Recommendation {i}/{len(self.recommendations)}
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; font-weight: bold; color: #0891b2;
                                   white-space: nowrap; width: 150px;">Type</td>
                        <td style="padding: 8px; color: #374151;">
                            {rec.type.value.replace('_', ' ').title()}
                        </td>
                    </tr>
                    {violations_row}
                    <tr>
                        <td style="padding: 8px; font-weight: bold; color: #0891b2;
                                   white-space: nowrap; vertical-align: top;">Description</td>
                        <td style="padding: 8px; color: #374151;">{rec.description}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; color: #0891b2;
                                   white-space: nowrap;">Expected Improvement</td>
                        <td style="padding: 8px; font-weight: bold; color: #16a34a;">
                            {rec.expected_improvement}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: bold; color: #0891b2;
                                   white-space: nowrap;">Effort Level</td>
                        <td style="padding: 8px; font-weight: bold; color: {effort_color};">
                            {rec.effort_level.value.upper()}
                        </td>
                    </tr>
                    {user_explanation_row}
                    <tr>
                        <td colspan="2" style="padding: 16px 8px 0px 8px;">
                            <div style="font-weight: bold; color: #0891b2; margin-bottom: 8px;">
                                Implementation:
                            </div>
                            <div style="color: #374151; line-height: 1.6;
                                       white-space: pre-wrap;">{rec.implementation}</div>
                        </td>
                    </tr>
                </table>
            </div>
            """
            html_sections.append(html)

        return html_sections

    def _print_severity_section(self, console: "Console") -> None:
        """Print severity assessment section.

        Args:
            console: Rich Console instance
        """
        if not self.severity_assessment:
            return

        sev = self.severity_assessment

        severity_text = Text()
        severity_text.append(f"{sev.get_severity_icon()} ", style="bold")
        severity_text.append(
            sev.severity.value.upper(),
            style=f"bold {sev.get_severity_color()}",
        )
        severity_text.append(f" (Confidence: {sev.confidence:.0%})", style="dim")

        content = Text.assemble(
            ("Severity: ", "bold"), severity_text, "\n\n", sev.evidence
        )

        console.print(
            Panel(
                content,
                title="🎯 Severity Assessment",
                border_style=sev.get_severity_color(),
            )
        )

    def _print_violation_explanation_section(self, console: "Console") -> None:
        """Print violation explanation section.

        Args:
            console: Rich Console instance
        """
        if not self.violation_explanation:
            return

        console.print(
            Panel(
                Markdown(self.violation_explanation),
                title="📋 What's Wrong & Why It Matters",
                border_style="cyan",
            )
        )

    def _print_impact_section(self, console: "Console") -> None:
        """Print impact analysis section.

        Args:
            console: Rich Console instance
        """
        if not self.impact_analysis:
            return

        impact = self.impact_analysis

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="cyan bold", no_wrap=True)
        table.add_column("Value")

        table.add_row("Root Cause", impact.root_cause)

        if impact.cost_impact_usd:
            table.add_row(
                "Cost Impact", f"[yellow bold]${impact.cost_impact_usd:.2f}/day[/]"
            )

        table.add_row("Performance", impact.performance_impact)

        if impact.affected_resources:
            table.add_row(
                "Affected Resources", ", ".join(impact.affected_resources)
            )

        console.print(Panel(table, title="📊 Impact Analysis", border_style="yellow"))

    def _print_recommendations_section(self, console: "Console") -> None:
        """Print recommendations section with ranked options.

        Args:
            console: Rich Console instance
        """
        if not self.recommendations:
            return

        for i, rec in enumerate(self.recommendations, 1):
            title_text = Text()
            title_text.append(f"{rec.get_type_icon()} ", style="bold")
            title_text.append(f"Recommendation {i}/{len(self.recommendations)}")

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Field", style="cyan bold", no_wrap=True)
            table.add_column("Value")

            table.add_row("Type", rec.type.value.replace("_", " ").title())

            if rec.violations_addressed:
                violations_text = Text(", ".join(rec.violations_addressed), style="dim")
                table.add_row("Fixes Violations", violations_text)

            table.add_row("Description", rec.description)
            table.add_row(
                "Expected Improvement",
                f"[green bold]{rec.expected_improvement}[/]",
            )

            effort_text = Text()
            effort_text.append(
                rec.effort_level.value.upper(),
                style=f"bold {rec.get_effort_color()}",
            )
            table.add_row("Effort Level", effort_text)

            if rec.user_explanation:
                table.add_row("", "")
                table.add_row(
                    "[cyan bold]Why This Matters:",
                    rec.user_explanation
                )

            table.add_row("", "")
            table.add_row(
                "[cyan bold]Implementation:",
                Markdown(rec.implementation) if rec.implementation else "[dim]N/A[/]",
            )

            console.print(
                Panel(table, title=title_text, border_style="green", padding=(1, 2))
            )

            if i < len(self.recommendations):
                console.print()

    def to_markdown(self) -> str:
        """Export result as markdown for documentation or sharing.

        Display order: Severity → Violation Explanation → Impact → Recommendations

        Returns:
            Markdown-formatted string of the analysis result
        """
        if self.error:
            status = "⚠️ PARTIAL RESULT" if self.partial else "❌ ERROR"
            return f"# {status}\n\n{self.error}"

        lines = ["# Analysis Result\n"]

        if self.severity_assessment:
            sev = self.severity_assessment
            lines.append("## 🎯 Severity Assessment\n")
            lines.append(
                f"**{sev.get_severity_icon()} Severity:** {sev.severity.value.upper()}\n"
            )
            lines.append(f"**Confidence:** {sev.confidence:.0%}\n")
            lines.append(f"**Evidence:** {sev.evidence}\n\n")

        if self.violation_explanation:
            lines.append("## 📋 What's Wrong & Why It Matters\n")
            lines.append(f"{self.violation_explanation}\n\n")

        if self.impact_analysis:
            impact = self.impact_analysis
            lines.append("## 📊 Impact Analysis\n")
            lines.append(f"**Root Cause:** {impact.root_cause}\n")
            if impact.cost_impact_usd:
                lines.append(
                    f"**Cost Impact:** ${impact.cost_impact_usd:.2f}/day\n"
                )
            lines.append(f"**Performance Impact:** {impact.performance_impact}\n")
            if impact.affected_resources:
                lines.append(
                    f"**Affected Resources:** {', '.join(impact.affected_resources)}\n"
                )
            lines.append("\n")

        if self.recommendations:
            lines.append(f"## 💡 Recommendations ({len(self.recommendations)})\n")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"### {i}. {rec.get_type_icon()} {rec.description}\n")
                lines.append(
                    f"- **Type:** {rec.type.value.replace('_', ' ').title()}\n"
                )
                if rec.violations_addressed:
                    lines.append(
                        f"- **Fixes Violations:** {', '.join(rec.violations_addressed)}\n"
                    )
                lines.append(
                    f"- **Expected Improvement:** {rec.expected_improvement}\n"
                )
                lines.append(f"- **Effort:** {rec.effort_level.value}\n")
                if rec.user_explanation:
                    lines.append(f"\n**Why This Matters:**\n{rec.user_explanation}\n")
                lines.append(f"\n**Implementation:**\n\n{rec.implementation}\n\n")

        return "\n".join(lines)


# Type alias for routing decisions
RoutingDecision = Literal["analyze_impact", "END"]
