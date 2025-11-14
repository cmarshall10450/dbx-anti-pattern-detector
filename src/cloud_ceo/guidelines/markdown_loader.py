"""Load and process unstructured Markdown guideline documents.

This module provides flexible loading of company documentation in Markdown format,
extracting relevant guidelines without requiring rigid structure.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from cloud_ceo.rule_engine.exceptions import validate_file_size


@dataclass
class MarkdownSection:
    """Extracted section from a Markdown document.

    Attributes:
        heading: Section heading text
        level: Heading level (1-6)
        content: Full content of this section (including subsections)
        priority_keywords: Keywords indicating importance (CRITICAL, HIGH, etc.)
        code_blocks: SQL code blocks found in this section
        affected_tables: Table/column names mentioned in this section
        line_start: Starting line number in source document
        line_end: Ending line number in source document
    """

    heading: str
    level: int
    content: str
    priority_keywords: Set[str]
    code_blocks: List[str]
    affected_tables: Set[str]
    line_start: int
    line_end: int

    @property
    def priority_score(self) -> int:
        """Calculate priority score based on keywords and structure.

        Returns:
            Priority score (lower = higher priority)
        """
        # Find the highest priority (lowest score) among all keywords
        min_score = 40  # Default for no priority keywords

        for keyword in self.priority_keywords:
            if keyword in ("CRITICAL", "MUST", "REQUIRED", "MANDATORY"):
                min_score = min(min_score, 10)
            elif keyword in ("HIGH PRIORITY", "IMPORTANT", "URGENT", "SEVERE", "WARNING"):
                min_score = min(min_score, 20)
            elif keyword in ("MEDIUM", "SHOULD"):
                min_score = min(min_score, 30)

        return min_score

    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens in this section.

        Returns:
            Estimated token count (content_chars / 4)
        """
        return len(self.content) // 4


class MarkdownGuidelineLoader:
    """Load unstructured Markdown documentation as custom guidelines.

    This loader:
    1. Accepts ANY Markdown document (no structure required)
    2. Intelligently extracts sections and metadata
    3. Prioritizes content based on keywords and code examples
    4. Manages token budgets for LLM context
    5. Handles multiple files in a directory

    Design Philosophy:
    - Zero friction: Users drop in existing docs with no modification
    - Smart extraction: Leverage LLM's NLP understanding
    - Token efficient: Filter and chunk based on relevance
    - Graceful degradation: Works even with poorly structured markdown
    """

    # Keywords that indicate priority/severity
    PRIORITY_KEYWORDS = {
        "CRITICAL", "HIGH PRIORITY", "IMPORTANT", "MUST", "REQUIRED",
        "MANDATORY", "SEVERE", "URGENT", "WARNING"
    }

    # Pattern for extracting table names (simple heuristic)
    TABLE_PATTERN = re.compile(
        r'\b(?:FROM|JOIN|INTO|TABLE)\s+([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*)',
        re.IGNORECASE
    )

    def __init__(self, max_file_size_mb: int = 10) -> None:
        """Initialize markdown guideline loader.

        Args:
            max_file_size_mb: Maximum allowed file size in MB (default: 10)
        """
        self._sections: List[MarkdownSection] = []
        self._raw_content: str = ""
        self._file_paths: List[Path] = []
        self._max_file_size_mb = max_file_size_mb

    def load_from_file(self, file_path: Path) -> "MarkdownGuidelineLoader":
        """Load guidelines from a Markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Self for chaining

        Raises:
            FileNotFoundError: If file doesn't exist
            FileSizeLimitExceeded: If file exceeds size limit
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Guideline file not found: {file_path}")

        validate_file_size(file_path, self._max_file_size_mb)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        self._raw_content = content
        self._file_paths.append(file_path)
        self._parse_markdown(content)

        return self

    def load_from_directory(
        self,
        directory: Path,
        recursive: bool = True,
        pattern: str = "*.md"
    ) -> "MarkdownGuidelineLoader":
        """Load all markdown files from a directory.

        Args:
            directory: Directory containing markdown files
            recursive: Whether to search subdirectories
            pattern: Glob pattern for files (default: *.md)

        Returns:
            Self for chaining

        Raises:
            FileNotFoundError: If directory doesn't exist
        """
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        glob_method = directory.rglob if recursive else directory.glob

        for file_path in glob_method(pattern):
            if file_path.is_file():
                try:
                    self.load_from_file(file_path)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue

        return self

    def _parse_markdown(self, content: str) -> None:
        """Parse markdown content into sections.

        Args:
            content: Markdown content to parse
        """
        lines = content.split("\n")
        current_section: Optional[dict] = None
        line_num = 0

        for i, line in enumerate(lines):
            # Detect headings
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if heading_match:
                level = len(heading_match.group(1))

                # Skip level 1 headings (document titles)
                if level == 1:
                    continue

                # Only create new sections for level 2 headings
                # Level 3+ headings are treated as part of the parent section
                if level == 2:
                    # Save previous section
                    if current_section:
                        self._finalize_section(current_section, i - 1)

                    # Start new section
                    heading = heading_match.group(2).strip()

                    current_section = {
                        "heading": heading,
                        "level": level,
                        "content_lines": [],
                        "line_start": i,
                    }
                elif current_section:
                    # Level 3+ headings are part of the current section's content
                    current_section["content_lines"].append(line)
            elif current_section:
                # Add line to current section
                current_section["content_lines"].append(line)

        # Finalize last section
        if current_section:
            self._finalize_section(current_section, len(lines) - 1)

    def _finalize_section(self, section_dict: dict, line_end: int) -> None:
        """Convert section dict to MarkdownSection and extract metadata.

        Args:
            section_dict: Section data being built
            line_end: Ending line number
        """
        content = "\n".join(section_dict["content_lines"])

        # Extract priority keywords from both heading and content
        priority_keywords = set()
        heading_upper = section_dict["heading"].upper()
        content_upper = content.upper()
        for keyword in self.PRIORITY_KEYWORDS:
            if keyword in heading_upper or keyword in content_upper:
                priority_keywords.add(keyword)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)

        # Extract table references
        affected_tables = set()
        for code_block in code_blocks:
            tables = self.TABLE_PATTERN.findall(code_block)
            affected_tables.update(t.lower() for t in tables)

        section = MarkdownSection(
            heading=section_dict["heading"],
            level=section_dict["level"],
            content=content,
            priority_keywords=priority_keywords,
            code_blocks=code_blocks,
            affected_tables=affected_tables,
            line_start=section_dict["line_start"],
            line_end=line_end,
        )

        self._sections.append(section)

    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract SQL code blocks from markdown content.

        Args:
            content: Markdown content

        Returns:
            List of code blocks (without backticks)
        """
        # Match ```sql ... ``` or ``` ... ```
        pattern = re.compile(r'```(?:sql)?\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(content)
        return [m.strip() for m in matches if m.strip()]

    def get_all_content(self, max_tokens: Optional[int] = None) -> str:
        """Get all markdown content (optionally truncated).

        Args:
            max_tokens: Optional token limit

        Returns:
            Markdown content as string
        """
        if max_tokens is None:
            return self._raw_content

        # Rough token estimate: 4 chars per token
        max_chars = max_tokens * 4
        if len(self._raw_content) <= max_chars:
            return self._raw_content

        # Truncate with warning
        truncated = self._raw_content[:max_chars]
        return truncated + f"\n\n[Content truncated at {max_tokens} tokens]"

    def get_relevant_sections(
        self,
        query_tables: Optional[List[str]] = None,
        max_tokens: int = 2000,
        include_all_critical: bool = True
    ) -> str:
        """Get relevant sections based on query context.

        Args:
            query_tables: Tables mentioned in the query being analyzed
            max_tokens: Maximum tokens to return
            include_all_critical: Always include CRITICAL sections

        Returns:
            Formatted markdown with most relevant sections
        """
        if not self._sections:
            return self.get_all_content(max_tokens)

        # Score and sort sections
        scored_sections = []
        for section in self._sections:
            score = self._score_section(section, query_tables)
            scored_sections.append((score, section))

        scored_sections.sort(key=lambda x: x[0])  # Lower score = higher priority

        # Build output within token budget
        selected_sections = []
        token_count = 0

        # First pass: Include all critical sections
        if include_all_critical:
            for score, section in scored_sections:
                if section.priority_score <= 10:  # CRITICAL
                    if token_count + section.token_estimate <= max_tokens:
                        selected_sections.append(section)
                        token_count += section.token_estimate

        # Second pass: Add remaining sections by priority
        for score, section in scored_sections:
            if section not in selected_sections:
                if token_count + section.token_estimate <= max_tokens:
                    selected_sections.append(section)
                    token_count += section.token_estimate
                else:
                    break  # Exceeded budget

        # Format output
        return self._format_sections(selected_sections)

    def _score_section(
        self,
        section: MarkdownSection,
        query_tables: Optional[List[str]]
    ) -> int:
        """Score section for relevance (lower = more relevant).

        Args:
            section: Section to score
            query_tables: Tables from query context

        Returns:
            Relevance score (lower = higher priority)
        """
        score = section.priority_score

        # Boost if tables match query
        if query_tables and section.affected_tables:
            query_tables_lower = {t.lower() for t in query_tables}
            if section.affected_tables & query_tables_lower:
                score -= 15  # Strong boost

        # Boost if has code examples
        if section.code_blocks:
            score -= 5

        return score

    def _format_sections(self, sections: List[MarkdownSection]) -> str:
        """Format sections into readable markdown.

        Args:
            sections: Sections to format

        Returns:
            Formatted markdown string
        """
        if not sections:
            return ""

        parts = []

        for section in sections:
            # Recreate heading
            heading_prefix = "#" * section.level
            parts.append(f"{heading_prefix} {section.heading}\n")
            parts.append(section.content)
            parts.append("\n")

        return "\n".join(parts)

    def get_compact_summary(self, max_items: int = 10) -> str:
        """Get compact summary of guidelines for context display.

        Args:
            max_items: Maximum number of items to include

        Returns:
            Compact summary string
        """
        if not self._sections:
            return "Custom guidelines loaded from markdown"

        # Get top priority sections
        sorted_sections = sorted(self._sections, key=lambda s: s.priority_score)
        top_sections = sorted_sections[:max_items]

        lines = ["**Custom Guidelines:**"]
        for section in top_sections:
            priority = "CRITICAL" if section.priority_score <= 10 else "HIGH" if section.priority_score <= 20 else "MEDIUM"
            lines.append(f"- {section.heading} ({priority})")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get statistics about loaded guidelines.

        Returns:
            Dictionary with statistics
        """
        if not self._sections:
            return {
                "total_sections": 0,
                "total_code_blocks": 0,
                "total_tokens_estimate": 0,
                "files_loaded": len(self._file_paths),
            }

        return {
            "total_sections": len(self._sections),
            "critical_sections": len([s for s in self._sections if s.priority_score <= 10]),
            "high_sections": len([s for s in self._sections if 10 < s.priority_score <= 20]),
            "medium_sections": len([s for s in self._sections if 20 < s.priority_score <= 30]),
            "total_code_blocks": sum(len(s.code_blocks) for s in self._sections),
            "unique_tables": len(set().union(*[s.affected_tables for s in self._sections])),
            "total_tokens_estimate": sum(s.token_estimate for s in self._sections),
            "files_loaded": len(self._file_paths),
        }

    def clear(self) -> None:
        """Clear all loaded content."""
        self._sections.clear()
        self._raw_content = ""
        self._file_paths.clear()
