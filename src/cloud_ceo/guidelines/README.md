# Guidelines Package

This package provides functionality for loading and managing custom Markdown SQL guidelines in Cloud CEO.

## Features

- **Markdown Guidelines**: Load any existing Markdown documentation without conversion
- **Smart Filtering**: Automatically filter guidelines by table relevance and priority
- **Token Management**: Respect LLM token budgets while maximizing guideline coverage
- **Priority Detection**: Automatic priority scoring based on keywords (CRITICAL, HIGH, etc.)
- **Table Awareness**: Extract and match table references from code blocks

## Quick Start

### Loading Markdown Guidelines

```python
from cloud_ceo.guidelines import MarkdownGuidelineLoader

loader = MarkdownGuidelineLoader()
loader.load_from_file(Path("guidelines/databricks_best_practices.md"))

# Get relevant sections for specific tables
content = loader.get_relevant_sections(
    query_tables=["customers", "orders"],
    max_tokens=2000
)
```

### Loading Multiple Files

```python
loader = MarkdownGuidelineLoader()
loader.load_from_directory(Path("guidelines"), recursive=False)

stats = loader.get_stats()
print(f"Loaded {stats['files_loaded']} files with {stats['total_sections']} sections")
```

### Using with ContextBuilder

```python
from cloud_ceo.guidelines import MarkdownGuidelineLoader
from cloud_ceo.llm.context import ContextBuilder, Violation

# Load guidelines
loader = MarkdownGuidelineLoader()
loader.load_from_file(Path("guidelines/databricks_best_practices.md"))

# Get filtered content
markdown_content = loader.get_relevant_sections(
    query_tables=["customers"],
    max_tokens=2000
)

# Build context with guidelines
builder = ContextBuilder()
context = builder.build_context(
    violation=violation,
    all_violations=all_violations,
    custom_guidelines_text=markdown_content
)
```

## Architecture

### Core Components

1. **MarkdownGuidelineLoader** (`markdown_loader.py`)
   - Loads unstructured Markdown documents
   - Extracts sections, code blocks, and table references
   - Implements smart filtering and token management
   - Detects priority keywords (CRITICAL, HIGH, MEDIUM)

2. **MarkdownSection** (dataclass)
   - Represents a parsed section with metadata
   - Tracks heading, content, code blocks, priority score
   - Maintains table references for filtering

## Markdown Guideline Format

### Structure

```markdown
# Document Title

## Rule Name (PRIORITY_KEYWORD)

Description of the rule and why it matters.

### Why This Matters

Business context and impact explanation.

### Examples

**Bad:**
```sql
-- Anti-pattern example
SELECT * FROM table
```

**Good:**
```sql
-- Best practice example
SELECT specific_columns FROM table WHERE filter
```

### Affected Tables
- table1
- table2
- pattern_*
```

### Priority Keywords

- **CRITICAL**, **MUST**, **REQUIRED**: Highest priority (score: 10)
- **HIGH PRIORITY**, **IMPORTANT**: High priority (score: 20)
- **MEDIUM**, **SHOULD**, **RECOMMENDED**: Medium priority (score: 30)
- No keyword: Low priority (score: 40)

### Token Management

The loader automatically:
1. Always includes CRITICAL sections (if budget allows)
2. Boosts sections matching query tables
3. Prefers sections with code examples
4. Sorts by priority and fills token budget

## Statistics and Monitoring

```python
loader = MarkdownGuidelineLoader()
loader.load_from_file(Path("guidelines.md"))

stats = loader.get_stats()

# Available metrics
stats['total_sections']       # Total sections
stats['critical_sections']    # CRITICAL priority count
stats['high_sections']        # HIGH priority count
stats['total_code_blocks']    # SQL code examples
stats['unique_tables']        # Referenced tables
stats['total_tokens_estimate']# Token estimate
stats['files_loaded']         # Files loaded
```

## Integration Points

### With ContextBuilder

`ContextBuilder.build_context()` accepts `custom_guidelines_text` parameter for Markdown guidelines.

### With LLM Prompts

Guidelines are injected directly into context as formatted Markdown text.

### Query-Specific Filtering

Use `get_relevant_sections()` to filter guidelines based on query tables and token budget.

## Testing

```bash
# Run all guideline tests
pytest tests/test_markdown_guidelines.py -v

# Run integration tests
pytest tests/integration/test_guideline_integration.py -v

# Run examples
python examples/custom_guidelines_demo.py
python examples/test_guideline_loading.py
```

## Best Practices

1. **Start with existing docs**: Drop in existing Markdown documentation without conversion
2. **Use priority keywords**: CRITICAL for must-follow rules, HIGH for important ones
3. **Include code examples**: Provides few-shot learning for better LLM understanding
4. **Specify affected tables**: Enables smart relevance filtering
5. **Set token budgets**: Balance context richness with prompt size
6. **Organize by topic**: Separate files for different areas (performance, security, etc.)

## File Organization

```
guidelines/
├── README.md
├── databricks_best_practices.md   # Real-world examples
├── sql_coding_standards.md        # Team standards
└── performance_guidelines.md      # Performance tips
```

## Example Guidelines

See the `guidelines/` directory for real-world examples:
- `databricks_best_practices.md`: Databricks-specific patterns
- `sql_coding_standards.md`: General SQL best practices
- `performance_guidelines.md`: Query optimization guidelines

## Documentation

- User guide: `/guidelines/README.md`
- Examples: `/examples/custom_guidelines_demo.py`
- Tests: `/tests/test_markdown_guidelines.py`

## API Reference

### MarkdownGuidelineLoader

#### Methods

- `load_from_file(file_path: Path)`: Load a single Markdown file
- `load_from_directory(directory: Path, recursive: bool)`: Load all .md files from directory
- `get_relevant_sections(query_tables: List[str], max_tokens: int)`: Get filtered content
- `get_all_content(max_tokens: int)`: Get all content with token limit
- `get_compact_summary(max_items: int)`: Get compact summary for display
- `get_stats()`: Get statistics about loaded guidelines
- `clear()`: Clear all loaded content

## Support

For questions or issues:
1. Review the examples in `/examples/`
2. Check the test files in `/tests/`
3. Consult the main documentation in `/guidelines/`

## Migration from YAML

If you were previously using YAML guidelines:
1. Convert your structured rules to Markdown format
2. Use priority keywords (CRITICAL, HIGH) to maintain severity levels
3. Include code examples in Markdown code blocks
4. Update imports to use `MarkdownGuidelineLoader` instead of `GuidelineLoader`
