"""SQL utilities for safe identifier escaping and DDL generation.

This module provides secure SQL identifier escaping to prevent SQL injection
in dynamically generated DDL statements.
"""

import re
from typing import Optional


class SQLInjectionError(ValueError):
    """Raised when SQL injection attempt is detected in identifiers."""
    pass


def escape_identifier(identifier: str) -> str:
    """Escape SQL identifier for use in DDL statements.

    This function provides protection against SQL injection by:
    1. Validating identifier contains only safe characters
    2. Removing any existing backticks (to prevent escape sequence attacks)
    3. Wrapping the identifier in backticks for Databricks SQL

    Args:
        identifier: SQL identifier (table name, column name, etc.)

    Returns:
        Backtick-quoted identifier safe for DDL statements

    Raises:
        SQLInjectionError: If identifier contains invalid characters

    Examples:
        >>> escape_identifier("my_table")
        '`my_table`'
        >>> escape_identifier("user-data")  # Hyphens not allowed
        Traceback (most recent call last):
            ...
        SQLInjectionError: Invalid SQL identifier 'user-data'
    """
    if not identifier:
        raise SQLInjectionError("SQL identifier cannot be empty")

    if not isinstance(identifier, str):
        raise SQLInjectionError(
            f"SQL identifier must be a string, got {type(identifier).__name__}"
        )

    clean = identifier.replace("`", "")

    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', clean):
        raise SQLInjectionError(
            f"Invalid SQL identifier '{identifier}'. "
            f"Identifiers must start with a letter or underscore and "
            f"contain only alphanumeric characters and underscores."
        )

    reserved_keywords = {
        'select', 'from', 'where', 'insert', 'update', 'delete',
        'drop', 'create', 'alter', 'table', 'database', 'schema',
        'union', 'join', 'and', 'or', 'not', 'null', 'true', 'false'
    }

    if clean.lower() in reserved_keywords:
        raise SQLInjectionError(
            f"SQL identifier '{identifier}' is a reserved keyword. "
            f"Use a different name or add a prefix/suffix."
        )

    if len(clean) > 255:
        raise SQLInjectionError(
            f"SQL identifier '{identifier[:50]}...' exceeds maximum length of 255 characters"
        )

    return f"`{clean}`"


def escape_table_name(
    catalog: str,
    schema: str,
    table: str,
    validate_parts: bool = True
) -> str:
    """Escape fully-qualified table name for safe DDL usage.

    Args:
        catalog: Catalog/database name
        schema: Schema name
        table: Table name
        validate_parts: If True, validate each part (default: True)

    Returns:
        Fully-qualified escaped table name: `catalog`.`schema`.`table`

    Raises:
        SQLInjectionError: If any part contains invalid characters

    Examples:
        >>> escape_table_name("main", "default", "users")
        '`main`.`default`.`users`'
    """
    if validate_parts:
        escaped_catalog = escape_identifier(catalog)
        escaped_schema = escape_identifier(schema)
        escaped_table = escape_identifier(table)
    else:
        escaped_catalog = f"`{catalog}`"
        escaped_schema = f"`{schema}`"
        escaped_table = f"`{table}`"

    return f"{escaped_catalog}.{escaped_schema}.{escaped_table}"


def escape_column_name(column: str) -> str:
    """Escape column name for safe DDL usage.

    This is an alias for escape_identifier for clarity when dealing
    specifically with column names.

    Args:
        column: Column name to escape

    Returns:
        Backtick-quoted column name

    Raises:
        SQLInjectionError: If column name contains invalid characters

    Examples:
        >>> escape_column_name("user_id")
        '`user_id`'
    """
    return escape_identifier(column)


def build_create_table_ddl(
    catalog: str,
    schema: str,
    table: str,
    columns: dict[str, str],
    comment: Optional[str] = None
) -> str:
    """Build CREATE TABLE DDL statement with proper escaping.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name
        columns: Dictionary mapping column names to their types
        comment: Optional table comment

    Returns:
        Safe CREATE TABLE DDL statement

    Raises:
        SQLInjectionError: If any identifier is invalid

    Examples:
        >>> ddl = build_create_table_ddl(
        ...     "main", "default", "users",
        ...     {"user_id": "BIGINT", "username": "STRING"}
        ... )
        >>> "CREATE TABLE" in ddl
        True
    """
    table_name = escape_table_name(catalog, schema, table)

    column_defs = []
    for col_name, col_type in columns.items():
        escaped_col = escape_column_name(col_name)

        if not re.match(r'^[A-Z][A-Z0-9_<>(),\s]*$', col_type.upper()):
            raise SQLInjectionError(
                f"Invalid column type '{col_type}' for column '{col_name}'"
            )

        column_defs.append(f"  {escaped_col} {col_type.upper()}")

    columns_str = ",\n".join(column_defs)

    ddl = f"CREATE TABLE {table_name} (\n{columns_str}\n)"

    if comment:
        safe_comment = comment.replace("'", "''")
        ddl += f"\nCOMMENT '{safe_comment}'"

    return ddl


def build_comment_on_table_ddl(
    catalog: str,
    schema: str,
    table: str,
    comment: str
) -> str:
    """Build COMMENT ON TABLE DDL statement with proper escaping.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name
        comment: Comment text

    Returns:
        Safe COMMENT ON TABLE DDL statement

    Raises:
        SQLInjectionError: If any identifier is invalid

    Examples:
        >>> ddl = build_comment_on_table_ddl(
        ...     "main", "default", "users", "User table"
        ... )
        >>> "COMMENT ON TABLE" in ddl
        True
    """
    table_name = escape_table_name(catalog, schema, table)
    safe_comment = comment.replace("'", "''")

    return f"COMMENT ON TABLE {table_name} IS '{safe_comment}'"


def build_alter_table_add_column_ddl(
    catalog: str,
    schema: str,
    table: str,
    column: str,
    column_type: str,
    comment: Optional[str] = None
) -> str:
    """Build ALTER TABLE ADD COLUMN DDL with proper escaping.

    Args:
        catalog: Catalog name
        schema: Schema name
        table: Table name
        column: New column name
        column_type: Column data type
        comment: Optional column comment

    Returns:
        Safe ALTER TABLE ADD COLUMN DDL statement

    Raises:
        SQLInjectionError: If any identifier is invalid

    Examples:
        >>> ddl = build_alter_table_add_column_ddl(
        ...     "main", "default", "users", "email", "STRING"
        ... )
        >>> "ALTER TABLE" in ddl
        True
    """
    table_name = escape_table_name(catalog, schema, table)
    column_name = escape_column_name(column)

    if not re.match(r'^[A-Z][A-Z0-9_<>(),\s]*$', column_type.upper()):
        raise SQLInjectionError(
            f"Invalid column type '{column_type}'"
        )

    ddl = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type.upper()}"

    if comment:
        safe_comment = comment.replace("'", "''")
        ddl += f" COMMENT '{safe_comment}'"

    return ddl


def validate_table_reference(table_ref: str) -> tuple[Optional[str], Optional[str], str]:
    """Parse and validate a table reference string.

    Accepts formats like:
    - "table"
    - "schema.table"
    - "catalog.schema.table"

    Args:
        table_ref: Table reference string

    Returns:
        Tuple of (catalog, schema, table) where catalog and schema may be None

    Raises:
        SQLInjectionError: If table reference is invalid

    Examples:
        >>> validate_table_reference("users")
        (None, None, 'users')
        >>> validate_table_reference("default.users")
        (None, 'default', 'users')
        >>> validate_table_reference("main.default.users")
        ('main', 'default', 'users')
    """
    if not table_ref:
        raise SQLInjectionError("Table reference cannot be empty")

    parts = table_ref.split(".")

    if len(parts) > 3:
        raise SQLInjectionError(
            f"Invalid table reference '{table_ref}'. "
            f"Maximum 3 parts allowed: catalog.schema.table"
        )

    for part in parts:
        escape_identifier(part)

    if len(parts) == 1:
        return (None, None, parts[0])
    elif len(parts) == 2:
        return (None, parts[0], parts[1])
    else:
        return (parts[0], parts[1], parts[2])
