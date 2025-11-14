"""Utility functions for Cloud CEO."""

from cloud_ceo.utils.sql_utils import (
    escape_identifier,
    escape_table_name,
    escape_column_name,
    build_create_table_ddl,
    build_comment_on_table_ddl,
    build_alter_table_add_column_ddl,
    validate_table_reference,
    SQLInjectionError,
)

__all__ = [
    "escape_identifier",
    "escape_table_name",
    "escape_column_name",
    "build_create_table_ddl",
    "build_comment_on_table_ddl",
    "build_alter_table_add_column_ddl",
    "validate_table_reference",
    "SQLInjectionError",
]
