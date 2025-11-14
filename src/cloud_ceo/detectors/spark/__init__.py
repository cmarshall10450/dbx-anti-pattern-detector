"""Spark SQL anti-pattern detectors.

This module contains detector implementations specific to Apache Spark SQL
and Databricks SQL.
"""

from cloud_ceo.detectors.spark.cartesian_join import CartesianJoinDetector
from cloud_ceo.detectors.spark.correlated_subquery import CorrelatedSubqueryDetector
from cloud_ceo.detectors.spark.distinct_on_unique import DistinctOnUniqueDetector
from cloud_ceo.detectors.spark.group_by_cardinality import GroupByCardinalityDetector
from cloud_ceo.detectors.spark.implicit_type_conversion import ImplicitTypeConversionDetector
from cloud_ceo.detectors.spark.inefficient_merge import InefficientMergeDetector
from cloud_ceo.detectors.spark.like_without_wildcards import LikeWithoutWildcardsDetector
from cloud_ceo.detectors.spark.missing_partition_filter import MissingPartitionFilterDetector
from cloud_ceo.detectors.spark.multiple_count_distinct import MultipleCountDistinctDetector
from cloud_ceo.detectors.spark.non_sargable_predicate import NonSargablePredicateDetector
from cloud_ceo.detectors.spark.not_in_with_nulls import NotInWithNullsDetector
from cloud_ceo.detectors.spark.or_to_in import OrToInDetector
from cloud_ceo.detectors.spark.select_star import SelectStarDetector
from cloud_ceo.detectors.spark.union_without_all import UnionWithoutAllDetector
from cloud_ceo.detectors.spark.window_without_partition import WindowWithoutPartitionDetector

__all__ = [
    "CartesianJoinDetector",
    "CorrelatedSubqueryDetector",
    "DistinctOnUniqueDetector",
    "GroupByCardinalityDetector",
    "ImplicitTypeConversionDetector",
    "InefficientMergeDetector",
    "LikeWithoutWildcardsDetector",
    "MissingPartitionFilterDetector",
    "MultipleCountDistinctDetector",
    "NonSargablePredicateDetector",
    "NotInWithNullsDetector",
    "OrToInDetector",
    "SelectStarDetector",
    "UnionWithoutAllDetector",
    "WindowWithoutPartitionDetector",
]
