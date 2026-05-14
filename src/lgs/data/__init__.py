"""Data generation and curriculum helpers."""

from lgs.data.curriculum import FixedCurriculum
from lgs.data.target_generators import (
    make_common_factor_instance,
    make_product_of_sums_instance,
    make_square_instance,
    make_tiny_train_instances,
    make_tiny_validation_instances,
)

__all__ = [
    "FixedCurriculum",
    "make_common_factor_instance",
    "make_product_of_sums_instance",
    "make_square_instance",
    "make_tiny_train_instances",
    "make_tiny_validation_instances",
]
