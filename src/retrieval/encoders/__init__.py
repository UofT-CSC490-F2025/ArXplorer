"""Text encoders for dense and sparse representations."""

from .base import BaseEncoder
from .dense import DenseEncoder
from .sparse import SparseEncoder

__all__ = ["BaseEncoder", "DenseEncoder", "SparseEncoder"]
