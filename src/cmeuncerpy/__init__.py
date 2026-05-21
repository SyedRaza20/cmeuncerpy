"""
Machine learning models with PyTorch and Rust backends.

Core Features:
- Linear Regression
- Neural Network Regression
"""

from cmeuncerpy._core import hello_from_bin
from cmeuncerpy.models import LinearRegression

__version__ = "0.1.0"

__all__ = [
    "hello_from_bin",
    "LinearRegression",
]


def hello() -> str:
    return hello_from_bin()

