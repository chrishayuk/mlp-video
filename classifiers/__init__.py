"""
Classifier models for the video demo.

Contains manually implemented classifiers for educational purposes.
"""

from .linear import LinearClassifier
from .mlp import MLPClassifier

__all__ = ["LinearClassifier", "MLPClassifier"]
