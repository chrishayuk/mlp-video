"""
Linear Classifier for binary classification.

A simple linear model that learns a linear decision boundary.
"""

import mlx.core as mx
import mlx.nn as nn


class LinearClassifier(nn.Module):
    """
    Manual Linear Classifier for binary classification.

    This is a simple linear model: y = Wx + b
    where W is the weight matrix and b is the bias vector.

    For a 2-feature input (coursework, exam scores) and 2 classes (fail, pass):
    - Input: (batch_size, 2)
    - Output: (batch_size, 2) logits

    The model learns a linear decision boundary in the feature space.
    """

    def __init__(self, input_features: int, num_classes: int):
        super().__init__()
        # Xavier/Glorot initialization for better training
        scale = (2.0 / (input_features + num_classes)) ** 0.5
        self.weight = mx.random.normal((input_features, num_classes)) * scale
        self.bias = mx.zeros((num_classes,))

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass: compute logits.

        Args:
            x: Input features of shape (batch_size, input_features)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Linear transformation: y = x @ W + b
        return x @ self.weight + self.bias
