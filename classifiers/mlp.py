"""
MLP Classifier for non-linear classification.

A multi-layer perceptron that can learn non-linear decision boundaries.
"""

import mlx.core as mx
import mlx.nn as nn


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron Classifier.

    A simple single hidden layer network with ReLU activation:
    - Hidden: Linear(input_features, hidden_size) + ReLU
    - Output: Linear(hidden_size, num_classes)
    """

    def __init__(self, input_features: int, hidden_size: int, num_classes: int):
        super().__init__()
        scale1 = (2.0 / (input_features + hidden_size)) ** 0.5
        self.weight1 = mx.random.normal((input_features, hidden_size)) * scale1
        self.bias1 = mx.zeros((hidden_size,))

        scale2 = (2.0 / (hidden_size + num_classes)) ** 0.5
        self.weight2 = mx.random.normal((hidden_size, num_classes)) * scale2
        self.bias2 = mx.zeros((num_classes,))

    def __call__(self, x: mx.array) -> mx.array:
        # Hidden layer with ReLU
        h = x @ self.weight1 + self.bias1
        h = mx.maximum(h, 0)

        # Output layer
        return h @ self.weight2 + self.bias2
