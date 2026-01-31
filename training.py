"""Training utilities for classification models."""

import csv
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class ClassificationDataset:
    """Dataset of numerical features and integer labels."""

    def __init__(self, features: mx.array, labels: mx.array):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        feature_columns: list[str],
        label_column: str,
        normalize: float = 1.0,
    ) -> "ClassificationDataset":
        """Load dataset from CSV file.

        Args:
            path: Path to CSV file.
            feature_columns: Column names to use as features.
            label_column: Column name for labels.
            normalize: Divide feature values by this number.
        """
        features, labels = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                features.append([float(row[c]) / normalize for c in feature_columns])
                labels.append(int(row[label_column]))
        return cls(mx.array(features), mx.array(labels))

    def split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> tuple["ClassificationDataset", "ClassificationDataset"]:
        """Split into train and test sets."""
        n = len(self)
        indices = list(range(n))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)
        split = int(n * train_ratio)
        train_idx = mx.array(indices[:split])
        test_idx = mx.array(indices[split:])
        return (
            ClassificationDataset(self.features[train_idx], self.labels[train_idx]),
            ClassificationDataset(self.features[test_idx], self.labels[test_idx]),
        )


def train_classifier(
    model: nn.Module,
    dataset: ClassificationDataset,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train a classification model with cross-entropy loss and Adam."""

    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(1, epochs + 1):
        # Shuffle each epoch
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        total_loss = 0.0
        steps = 0

        for start in range(0, len(dataset), batch_size):
            batch_idx = indices[start : start + batch_size]
            x = dataset.features[mx.array(batch_idx)]
            y = dataset.labels[mx.array(batch_idx)]

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            total_loss += loss.item()
            steps += 1

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{epochs}  loss={total_loss / steps:.4f}")


def evaluate(model: nn.Module, dataset: ClassificationDataset) -> dict:
    """Evaluate model accuracy on a dataset."""
    logits = model(dataset.features)
    preds = mx.argmax(logits, axis=1)
    accuracy = mx.mean(preds == dataset.labels).item()
    return {"accuracy": accuracy}


def predict(model: nn.Module, features: list[float]) -> tuple[int, float]:
    """Predict class and confidence for a single sample.

    Returns:
        (predicted_label, confidence) tuple.
    """
    logits = model(mx.array([features]))
    probs = mx.softmax(logits, axis=1)[0]
    label = mx.argmax(probs).item()
    confidence = probs[label].item()
    return int(label), confidence
