"""Train a Linear Classifier on the exam dataset."""

from pathlib import Path

import mlx.core as mx

from classifiers import LinearClassifier
from chuk_lazarus.data import ClassificationDataset
from chuk_lazarus.training import evaluate, predict, train_classifier

DATA_PATH = Path(__file__).parent / "output" / "exam_linear.csv"


def main():
    # Load data
    print("Loading dataset...")
    dataset = ClassificationDataset.from_csv(
        DATA_PATH,
        feature_columns=["coursework", "exam"],
        label_column="label",
        normalize=100.0,
    )
    train, test = dataset.split(train_ratio=0.8, seed=42)
    print(f"  Train: {len(train)} samples, Test: {len(test)} samples")

    # Create model
    print("\nCreating LinearClassifier...")
    model = LinearClassifier(input_features=2, num_classes=2)

    # Train
    print("\nTraining...")
    train_classifier(model, train, epochs=20, lr=0.1, batch_size=32)

    # Evaluate
    result = evaluate(model, test)
    print(f"\nTest Accuracy: {result['accuracy']:.2%}")

    # Demo predictions
    print("\nSample predictions:")
    for cw, ex in [(85, 90), (30, 25), (55, 60), (50, 50)]:
        features = [cw / 100, ex / 100]
        logits = model(mx.array([features]))[0]
        label, conf = predict(model, features)
        print(f"  ({cw}, {ex}) -> logits: [{float(logits[0]):.2f}, {float(logits[1]):.2f}] -> {'Pass' if label else 'Fail'} ({conf:.1%})")


if __name__ == "__main__":
    main()
