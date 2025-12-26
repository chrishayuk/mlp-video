"""
Generate a standard two-moons dataset for classification demos.

This creates a classic non-linearly separable dataset used to demonstrate
the limitations of linear classifiers and the power of neural networks.

Output saved to: output/two_moons.png
"""

import math
from pathlib import Path

import mlx.core as mx

# Output directory (relative to script location)
OUTPUT_DIR = Path(__file__).parent / "output"


def make_two_moons(n=1000, noise=0.1, seed=0):
    """
    Generate a 2D two-moons classification dataset.

    Args:
        n: total number of points
        noise: gaussian noise added to points
        seed: RNG seed

    Returns:
        X: (n, 2) float32 features
        y: (n,) int32 labels {0,1}
    """
    mx.random.seed(seed)
    n2 = n // 2

    # angles for each moon
    t1 = mx.random.uniform(0, math.pi, (n2,))
    t2 = mx.random.uniform(0, math.pi, (n2,))

    # first moon
    x1 = mx.stack([mx.cos(t1), mx.sin(t1)], axis=1)

    # second moon (shifted + flipped)
    x2 = mx.stack([1.0 - mx.cos(t2), 1.0 - mx.sin(t2) - 0.5], axis=1)

    # combine
    X = mx.concatenate([x1, x2], axis=0)
    X = X + noise * mx.random.normal(X.shape)

    y = mx.concatenate(
        [mx.zeros((n2,), dtype=mx.int32), mx.ones((n2,), dtype=mx.int32)], axis=0
    )

    # shuffle
    idx = mx.random.permutation(n)
    return X[idx].astype(mx.float32), y[idx]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_two_moons(n=1500, noise=0.12, seed=42)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap="coolwarm")
    plt.title("Two Moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)

    output_path = OUTPUT_DIR / "two_moons.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to: {output_path}")
