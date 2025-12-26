"""
Generate exam score datasets for classification demos.

Outputs are saved to the output/ directory:
- output/exam_linear.csv + output/exam_linear.png
- output/exam_moons.csv + output/exam_moons.png
"""

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

# Output directory (relative to script location)
OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------
# Linear dataset: exam scores (linearly separable)
# ---------------------------------------------------------
def make_exam_scores_linear(n=1500, seed=0):
    mx.random.seed(seed)
    n4 = n // 4

    # Generate clusters: two passing, two failing
    # Pass clusters (average > 55)
    c1 = mx.random.normal((n4, 2), loc=0.0, scale=8.0) + mx.array([80.0, 80.0])  # avg 80
    c2 = mx.random.normal((n4, 2), loc=0.0, scale=8.0) + mx.array([60.0, 70.0])  # avg 65
    # Fail clusters (average < 55)
    c3 = mx.random.normal((n4, 2), loc=0.0, scale=8.0) + mx.array([30.0, 30.0])  # avg 30
    c4 = mx.random.normal((n4, 2), loc=0.0, scale=8.0) + mx.array([40.0, 50.0])  # avg 45

    X = mx.concatenate([c1, c2, c3, c4], axis=0)

    # Linear grading rule: pass if average score > 55
    score = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    y = (score > 55).astype(mx.int32)

    idx = mx.random.permutation(X.shape[0])
    return X[idx], y[idx]


# ---------------------------------------------------------
# Nonlinear dataset: exam moons
# ---------------------------------------------------------
def make_exam_scores_moons(n=1500, seed=0):
    mx.random.seed(seed)
    n2 = n // 2

    t1 = mx.random.uniform(0, mx.pi, (n2,))
    t2 = mx.random.uniform(0, mx.pi, (n2,))

    # Upper moon (pass): arcs from left to right, curving up
    coursework_1 = 35 + 30 * mx.cos(t1)
    exam_1 = 25 + 40 * mx.sin(t1)

    # Lower moon (fail): shifted right and flipped, interlocking
    coursework_2 = 65 - 30 * mx.cos(t2)
    exam_2 = 75 - 40 * mx.sin(t2)

    X = mx.stack(
        [
            mx.concatenate([coursework_1, coursework_2]),
            mx.concatenate([exam_1, exam_2]),
        ],
        axis=1,
    )

    y = mx.concatenate(
        [mx.ones((n2,), dtype=mx.int32), mx.zeros((n2,), dtype=mx.int32)]
    )

    X = X + mx.random.normal(X.shape, scale=1.5)

    idx = mx.random.permutation(n)
    return X[idx], y[idx]


# ---------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------
def save_dataset(X, y, path):
    X_np = np.array(X).round().astype(int)
    y_np = np.array(y)
    with open(path, "w") as f:
        f.write("coursework,exam,label\n")
        for i in range(len(y_np)):
            f.write(f"{X_np[i, 0]},{X_np[i, 1]},{y_np[i]}\n")
    print(f"Saved dataset to {path}")


def plot_dataset(X, y, output_path):
    X_np = np.array(X)
    y_np = np.array(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c='red', label='Fail', alpha=0.6)
    plt.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c='blue', label='Pass', alpha=0.6)
    plt.xlabel('Coursework')
    plt.ylabel('Exam')
    plt.title('Exam Scores Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate exam score datasets")
    parser.add_argument(
        "--mode",
        choices=["linear", "moons", "both"],
        default="both",
        help="Dataset type to generate (default: both)",
    )
    parser.add_argument("--n", type=int, default=1500, help="Number of samples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("linear", "both"):
        print("Generating linear dataset...")
        X, y = make_exam_scores_linear(args.n, args.seed)
        csv_path = OUTPUT_DIR / "exam_linear.csv"
        png_path = OUTPUT_DIR / "exam_linear.png"
        save_dataset(X, y, csv_path)
        plot_dataset(X, y, png_path)

    if args.mode in ("moons", "both"):
        print("Generating moons dataset...")
        X, y = make_exam_scores_moons(args.n, args.seed)
        csv_path = OUTPUT_DIR / "exam_moons.csv"
        png_path = OUTPUT_DIR / "exam_moons.png"
        save_dataset(X, y, csv_path)
        plot_dataset(X, y, png_path)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
