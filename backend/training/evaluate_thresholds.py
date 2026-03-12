"""
Evaluate and calibrate thresholds using Youden's J statistic.
Run after training to find optimal ALLOW/ALERT/BLOCK boundaries.
"""

import numpy as np
from sklearn.metrics import roc_curve
import json


def youden_j_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Find optimal threshold using Youden's J = Sensitivity + Specificity - 1
    Maximizes correct identification of both harmful and safe messages.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def evaluate_thresholds(predictions: list, labels: list):
    """
    Given model predictions and true labels, compute optimal thresholds.
    predictions: list of harm scores (0.0 to 1.0)
    labels: list of true labels (0=safe, 1=harmful)
    """
    y_scores = np.array(predictions)
    y_true = np.array(labels)

    # Binary: safe vs harmful
    threshold, j_score = youden_j_threshold(y_true, y_scores)
    print(f"Optimal Youden-J threshold: {threshold:.3f} (J={j_score:.3f})")

    # Compute at our current thresholds
    for t in [0.40, 0.65, 0.80]:
        preds_binary = (y_scores >= t).astype(int)
        tp = np.sum((preds_binary == 1) & (y_true == 1))
        fp = np.sum((preds_binary == 1) & (y_true == 0))
        tn = np.sum((preds_binary == 0) & (y_true == 0))
        fn = np.sum((preds_binary == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        fpr_val = fp / (fp + tn + 1e-8)

        print(f"Threshold {t:.2f}: Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f} FPR={fpr_val:.3f}")

    return {"optimal_threshold": threshold, "youden_j_score": j_score}


if __name__ == "__main__":
    # Example usage — replace with actual model outputs
    sample_scores = np.random.beta(0.5, 0.5, 1000)
    sample_labels = (sample_scores > 0.5).astype(int)
    results = evaluate_thresholds(sample_scores.tolist(), sample_labels.tolist())
    print(json.dumps(results, indent=2))
