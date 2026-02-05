"""
ROC Curve and Precision-Recall Curve Evaluation

ROC CURVE (Receiver Operating Characteristic):
- Shows trade-off between True Positive Rate (Recall) and False Positive Rate
- Threshold-independent: considers all decision thresholds
- ROC-AUC measures overall discriminative ability (0.5=random, 1.0=perfect)

PRECISION-RECALL CURVE:
- Shows trade-off between Precision and Recall
- PREFERRED for imbalanced datasets (more informative than ROC)
- PR-AUC is the area under the curve
- Ideal: high precision AND high recall

WHY PR CURVE FOR MEDICAL DATA:
In stroke prediction (5% positive cases), maximizing ROC-AUC might miss
high-recall importance. PR curve directly shows precision-recall trade-off,
making it more suitable for healthcare decisions.

THRESHOLD TUNING:
- Default: 0.5 (equal cost for false positives and negatives)
- Medical: 0.3-0.4 (accept more false alarms to catch stroke cases)
- Choose threshold based on clinical requirements and cost-benefit
"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import numpy as np


def plot_roc(model, X_test, y_test, is_nn=False, label="Model"):
    """
    Plot ROC curve.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        is_nn: Whether model is neural network
        label: Label for legend
    """
    if is_nn:
        y_probs = model.predict(X_test, verbose=0).ravel()
    else:
        y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.return_value = roc_auc


def plot_precision_recall_curve(model, X_test, y_test, is_nn=False, label="Model"):
    """
    Plot Precision-Recall curve.
    
    More informative than ROC for imbalanced classification.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        is_nn: Whether model is neural network
        label: Label for legend
        
    Returns:
        PR-AUC score
    """
    if is_nn:
        y_probs = model.predict(X_test, verbose=0).ravel()
    else:
        y_probs = model.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, label=f"{label} (PR-AUC = {pr_auc:.3f})")
    plt.xlabel("Recall (True Positive Rate)")
    plt.ylabel("Precision")
    plt.return_value = pr_auc
    
    return pr_auc


def plot_threshold_performance(model, X_test, y_test, is_nn=False, label="Model"):
    """
    Plot how metrics change with decision threshold.
    
    Helps select optimal threshold for clinical deployment.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        is_nn: Whether model is neural network
        label: Label for legend
    """
    if is_nn:
        y_probs = model.predict(X_test, verbose=0).ravel()
    else:
        y_probs = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.0, 1.01, 0.05)
    recalls = []
    precisions = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        from sklearn.metrics import recall_score, precision_score, f1_score
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, marker='o', label='Recall', linewidth=2)
    plt.plot(thresholds, precisions, marker='s', label='Precision', linewidth=2)
    plt.plot(thresholds, f1_scores, marker='^', label='F1-Score', linewidth=2)
    
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
    
    # Highlight medical-optimal threshold (high recall)
    optimal_recall_idx = np.argmax(recalls)
    optimal_recall_thresh = thresholds[optimal_recall_idx]
    plt.axvline(x=optimal_recall_thresh, color='red', linestyle='--', alpha=0.7, label=f'Optimal Recall ({optimal_recall_thresh:.2f})')
    
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(f"{label}: Metrics vs Decision Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return thresholds, recalls, precisions, f1_scores
