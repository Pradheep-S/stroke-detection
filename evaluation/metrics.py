"""
Evaluation Metrics for Stroke Prediction Models

MEDICAL RELEVANCE EXPLANATION:
In stroke prediction, missing a positive case (false negative) is clinically
more dangerous than a false positive. Therefore:

- RECALL is the PRIMARY METRIC: "Of all actual stroke cases, how many did we catch?"
  A recall of 0.8 means we catch 80% of stroke patients (20% miss rate).
  
- PRECISION: "Of our positive predictions, how many are actually correct?"
  
- F1-SCORE: Harmonic mean of precision and recall (balanced view)

- ROC-AUC: Area under the ROC curve (threshold-independent performance)

WHY NOT ACCURACY:
In imbalanced datasets (5% stroke cases), predicting "no stroke" for everyone
gives 95% accuracy but catches ZERO stroke cases. This is clinically useless.

THRESHOLD TUNING:
Default threshold is 0.5, but for medical applications, we may lower it
to 0.3-0.4 to catch more cases (increase recall), accepting more false alarms.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import numpy as np


def evaluate_model(model, X_test, y_test, is_nn=False, threshold=0.5):
    """
    Comprehensive evaluation for binary classification.
    
    Args:
        model: Trained model (sklearn or Keras)
        X_test: Test features
        y_test: Test labels
        is_nn: Whether model is a neural network (Keras)
        threshold: Decision threshold (default 0.5)
        
    Returns:
        Dictionary with metrics
    """
    if is_nn:
        y_probs = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
    }
    
    # ROC-AUC (threshold-independent metric)
    if y_probs is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_test, y_probs)
        except:
            metrics["ROC-AUC"] = 0.0
    
    return metrics


def find_optimal_threshold(y_true, y_probs, metric='f1'):
    """
    Find the optimal decision threshold for a given metric.
    
    For medical applications, you might prioritize recall over precision.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        metric: 'f1', 'recall', or 'precision' to optimize
        
    Returns:
        Optimal threshold and corresponding metric value
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    return optimal_threshold, optimal_score


def detailed_metrics_report(model, X_test, y_test, is_nn=False, threshold=0.5):
    """
    Generate a detailed medical-relevant metrics report.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        is_nn: Whether model is neural network
        threshold: Decision threshold
        
    Returns:
        Dictionary with detailed metrics and explanations
    """
    if is_nn:
        y_probs = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    tn = sum((y_pred == 0) & (y_test == 0))
    fp = sum((y_pred == 1) & (y_test == 0))
    fn = sum((y_pred == 0) & (y_test == 1))
    tp = sum((y_pred == 1) & (y_test == 1))
    
    report = {
        "Threshold": threshold,
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Sensitivity (Recall)": float(recall_score(y_test, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    if y_probs is not None:
        try:
            report["ROC-AUC"] = float(roc_auc_score(y_test, y_probs))
        except:
            report["ROC-AUC"] = 0.0
    
    return report
