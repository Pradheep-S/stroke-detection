"""
STROKE PREDICTION: DEEP LEARNING APPROACH

PROBLEM STATEMENT:
Stroke is a leading cause of disability globally. Early risk prediction enables
preventive intervention. This project develops a neural network model for
binary classification using the Kaggle Stroke Dataset (highly imbalanced: ~5% positive).

PROJECT STRUCTURE:
1. BASELINE MODELS: Classical ML (Logistic Regression, Random Forest)
   - Purpose: Benchmark comparison only
   - Note: Not optimized for recall; included for completeness
   
2. PROPOSED MODEL: Deep Learning (Neural Network)
   - Purpose: Primary model for deployment
   - Advantages:
     * Handles non-linear relationships in medical data
     * Flexible architecture tuning
     * Better recall with proper imbalance handling and threshold tuning
     * Feature interaction learning

3. ABLATION STUDY: Systematic comparison of design choices
   - Focal Loss vs Binary Cross-Entropy
   - SMOTE vs Class Weighting vs Combined
   - Dropout regularization impact

4. INTERPRETABILITY: Feature importance and sensitivity analysis
   - Clinical transparency and trust
   - Regulatory compliance (explain predictions)

EVALUATION FOCUS:
- Recall (primary): How many stroke cases do we catch?
- Precision: How many of our predictions are correct?
- F1-Score: Balanced metric
- ROC-AUC: Threshold-independent performance
- PR-AUC: Preferred for imbalanced datasets
- Threshold tuning: Medical-optimal decision boundary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Data preprocessing
from preprocessing.data_cleaning import clean_data
from preprocessing.encoding import encode_and_scale
from imbalance.smote_handler import apply_smote

# Baseline models (for comparison only)
from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest

# Proposed deep learning model
from models.neural_network import train_neural_network

# Evaluation
from evaluation.metrics import evaluate_model, detailed_metrics_report, find_optimal_threshold
from evaluation.roc_curve import plot_roc, plot_precision_recall_curve, plot_threshold_performance
from evaluation.confusion_matrix import plot_confusion_matrix

# Advanced evaluation
from ablation_study import run_full_ablation_study
from interpretability import interpret_stroke_model

# ============================================================================
# PHASE 0: DATA LOADING & PREPARATION
# ============================================================================

print("\n" + "="*80)

df = pd.read_csv("data/stroke.csv")
print(f"\nDataset shape: {df.shape}")
print(f"Class distribution: {df['stroke'].value_counts().to_dict()}")
print(f"Imbalance ratio: {df['stroke'].value_counts()[0] / df['stroke'].value_counts()[1]:.1f}:1")

# Clean data
df = clean_data(df)

# Encode and scale
X, y = encode_and_scale(df)

# Get feature names for interpretability
feature_names = df.drop('stroke', axis=1).columns.tolist()
print(f"\nFeatures: {feature_names}")
print(f"Feature count: {len(feature_names)}")

# Train-test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain set: {X_train.shape[0]} samples ({y_train.sum()} strokes)")
print(f"Test set: {X_test.shape[0]} samples ({y_test.sum()} strokes)")

# ============================================================================
# PHASE 1: BASELINE MODELS (Classical ML - For Reference Only)
# ============================================================================

print("\n" + "="*80)
print("BASELINE MODELS")
print("="*80)

# Apply SMOTE for baseline models
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

print(f"\nAfter SMOTE: {X_train_smote.shape[0]} samples ({y_train_smote.sum()} strokes)")

# Train baselines
lr_model = train_logistic_regression(X_train_smote, y_train_smote)
rf_model = train_random_forest(X_train_smote, y_train_smote)

lr_metrics = evaluate_model(lr_model, X_test, y_test)
print(f"\nLogistic Regression (on test set):")
for metric, value in lr_metrics.items():
    print(f"  {metric:15s}: {value:.4f}")

rf_metrics = evaluate_model(rf_model, X_test, y_test)
print(f"\nRandom Forest (on test set):")
for metric, value in rf_metrics.items():
    print(f"  {metric:15s}: {value:.4f}")

# ============================================================================
# PHASE 2: PROPOSED DEEP LEARNING MODEL
# ============================================================================

print("\n" + "="*80)
print("NEURAL NETWORK MODEL")
print("="*80)
nn_model, history = train_neural_network(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    dropout_rate=0.3,
    class_weight={0: 1.0, 1: 3.0},
    loss_function='binary_crossentropy',
    verbose=0
)



# ============================================================================
# PHASE 3: NEURAL NETWORK EVALUATION
# ============================================================================

print("\n" + "="*80)
print("NEURAL NETWORK EVALUATION")
print("="*80)

# Standard evaluation at default threshold (0.5)
nn_metrics_default = evaluate_model(nn_model, X_test, y_test, is_nn=True, threshold=0.5)
print(f"\nNeural Network Performance (threshold=0.5):")
for metric, value in nn_metrics_default.items():
    print(f"  {metric:15s}: {value:.4f}")

y_nn_probs = nn_model.predict(X_test, verbose=0).ravel()

optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_nn_probs, metric='f1')
print(f"Optimal threshold (F1): {optimal_threshold:.2f} (F1={optimal_f1:.4f})")

# Alternative: Maximize recall for medical safety
from evaluation.metrics import find_optimal_threshold
recall_threshold = 0.3  # Medical decision: accept more false alarms
nn_metrics_medical = detailed_metrics_report(
    nn_model, X_test, y_test, is_nn=True, threshold=recall_threshold
)

print(f"\nNeural Network Performance (threshold={recall_threshold} - Medical Tuning):")
for metric, value in nn_metrics_medical.items():
    print(f"  {metric:15s}: {value:.4f}")



# ============================================================================
# PHASE 4: COMPARATIVE VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)
print("\nRecall Comparison:")
print(f"  Logistic Regression:     {lr_metrics['Recall']:.4f}")
print(f"  Random Forest:           {rf_metrics['Recall']:.4f}")
print(f"  Neural Network (0.5):    {nn_metrics_default['Recall']:.4f}")
print(f"  Neural Network (0.3):    {nn_metrics_medical['Sensitivity (Recall)']:.4f}")

# Create comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curves
ax1 = axes[0]
plt.sca(ax1)
plot_roc(lr_model, X_test, y_test, label="Logistic Regression")
plot_roc(rf_model, X_test, y_test, label="Random Forest")
plot_roc(nn_model, X_test, y_test, is_nn=True, label="Neural Network")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve Comparison")
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Precision-Recall Curves
ax2 = axes[1]
plt.sca(ax2)
plot_precision_recall_curve(lr_model, X_test, y_test, label="Logistic Regression")
plot_precision_recall_curve(rf_model, X_test, y_test, label="Random Forest")
plot_precision_recall_curve(nn_model, X_test, y_test, is_nn=True, label="Neural Network")
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])
ax2.set_xlabel("Recall (Sensitivity)")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve (PReferredfor Imbalanced Data)")
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PHASE 5: THRESHOLD ANALYSIS FOR MEDICAL DEPLOYMENT
# ============================================================================

print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

fig = plt.figure(figsize=(12, 6))
plot_threshold_performance(nn_model, X_test, y_test, is_nn=True, label="Neural Network")
plt.savefig('results/threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PHASE 6: ABLATION STUDY (Optional - Demonstrates Engineering Depth)
# ============================================================================



# ============================================================================
# PHASE 7: INTERPRETABILITY & CLINICAL RELEVANCE
# ============================================================================

print("\n" + "="*80)
print("MODEL INTERPRETABILITY")
print("="*80)

interpretability_results = interpret_stroke_model(
    nn_model, X_test, y_test,
    feature_names=feature_names,
    is_nn=True
)

# Visualize feature importance
fig = interpretability_results['importance']
from interpretability import plot_feature_importance
fig = plot_feature_importance(interpretability_results['importance'], top_k=10)
plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# FINAL CONCLUSIONS
# ============================================================================


