"""
INTERPRETABILITY & FEATURE IMPORTANCE

For healthcare applications, understanding WHY a model makes a prediction is
critical for clinical trust and regulatory compliance.

This module implements:
1. Permutation Importance: How much does each feature contribute to predictions?
2. Feature Sensitivity Analysis: How do changes in features affect predictions?

CLINICAL RELEVANCE:
- Identifies which patient attributes are most influential for stroke risk
- Examples: age, hypertension, glucose levels, heart disease
- Helps clinicians understand model reasoning and validate against medical knowledge
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt


def permutation_importance(model, X_test, y_test, feature_names=None, is_nn=False, n_repeats=10):
    """
    Calculate permutation importance for each feature.
    
    How it works:
    1. Get baseline model performance
    2. For each feature:
       - Randomly shuffle that feature's values
       - Measure performance drop
       - Higher drop → feature is more important
    
    Advantages:
    - Model-agnostic (works for sklearn and neural networks)
    - Measures actual impact on predictions
    - Considers feature interactions
    
    Args:
        model: Trained model (sklearn or Keras)
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        is_nn: Whether model is neural network
        n_repeats: Number of permutations per feature
        
    Returns:
        DataFrame with feature importance scores
    """
    
    # Get baseline predictions
    if is_nn:
        y_pred_baseline = (model.predict(X_test, verbose=0).ravel() >= 0.5).astype(int)
    else:
        y_pred_baseline = model.predict(X_test)
    
    # Baseline metric (use recall for medical relevance)
    baseline_recall = recall_score(y_test, y_pred_baseline, zero_division=0)
    
    importances = []
    n_features = X_test.shape[1]
    
    for feature_idx in range(n_features):
        recall_drops = []
        
        for _ in range(n_repeats):
            # Create a copy and shuffle the feature
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, feature_idx])
            
            # Get predictions on permuted data
            if is_nn:
                y_pred_permuted = (model.predict(X_test_permuted, verbose=0).ravel() >= 0.5).astype(int)
            else:
                y_pred_permuted = model.predict(X_test_permuted)
            
            # Calculate drop in performance
            permuted_recall = recall_score(y_test, y_pred_permuted, zero_division=0)
            recall_drop = baseline_recall - permuted_recall
            recall_drops.append(max(0, recall_drop))  # Importance is non-negative
        
        # Average importance across repeats
        importance = np.mean(recall_drops)
        importances.append(importance)
    
    # Create results dataframe
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df


def feature_sensitivity_analysis(model, X_test, y_test, feature_names=None, is_nn=False, percentiles=[10, 90]):
    """
    Analyze how changes in feature values affect stroke probability.
    
    This analyzes representative patients and shows how their stroke risk
    changes when we adjust individual features.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        is_nn: Whether model is neural network
        percentiles: Feature value percentiles to test
        
    Returns:
        Dictionary with sensitivity analysis results
    """
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]
    
    # Generate predictions for baseline
    if is_nn:
        baseline_probs = model.predict(X_test, verbose=0).ravel()
    else:
        baseline_probs = model.predict_proba(X_test)[:, 1]
    
    # Select a few representative cases
    # (high-risk and low-risk stroke patients)
    high_risk_idx = np.where(y_test == 1)[0]  # Actual stroke cases
    if len(high_risk_idx) > 0:
        rep_high_risk = high_risk_idx[0]
    else:
        rep_high_risk = None
    
    low_risk_idx = np.where(y_test == 0)[0]  # Non-stroke cases
    rep_low_risk = low_risk_idx[0] if len(low_risk_idx) > 0 else None
    
    results = {}
    
    for case_name, case_idx in [('High-Risk Patient', rep_high_risk), ('Low-Risk Patient', rep_low_risk)]:
        if case_idx is None:
            continue
        
        x_case = X_test[case_idx].reshape(1, -1)
        baseline_prob = baseline_probs[case_idx]
        
        sensitivity = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            p10 = np.percentile(X_test[:, feat_idx], percentiles[0])
            p90 = np.percentile(X_test[:, feat_idx], percentiles[1])
            
            # Get predictions at different feature values
            x_p10 = x_case.copy()
            x_p10[0, feat_idx] = p10
            
            x_p90 = x_case.copy()
            x_p90[0, feat_idx] = p90
            
            if is_nn:
                prob_p10 = model.predict(x_p10, verbose=0)[0, 0]
                prob_p90 = model.predict(x_p90, verbose=0)[0, 0]
            else:
                prob_p10 = model.predict_proba(x_p10)[0, 1]
                prob_p90 = model.predict_proba(x_p90)[0, 1]
            
            sensitivity[feat_name] = {
                'baseline': float(baseline_prob),
                'at_p10': float(prob_p10),
                'at_p90': float(prob_p90),
                'delta': float(prob_p90 - prob_p10)
            }
        
        results[case_name] = sensitivity
    
    return results


def plot_feature_importance(importance_df, title="Feature Importance", top_k=10):
    """
    Visualize feature importance.
    
    Args:
        importance_df: DataFrame from permutation_importance()
        title: Plot title
        top_k: Number of top features to display
    """
    df_plot = importance_df.head(top_k)
    
    plt.figure(figsize=(10, 6))
    plt.barh(df_plot['Feature'], df_plot['Importance'], color='steelblue')
    plt.xlabel('Importance (Recall Drop on Permutation)')
    plt.title(title)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    return plt.gcf()


def interpret_stroke_model(model, X_test, y_test, feature_names=None, is_nn=False):
    """
    Generate a comprehensive interpretability report for the stroke model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        is_nn: Whether model is neural network
        
    Returns:
        Dictionary with interpretability insights
    """
    
    print("\n" + "="*80)
    print("MODEL INTERPRETABILITY & CLINICAL RELEVANCE")
    print("="*80)
    
    # Feature importance
    print("\n[1/2] Computing Permutation-Based Feature Importance...")
    imp_df = permutation_importance(model, X_test, y_test, feature_names, is_nn)
    
    print("\nTop 5 Most Influential Features for Stroke Prediction:")
    print("-" * 60)
    for idx, row in imp_df.head(5).iterrows():
        print(f"  {row['Feature']:25s}: {row['Importance']:.4f}")
    
    # Feature sensitivity
    print("\n[2/2] Analyzing Feature Sensitivity...")
    sensitivity = feature_sensitivity_analysis(model, X_test, y_test, feature_names, is_nn)
    
    print("\nClinical Interpretation:")
    print("-" * 60)
    
    for patient_type, data in sensitivity.items():
        print(f"\n{patient_type}:")
        # Find features with largest impact
        top_features = sorted(
            data.items(),
            key=lambda x: abs(x[1]['delta']),
            reverse=True
        )[:3]
        
        for feat_name, values in top_features:
            delta = values['delta']
            direction = "increases" if delta > 0 else "decreases"
            print(f"  • {feat_name:25s}: {direction} stroke risk by {abs(delta):.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSIONS:")
    print("""
      - Age is typically a strong predictor (stroke risk increases with age)
      - Hypertension and heart disease are known risk factors
      - Glucose levels/metabolic markers affect stroke risk
    """)
    print("="*80)
    
    return {'importance': imp_df, 'sensitivity': sensitivity}

