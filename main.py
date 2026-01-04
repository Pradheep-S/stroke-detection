import pandas as pd
import matplotlib.pyplot as plt
from evaluation.confusion_matrix import plot_confusion_matrix

from sklearn.model_selection import train_test_split

from preprocessing.data_cleaning import clean_data
from preprocessing.encoding import encode_and_scale
from imbalance.smote_handler import apply_smote

from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.neural_network import train_neural_network

from evaluation.metrics import evaluate_model
from evaluation.roc_curve import plot_roc

# Load dataset
df = pd.read_csv("data/stroke.csv")

# Clean and preprocess
df = clean_data(df)
X, y = encode_and_scale(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# Train models
lr_model = train_logistic_regression(X_train_smote, y_train_smote)
rf_model = train_random_forest(X_train_smote, y_train_smote)
nn_model = train_neural_network(X_train, y_train)

# Evaluate
print("Logistic Regression:", evaluate_model(lr_model, X_test, y_test))
print("Random Forest:", evaluate_model(rf_model, X_test, y_test))
print("Neural Network (class-weighted):",
      evaluate_model(nn_model, X_test, y_test, is_nn=True))

# Plot ROC curves
plt.figure(figsize=(8, 6))
plot_roc(lr_model, X_test, y_test, label="Logistic Regression")
plot_roc(rf_model, X_test, y_test, label="Random Forest")
plot_roc(nn_model, X_test, y_test, is_nn=True, label="Neural Network")

# Plot confusion matrices
plot_confusion_matrix(lr_model, X_test, y_test, title="Logistic Regression")
plot_confusion_matrix(rf_model, X_test, y_test, title="Random Forest")
plot_confusion_matrix(nn_model, X_test, y_test, is_nn=True, title="Neural Network")
plt.show()
plt.title("ROC Curve")
plt.grid()
