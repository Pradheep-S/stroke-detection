import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(model, X_test, y_test, is_nn=False, label="Model"):
    if is_nn:
        y_probs = model.predict(X_test)
    else:
        y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
