import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, X_test, y_test, is_nn=False, threshold=0.5, title="Confusion Matrix"):
    if is_nn:
        y_probs = model.predict(X_test).ravel()
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Stroke", "Stroke"],
                yticklabels=["No Stroke", "Stroke"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()
