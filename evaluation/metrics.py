from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, is_nn=False, threshold=0.5):
    if is_nn:
        y_probs = model.predict(X_test).ravel()
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    }
