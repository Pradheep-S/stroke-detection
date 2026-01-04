from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model
