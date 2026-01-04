from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

def train_neural_network(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    class_weight = {
        0: 1.0,
        1: 3.0
    }

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=0
    )

    return model
