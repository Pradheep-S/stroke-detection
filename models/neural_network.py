"""
Neural Network Model for Stroke Prediction

This module implements a deep learning (MLP) model for binary classification
of stroke risk using tabular medical data.

ARCHITECTURE:
- Input layer: Matches number of features (11 in stroke dataset)
- Hidden layers: 2 fully connected layers (128, 64 units)
- Regularization: Dropout layers (rate=0.3) to prevent overfitting
- Output layer: Sigmoid activation for binary classification
- Optimizer: Adam with default learning rate (0.001)
- Loss function: Binary Cross-Entropy or Focal Loss (configurable)

IMBALANCE HANDLING:
- Class weighting: Weights minority class (stroke) higher during training
- SMOTE: Optional resampling before training
- The model uses class_weight parameter to account for stroke class imbalance
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for handling class imbalance
    
    Focal loss applies a modulating term to the cross entropy loss to focus
    learning on hard negative examples and down-weight easy examples.
    
    Args:
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing parameter for class imbalance
    
    Returns:
        Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross entropy
        ce_loss = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal term
        focal_weight = tf.where(
            tf.equal(y_true, 1),
            alpha * tf.pow(1 - y_pred, gamma),
            (1 - alpha) * tf.pow(y_pred, gamma)
        )
        
        return focal_weight * ce_loss
    
    return focal_loss_fixed


def build_neural_network(
    input_dim,
    hidden_units=[128, 64],
    dropout_rate=0.3,
    loss_function='binary_crossentropy',
    learning_rate=0.001
):
    """
    Build a neural network for stroke prediction.
    
    Args:
        input_dim: Number of input features
        hidden_units: List of hidden layer unit counts
        dropout_rate: Dropout rate for regularization (0.0-0.5 typical)
        loss_function: 'binary_crossentropy' or 'focal_loss'
        learning_rate: Adam optimizer learning rate
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=(input_dim,)))
    
    # Hidden layers with ReLU activation and Dropout
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))  # Regularization: randomly drop neurons
    
    # Output layer: Sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Select loss function
    if loss_function == 'focal_loss':
        loss = focal_loss(gamma=2., alpha=0.25)
    else:
        loss = 'binary_crossentropy'
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def train_neural_network(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    epochs=60,
    batch_size=32,
    dropout_rate=0.3,
    class_weight=None,
    loss_function='binary_crossentropy',
    verbose=0
):
    """
    Train a neural network for stroke prediction.
    
    IMBALANCE HANDLING:
    - class_weight: Applies higher weights to minority class (stroke=1)
      This helps prevent the model from collapsing to majority class prediction
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        epochs: Number of training epochs
        batch_size: Batch size for gradient descent
        dropout_rate: Dropout regularization rate
        class_weight: Dictionary mapping class to weight (e.g., {0: 1.0, 1: 3.0})
        loss_function: 'binary_crossentropy' or 'focal_loss'
        verbose: Verbosity level (0=silent, 1=progress bar)
        
    Returns:
        Trained Keras model
    """
    
    # Build model
    model = build_neural_network(
        input_dim=X_train.shape[1],
        hidden_units=[128, 64],
        dropout_rate=dropout_rate,
        loss_function=loss_function
    )
    
    # Set default class weights if not provided
    if class_weight is None:
        # Weight minority class (stroke) 3x higher than majority class
        class_weight = {0: 1.0, 1: 3.0}
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    else:
        # Use 20% of training data for validation if not provided
        pass
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2 if validation_data is None else 0.0,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weight,  # CRITICAL: Handle class imbalance
        verbose=verbose
    )
    
    return model, history
