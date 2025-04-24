# ##################
# This script trains additional layers from a fine-tuned DistilBERT model.
# Codes in this script are executed on a Linux-based virtual machine with the following computational requirements:
# GPU:  RTX2080 Super
# vCPU:  8 
# CPU Memory: 48GB 
# GPU Memory: 8GB
# Author: Kelvin Mock
# ##################

import os
TRAINING_PATH = os.path.dirname(os.path.realpath(__file__))
import numpy as np
from sklearn.model_selection import KFold # cross validation
import tensorflow as tf
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Logging
import time

def plot(history, fold: int, modelType: str):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} - {modelType} Model Accuracy')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_PATH, "models", f"{modelType}_accuracy_fold_{fold + 1}.png"))

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - {modelType} Model Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_PATH, "models", f"{modelType}_loss_fold_{fold + 1}.png"))

startTime = time.time()
# ============================
# STEP 7: Define custom classifier model
# ============================
def build_model(hp: dict):
    step7StartTime = time.time()
    print(f"🔧 Building model for {hp}...")
    inputs = tf.keras.layers.Input(shape=(X_train_resampled.shape[1],))  # (None, embedding_size)
    
    x = tf.keras.layers.Reshape((3, -1))(inputs)

    # Bi-directional LSTM Layer
    lstm_units = hp.get("lstm_units", [64, 128, 256])
    if isinstance(lstm_units, int):
        lstm_units = [lstm_units]
    for units in lstm_units:
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=units,
                return_sequences=False
            )
        )(x)

    # Dropout Layer
    dropout_rates = hp.get("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5])
    if isinstance(dropout_rates, float):
        dropout_rates = [dropout_rates]
    for dropout_rate in dropout_rates:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    # Dense Layer
    dense_units = hp.get("dense_units", [32, 64, 128, 256])
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    for units in dense_units:
        x = tf.keras.layers.Dense(
            units=units,
            activation="relu"
        )(x)

    # Output
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Build the Model with dummy data
    model(tf.zeros((1, X_train_resampled.shape[1])))

    # Compile the Model
    learning_rate = hp.get("learning_rate", [1e-5, 1e-4, 1e-3])
    if isinstance(learning_rate, float):
        learning_rate = [learning_rate]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp["learning_rate"]
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")] # for sigmoid outputs with non-integer predictions
    )
    step7EndTime = time.time()
    return model, step7EndTime - step7StartTime

# ============================
# STEP 8: Cross-validation and manual hyperparameter tuning for custom classifier
# ============================
step8StartTime = time.time()

FOLDS = 5
CUSTOM_EPOCHS = 20
TESTING_EPOCHS = 5

# Obtain Resampled Data
X_train_resampled = np.load(os.path.join(TRAINING_PATH, "resampled data", "X_train_resampled.npy"))
y_train_resampled = np.load(os.path.join(TRAINING_PATH, "resampled data", "y_train_resampled.npy"))

print(f"✅ Finished loading resampled data for custom classifier. Shape of X_train_resampled: {X_train_resampled.shape}, y_train_resampled: {y_train_resampled.shape}")

# Define candidate hyperparameter values
print("Start creating a grid for manual hyperparameter tuning for custom classifier...")
lstm_units_candidates = [64, 128, 256]
dropout_rate_candidates = [0.1, 0.2, 0.3, 0.4, 0.5]
dense_units_candidates = [32, 64, 128, 256]
learning_rate_candidates = [1e-5, 1e-4, 1e-3]

# cross validation folds
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

global_best_val_acc = 0.0
global_best_hp = None
global_best_history = None
global_best_fold = None

print("Starting cross-validation and manual hyperparameter tuning for custom classifier...")

for fold, (train_index, val_index) in enumerate(kf.split(X_train_resampled)):
    print(f"\nCustom Classifier Fold {fold + 1}")

    X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[val_index]
    y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[val_index]

    train_fold_dataset = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(4)
    val_fold_dataset = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(4)

    best_val_acc_fold = 0.0
    best_hp_fold = None
    best_history_fold = None
    best_elapsed_time = float('inf')

    # Manual grid search over hyperparameters
    for lstm_units in lstm_units_candidates:
        print(f"Evaluating LSTM units: {lstm_units}")
        for dropout_rate in dropout_rate_candidates:
            print(f"Evaluating dropout rate: {dropout_rate}")
            for dense_units in dense_units_candidates:
                print(f"Evaluating dense units: {dense_units}")
                for learning_rate in learning_rate_candidates:
                    print(f"Evaluating learning rate: {learning_rate}")
                    hp_candidate = {
                        "lstm_units": lstm_units,
                        "dropout_rate": dropout_rate,
                        "dense_units": dense_units,
                        "learning_rate": learning_rate
                    }
                    print(f"Evaluating candidate: {hp_candidate}")
                    model, step7ElapsedTime = build_model(hp_candidate)
                    print(f"🚀 Starting training for {hp_candidate}...")
                    history = model.fit(
                        train_fold_dataset,
                        validation_data=val_fold_dataset,
                        epochs=TESTING_EPOCHS,
                        verbose=0
                    )
                    print(f"✅ Training completed for {hp_candidate}. Elapsed time: {step7ElapsedTime:.2f} seconds")
                    current_val_acc = history.history['val_accuracy'][-1]
                    print(f"Candidate val_accuracy: {current_val_acc:.4f}")
                    if current_val_acc > best_val_acc_fold:
                        best_val_acc_fold = current_val_acc
                        best_hp_fold = hp_candidate
                        best_history_fold = history
                        best_elapsed_time = step7ElapsedTime

    if best_hp_fold is not None:
        print(f"✅ Best hyperparameters for fold {fold + 1}: {best_hp_fold} with val_accuracy: {best_val_acc_fold:.4f}")

        # Rebuild and train model using the best hyperparameters for this fold
        model, step7ElapsedTime = build_model(best_hp_fold)
        model.summary()
        history = model.fit(
            train_fold_dataset,
            validation_data=val_fold_dataset,
            epochs=CUSTOM_EPOCHS,
            verbose=1
        )
        val_acc = history.history['val_accuracy'][-1]
        model_save_path = os.path.join(TRAINING_PATH, "models", f"custom_classifier_fold_{fold + 1}.h5")
        model.save(model_save_path)
        print(f"✅ Model saved for fold {fold + 1} at: {model_save_path}")

        # Track overall best model across folds
        if val_acc > global_best_val_acc:
            global_best_val_acc = val_acc
            global_best_hp = best_hp_fold
            global_best_history = history
            global_best_fold = fold

if global_best_hp is not None:
    print(f"\nBest hyperparameters across all folds: {global_best_hp} with validation accuracy: {global_best_val_acc:.4f}")
    print(f"Saving the best model from fold {global_best_fold + 1}")
    best_model_path = os.path.join(TRAINING_PATH, "models", "BEST_custom_classifier.h5")
    model.save(best_model_path)
    print(f"✅ Best model saved at: {best_model_path}")

    modelType = "Custom Layers"
    plot(global_best_history, global_best_fold, modelType)
    print(f"✅ Convergence plots saved for {modelType} at fold {global_best_fold + 1}.")
else:
    print("⚠️ No best model found.")

step8EndTime = time.time()

# ============================
# Final Logging
# ============================
endTime = time.time()
print(f"Total Execution Time: {endTime - startTime} seconds")
print("✅ Pipeline for training custom layers (on top of DistilBERT) has been completed.")
# Correctly define Unbound variables
if best_elapsed_time is None:
    best_elapsed_time = 0
if step7ElapsedTime is None:
    step7ElapsedTime = 0
print(f"\nExecution Time for tuning the best custom classifier model in Step 7 within {TESTING_EPOCHS} epochs: {best_elapsed_time} seconds")
print(f"Execution Time for building the custom classifier model in Step 7 within {CUSTOM_EPOCHS} epochs: {step7ElapsedTime} seconds")
print(f"Execution Time for Step 8 (cross-validation and hyperparameter tuning): {step8EndTime - step8StartTime} seconds")