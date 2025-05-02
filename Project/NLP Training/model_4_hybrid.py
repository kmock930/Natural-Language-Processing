# Prerequisite: Please run all other models before running this script.
# This script combines the predictions from all other models to create a hybrid model.

import os
import joblib
import numpy as np
import tensorflow as tf
import json
import pandas as pd
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

TRAINING_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.realpath(os.path.join(TRAINING_PATH, "..","data"))

TEST_DATA_PATH = os.path.join(DATA_PATH, 'Depression_Tweets')

# =========================
# Create An Ensemble Model
# An ensemble based on 
# majority voting rule
# =========================
class CustomVotingClassifier:
    def __init__(self, estimators, input_map, voting='soft'):
        """
        estimators: list of (name, model) tuples.
        input_map: dict mapping model names to input sets ('X1', 'X2', etc.)
        voting: 'soft' or 'hard'
        """
        self.estimators = dict(estimators)
        self.input_map = input_map
        self.voting = voting
        self.fitted_estimators = {}

    def fit(self, input_dict, y_dict):
        """
        input_dict: {'X1': np.ndarray, 'X2': np.ndarray, ...}
        y_dict: {'X1': labels, 'X2': labels, ...}
        """
        for name, model in self.estimators.items():
            X = input_dict[self.input_map[name]]
            y = y_dict[self.input_map[name]]
            if isinstance(model, tf.keras.Model):
                # Handle TensorFlow Keras models
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X, y)
            self.fitted_estimators[name] = model
        return self

    def _normalize_proba(self, p, name=""):
        """Ensure predictions are (n_samples, 2)"""
        print(f"[DEBUG] {name} raw shape before normalization: {p.shape}")

        if isinstance(p, tf.Tensor):
            p = p.numpy()
        if p.ndim == 1:
            # (n,) → (n,2)
            p = np.vstack([1 - p, p]).T
        elif p.ndim == 2:
            if p.shape[1] == 1:
                # Sigmoid → softmax-style
                p = np.hstack([1 - p, p])
            elif p.shape[1] == 4:
                if np.allclose(p[:, 0], p[:, 3], rtol=1e-3) and np.allclose(p[:, 1], p[:, 2], rtol=1e-3):
                    print(f"[{name}] 4-column mirrored output detected → truncated to [:, :2]")
                    p = p[:, :2]
                else:
                    raise ValueError(f"[{name}] Unexpected 4-column output structure")
            elif p.shape[1] != 2:
                raise ValueError(f"[{name}] Unsupported output shape: {p.shape}")
        else:
            raise ValueError(f"[{name}] Output must be 1D or 2D, got shape {p.shape}")
        
        print(f"[DEBUG] {name} normalized shape: {p.shape}")
        return p

    def predict_proba(self, input_dict):
        probs = []
        for name, model in self.fitted_estimators.items():
            X = input_dict[self.input_map[name]]
            if isinstance(model, tf.keras.Model):
                # Handle TensorFlow Keras models
                p = model.predict(X, verbose=0) if isinstance(model, tf.keras.Model) else model.predict_proba(X)
                p = self._normalize_proba(p, name)
            else:
                # Handle sklearn models
                p = model.predict_proba(X)
                p = self._normalize_proba(p, name)
            
            # Convert Tensor → NumPy
            if isinstance(p, tf.Tensor):
                p = p.numpy()

            print(f"Model {name} probabilities shape: {p.shape}")
            
            # Final sanity check
            assert p.shape[1] == 2, f"{name} returned probabilities of shape {p.shape}, expected ({p.shape[0]}, 2)"

            probs.append(p)
        
        probs = np.stack(probs)
        # Safety Check
        first_shape = probs[0].shape
        for i, p in enumerate(probs):
            if p.shape != first_shape:
                raise ValueError(f"Model {i} returned probabilities of shape {p.shape}, expected {first_shape}")
        return np.mean(probs, axis=0)

    def predict(self, input_dict):
        if self.voting == 'soft':
            # Based on probabilities
            return np.argmax(self.predict_proba(input_dict), axis=1)
        elif self.voting == 'hard':
            # Based on majority voting of labels
            preds = []
            for name, model in self.fitted_estimators.items():
                X = input_dict[self.input_map[name]]
                if isinstance(model, tf.keras.Model):
                    # Handle TensorFlow Keras models
                    p = model.predict(X, verbose=0)

                    # Convert Tensor to NumPy
                    if isinstance(p, tf.Tensor):
                        p = p.numpy()
                    
                    if p.ndim == 2 and p.shape[1] == 1:
                        p = p.flatten()
                    
                    p = (p > 0.5).astype(int)
                else:
                    # Handle sklearn models
                    p = model.predict(X)

                    # Ensure 1D shape
                    if p.ndim == 2 and p.shape[1] == 1:
                        p = p.flatten()
                preds.append(p)

            preds = np.array(preds)
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        else:
            raise ValueError("Voting must be 'soft' or 'hard'")

if __name__ == "__main__":
    # =========================
    # Load the Models
    # =========================
    baseline_model = joblib.load(os.path.join(TRAINING_PATH, 'models', 'best_baseline_model_LogisticRegression.h5'))
    try:
        distilBERT_model = tf.keras.models.load_model(os.path.join(TRAINING_PATH, 'models', 'BEST_custom_classifier.h5'))
    except Exception as e:
        # Set this to the shape of your input array
        INPUT_DIM = 2304  # 3 * 768 from your original Reshape

        # Define the previously trained model architecture again
        def build_custom_classifier():
            inputs = tf.keras.Input(shape=(INPUT_DIM,), name="input")
            x = tf.keras.layers.Reshape((3, -1))(inputs)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False))(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(32, activation="relu")(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            return model

        # Load Paths
        old_model_path = os.path.join(TRAINING_PATH, "models", "BEST_custom_classifier.h5")
        new_model_path = os.path.join(TRAINING_PATH, "models", "BEST_custom_classifier.keras")

        # Build and load weights
        model = build_custom_classifier()
        model.load_weights(old_model_path)

        # Save in .keras format
        model.save(new_model_path, save_format="keras")
        print(f"✅ Model converted and saved to {new_model_path}")

        distilBERT_model = tf.keras.models.load_model(new_model_path,compile=False)

    # =========================
    # Load the Input Data
    # =========================
    # Data tokenized by raw DistilBERT
    baseline_X_train = np.load(os.path.join(DATA_PATH, 'Numpy Data', 'Text', 'X_train_text.npy'))
    baseline_X_val = np.load(os.path.join(DATA_PATH, 'Numpy Data', 'Text', 'X_val_text.npy'))
    baseline_X_test = np.load(os.path.join(DATA_PATH, 'Numpy Data', 'Text', 'X_test_text.npy'))
    baseline_y_train = np.load(os.path.join(DATA_PATH, 'Numpy Data', 'y_train_text.npy'))
    baseline_y_val = np.load(os.path.join(DATA_PATH, 'Numpy Data', 'y_val_text.npy'))

    # Data tokenized by fine-tuned DistilBERT
    distilBERT_X_train = np.load(os.path.join(TRAINING_PATH, 'resampled data', 'X_train_resampled.npy'))
    distilBERT_X_val = np.load(os.path.join(TRAINING_PATH, 'resampled data', 'X_val_resampled.npy'))
    distilBERT_y_train = np.load(os.path.join(TRAINING_PATH, 'resampled data', 'y_train_resampled.npy'))
    distilBERT_y_val = np.load(os.path.join(TRAINING_PATH, 'resampled data', 'y_val_resampled.npy'))

    print("Baseline X_train shape: ", baseline_X_train.shape)
    print("DistilBERT X_train shape: ", distilBERT_X_train.shape)

    # =========================
    # Reshape the Data
    # =========================
    baseline_X_train = baseline_X_train.reshape(baseline_X_train.shape[0], -1)
    baseline_X_val = baseline_X_val.reshape(baseline_X_val.shape[0], -1)
    baseline_X_test = baseline_X_test.reshape(baseline_X_test.shape[0], -1)

    distilBERT_X_train = distilBERT_X_train.reshape(distilBERT_X_train.shape[0], -1)
    distilBERT_X_val = distilBERT_X_val.reshape(distilBERT_X_val.shape[0], -1)

    print("After Reshaping:")
    print("Baseline X_train shape: ", baseline_X_train.shape)
    print("DistilBERT X_train shape: ", distilBERT_X_train.shape)

    # =========================
    # Instantiate the Ensembles
    # with different votings
    # =========================
    ensemble_soft = CustomVotingClassifier(
        estimators=[
            ('baseline', baseline_model),
            ('distilbert', distilBERT_model),
        ],
        input_map={
            'baseline': 'X1',
            'distilbert': 'X2',
        },
        voting='soft' # probability-based voting
    )

    ensemble_hard = CustomVotingClassifier(
        estimators=[
            ('baseline', baseline_model),
            ('distilbert', distilBERT_model),
        ],
        input_map={
            'baseline': 'X1',
            'distilbert': 'X2',
        },
        voting='hard' # label-based majority voting
    )

    # =========================
    # Fit the Ensemble Models
    # =========================
    ensemble_soft.fit(
        input_dict={
            'X1': baseline_X_train,
            'X2': distilBERT_X_train,
        },
        y_dict={
            'X1': baseline_y_train,
            'X2': distilBERT_y_train,
        }
    )

    ensemble_hard.fit(
        input_dict={
            'X1': baseline_X_train,
            'X2': distilBERT_X_train,
        },
        y_dict={
            'X1': baseline_y_train,
            'X2': distilBERT_y_train,
        }
    )

    #############################################################

    # ==================================
    # Predict with RAW DistilBERT
    # ==================================
    proba_soft = ensemble_soft.predict_proba(
        {
            'X1': baseline_X_val,
            'X2': baseline_X_val,
        }
    )
    labels_soft = (proba_soft[:, 1] > 0.5).astype(int)

    labels_hard = ensemble_hard.predict({
        'X1': baseline_X_val,
        'X2': baseline_X_val,
    })

    # =========================
    # Combine Predictions
    # Average of class 1 ('Suicidal') probabilities
    # =========================
    final_preds = (labels_soft + labels_hard >= 1).astype(int) # Majority voting

    # =========================
    # Save the Predictions 
    # And True Labels
    # =========================
    np.save(os.path.join(TRAINING_PATH, 'Results', 'model_4_hybrid_baseline_predictions.npy'), final_preds)
    np.save(os.path.join(TRAINING_PATH, 'Results', 'model_4_hybrid_baseline_true_labels.npy'), baseline_y_val)

    #############################################################

    # ==================================
    # Predict with fine-tuned DistilBERT
    # ==================================
    proba_soft = ensemble_soft.predict_proba(
        {
            'X1': distilBERT_X_val,
            'X2': distilBERT_X_val,
        }
    )
    labels_soft = (proba_soft[:, 1] > 0.5).astype(int)

    labels_hard = ensemble_hard.predict({
        'X1': distilBERT_X_val,
        'X2': distilBERT_X_val,
    })

    # =========================
    # Combine Predictions
    # Average of class 1 ('Suicidal') probabilities
    # =========================
    final_preds = (labels_soft + labels_hard >= 1).astype(int) # Majority voting

    print(f"Shape of Final Predictions: {final_preds.shape}")

    # =========================
    # Save the Predictions 
    # And True Labels
    # =========================
    np.save(os.path.join(TRAINING_PATH, 'Results', 'model_4_hybrid_distilBERT_predictions.npy'), final_preds)
    np.save(os.path.join(TRAINING_PATH, 'Results', 'model_4_hybrid_distilBERT_true_labels.npy'), distilBERT_y_val)

    # =========================
    # Save the Model
    # =========================
    joblib.dump(ensemble_soft, os.path.join(TRAINING_PATH, 'models', 'ensemble_soft_model.pkl'))
    joblib.dump(ensemble_hard, os.path.join(TRAINING_PATH, 'models', 'ensemble_hard_model.pkl'))

    # ===========================
    # Export Test Set Predictions
    # ===========================
    print(f"Shape of Test Set: {baseline_X_test.shape}")
    proba_soft = ensemble_soft.predict_proba(
        {
            'X1': baseline_X_test,
            'X2': baseline_X_test,
        }
    )
    labels_soft = (proba_soft[:, 1] > 0.5).astype(int)

    labels_hard = ensemble_hard.predict({
        'X1': baseline_X_test,
        'X2': baseline_X_test,
    })

    final_preds = (labels_soft + labels_hard >= 1).astype(int) # Majority voting

    print(f"Shape of Final Predictions (on the Test Set): {final_preds.shape}")

    results = []
    test_data = pd.read_json(os.path.join(TEST_DATA_PATH, 'depression_json'))
    print(f"Length of Test Data: {len(test_data)}")
    for idx, (text, label) in enumerate(zip(test_data['content'], final_preds)):
        results.append({
            'id': idx,
            'predicted_label': int(label),
            'raw_text': text,
        })

    # Define the output file path
    output_file_path = os.path.join(TRAINING_PATH, "Results", "Result_hybrid_model.jsonl")

    # Save the results to a JSONL file
    with open(output_file_path, "w") as jsonl_file:
        for record in results:
            jsonl_file.write(json.dumps(record) + "\n")

    print(f"Results saved to {output_file_path}")