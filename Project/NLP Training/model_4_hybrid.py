# Prerequisite: Please run all other models before running this script.
# This script combines the predictions from all other models to create a hybrid model.

import os
import shutil
import pandas as pd
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

TRAINING_PATH = os.path.dirname(os.path.realpath(__file__))

HYBRID_MODEL_OUTPUT_DIR = os.path.join(TRAINING_PATH, 'Hybrid')
if os.path.exists(HYBRID_MODEL_OUTPUT_DIR):
    shutil.rmtree(HYBRID_MODEL_OUTPUT_DIR)
os.makedirs(HYBRID_MODEL_OUTPUT_DIR)

DATA_PATH = os.path.realpath(os.path.join(TRAINING_PATH, "..","data"))


# =========================
# Load Prediction Array 
# from DEEPSEEK Model
# =========================
DEEPSEEK_OUTPUT_DIR = os.path.join('deepseek', 'deepseek_model')
deepseek_pred_table = pd.read_csv(os.path.join(TRAINING_PATH, 'deepseek', 'deepseek_model', 'test_predictions.csv'))
deepseek_true_labels = deepseek_pred_table['label']
deepseek_pred = deepseek_pred_table['predicted_label']
deepseek_pred_proba = deepseek_pred_table['predicted_probability']
assert len(deepseek_pred) == len(deepseek_true_labels) == len(deepseek_pred_proba)

# =========================
# Load Prediction Array
# from Baseline Model
# =========================
BASELINE_OUTPUT_DIR = os.path.join(TRAINING_PATH, 'Results')
baseline_pred = np.load(os.path.join(BASELINE_OUTPUT_DIR, 'baseline_y_pred.npy'))
baseline_true_labels = np.load(os.path.join(BASELINE_OUTPUT_DIR, 'baseline_y_true.npy'))
assert len(baseline_pred) == len(baseline_true_labels)