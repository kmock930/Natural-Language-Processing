# ##################
# This script fine-tunes the pre-trained DistilBERT model.
# Codes in this script are executed on a Linux-based virtual machine with the following computational requirements:
# GPU:  RTX2080 Super
# vCPU:  8 
# CPU Memory: 48GB 
# GPU Memory: 8GB
# Author: Kelvin Mock
# ##################
from transformers import DistilBertTokenizer, TFDistilBertModel
import os
import sys
TRAINING_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.realpath(os.path.join(TRAINING_PATH, "..","data"))
sys.path.append(DATA_PATH)
import tensorflow as tf
import numpy as np
# from keras_tuner import Hyperband, HyperParameters

import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold # cross validation
from data_processing import normalize
import pandas as pd
# import keras_tuner
import shutil

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

PRETRAINED_EPOCHS = 5
CUSTOM_EPOCHS = 20

# Logging
import time
startTime = time.time()

# Load a pretrained model
# https://huggingface.co/distilbert/distilbert-base-uncased
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# This directory is for saving models
if os.path.exists(os.path.join(TRAINING_PATH, "models")):
    shutil.rmtree(os.path.join(TRAINING_PATH, "models"))
os.makedirs(os.path.join(TRAINING_PATH, "models"))

# ============================
# STEP 1: Obtain and Preprocess the data
# ============================
step1StartTime = time.time()
# Defining Paths to Datasets - We need RAW texts for DistilBERT
PATHS = [
    # TRAINING DATA 
    os.path.join(DATA_PATH, 'Twitter_Suicidal_Data'),
    os.path.join(DATA_PATH, 'Social_Media_Sentiments_Analysis_Dataset'),
    # VALIDATION DATA
    os.path.join(DATA_PATH, 'Reddit_SuicideWatch'),
    # TEST DATA
    os.path.join(DATA_PATH, 'Depression_Tweets')
]

X_train_title = []
X_train_content = []
X_train_hashtags = []
X_val_title = []
X_val_content = []
X_val_hashtags = []
X_test_title = []
X_test_content = []
X_test_hashtags = []

y_train = []
y_val = []

# Identify groups - train, validation, test sets
# RAW texts - DON"T vectorize
for path in PATHS:
    if (path.endswith("Twitter_Suicidal_Data")):
        # Load the data into pandas
        df = pd.read_csv(os.path.join(path, "twitter-suicidal_data.csv"))
        # Preprocess Post content
        for index, row in df.iterrows():
            # Extract Features from Columns
            # Text Embeddings
            title = ''
            post_content = row['tweet']
            hashtags = ''
            # Identify Labels from Annotations
            label = int(row['intention'])
            # Normalize and vectorize the post content
            X_train_title.append(normalize(title))
            X_train_content.append(normalize(post_content))
            X_train_hashtags.append(normalize(hashtags))

            y_train.append(label)
    elif (path.endswith("Social_Media_Sentiments_Analysis_Dataset")):
        # Load the data into pandas
        df_full = pd.read_csv(os.path.join(path, "sentimentdataset_annotated_binary.csv"))
        df_sensitive = pd.read_csv(os.path.join(path, "sentimentdataset_binary_class_sensitive_attributes.csv"))
        df_nonsensitive = pd.read_csv(os.path.join(path, "sentimentdataset_binary_class_non_sensitive_attributes.csv"))
        # Preprocess Post content
        for index, row in df_full.iterrows():
            # Extract Features from Columns
            # Text Embeddings
            title = ''
            post_content = row['Text']
            hashtags = row['Hashtags']
            # Identify Labels from Annotations
            label = int(row['Annotation'])
            # Normalize and vectorize the post content
            X_train_title.append(normalize(title))
            X_train_content.append(normalize(post_content))
            X_train_hashtags.append(normalize(hashtags))

            y_train.append(label)
    elif (path.endswith("Reddit_SuicideWatch")):
        # Load the data into pandas
        df_full = pd.read_csv(os.path.join(path, "reddit_suicidewatch.csv"))
        df_sensitive = pd.read_csv(os.path.join(path, "reddit_suicidewatch_sensitive_attribute.csv"))
        df_nonsensitive = pd.read_csv(os.path.join(path, "reddit_suicidewatch_non_sensitive_attribute.csv"))
        # Preprocess Post content
        for index, row in df_full.iterrows():
            title = row['title']
            post_content = row['selftext']
            hashtags = ''
            label = 1 # everything here is suicidal
            # Normalize and vectorize the post content
            X_val_title.append(normalize(title))
            X_val_content.append(normalize(post_content))
            X_val_hashtags.append(normalize(hashtags))

            y_val.append(label)
    elif (path.endswith("Depression_Tweets")):
        # Load the data into pandas
        df_full = df_sensitive = df_nonsensitive = pd.read_json(os.path.join(path, "depression_json"))
        # Preprocess Post content
        for index, row in df_full.iterrows():
            title = ''
            post_content = row["content"]
            hashtags = ''
            # No ground truth label is provided in the test set
            # Normalize and vectorize the post content
            X_test_title.append(normalize(title))
            X_test_content.append(normalize(post_content))
            X_test_hashtags.append(normalize(hashtags))

            # No Ground Truth Labels

print("✅ Data has been loaded.")

# Inspect the data
print(f"X_train_title: {len(X_train_title)}")
print(f"X_train_content: {len(X_train_content)}")
print(f"X_train_hashtags: {len(X_train_hashtags)}")
print(f"X_val_title: {len(X_val_title)}")
print(f"X_val_content: {len(X_val_content)}")
print(f"X_val_hashtags: {len(X_val_hashtags)}")
print(f"X_test_title: {len(X_test_title)}")
print(f"X_test_content: {len(X_test_content)}")
print(f"X_test_hashtags: {len(X_test_hashtags)}")
print(f"y_train: {len(y_train)}")
print(f"y_val: {len(y_val)}")
assert len(X_train_title) == len(X_train_content) == len(X_train_hashtags) == len(y_train)
assert len(X_val_title) == len(X_val_content) == len(X_val_hashtags) == len(y_val)
assert len(X_test_title) == len(X_test_content) == len(X_test_hashtags)

print("✅ Data has been validated.")

# Observe class distributions in training and validation sets
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_val, counts_val = np.unique(y_val, return_counts=True)

print("Class distribution in y_train:")
for cls, count in zip(unique_train, counts_train):
    print(f"Class {cls}: {count} samples")

print("\nClass distribution in y_val:")
for cls, count in zip(unique_val, counts_val):
    print(f"Class {cls}: {count} samples")
step1EndTime = time.time()
# =============================
# Since the validation set only has 1 class, we perform a random train-test-split
# ============================
# STEP 2: Randomize data and reassign train/val split
# ============================
step2StartTime = time.time()

X_title = X_train_title + X_val_title
X_content = X_train_content + X_val_content
X_hashtags = X_train_hashtags + X_val_hashtags
y_labels = y_train + y_val

X_train_title, X_val_title, X_train_content, X_val_content, X_train_hashtags, X_val_hashtags, y_train, y_val = train_test_split(
    X_title,
    X_content,
    X_hashtags,
    y_labels,
    test_size=0.2,
    random_state=42,
    stratify=y_labels
)
print("✅ Data has been randomized and split into training and validation sets.")

def plot(history, fold: int, modelType: str):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold + 1} - {modelType} Model Accuracy')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_PATH, "models", f"{modelType} accuracy_fold_{fold + 1}.png"))

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold + 1} - {modelType} Model Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_PATH, "models", f"{modelType} loss_fold_{fold + 1}.png"))
step2EndTime = time.time()
# ============================
# STEP 3: Fine-tune DistilBERT
# ============================
step3StartTime = time.time()
class FixedHP:
    def Float(self, name, min_val, max_val, step=None, sampling=None):
        return 0.3 if name == "dropout_rate" else 3e-5

def build_finetune_bert_model(hp):
    # Load Hugging Face DistilBERT encoder
    bert_encoder = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    bert_encoder.trainable = True

    # Define input layers
    input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

    # Wrap the encoder using Lambda to prevent KerasTensor type mismatch
    def bert_layer(inputs):
        input_ids, attention_mask = inputs
        return bert_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    # Pass through BERT
    bert_outputs = tf.keras.layers.Lambda(
        bert_layer,
        output_shape=(128, 768) # (batch_size, sequence_length=128, hidden_size=768)
    )([input_ids, attention_mask])
    cls_output = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(bert_outputs)

    # Dropout and final Dense
    x = tf.keras.layers.Dropout(hp.Float("dropout_rate", 0.1, 0.5, step=0.1))(cls_output)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Assemble model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # Compile the Model with an optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Float("learning_rate", 1e-5, 1e-4, sampling="LOG")
    )
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, bert_encoder

# Prepare combined inputs - RAW texts
combined_train_texts = [f"{title} {content} {hashtags}".strip() for title, content, hashtags in zip(X_train_title, X_train_content, X_train_hashtags)]
combined_val_texts = [f"{title} {content} {hashtags}".strip() for title, content, hashtags in zip(X_val_title, X_val_content, X_val_hashtags)]

encoded_train = tokenizer(combined_train_texts, padding='max_length', truncation=True, max_length=128, return_tensors='tf')
encoded_val = tokenizer(combined_val_texts, padding='max_length', truncation=True, max_length=128, return_tensors='tf')

train_inputs = {
    'input_ids': encoded_train['input_ids'],
    'attention_mask': encoded_train['attention_mask']
}

val_inputs = {
    'input_ids': encoded_val['input_ids'],
    'attention_mask': encoded_val['attention_mask']
}

# cross validation on fine-tuning distilBERT
kf = KFold(n_splits=3, shuffle=True, random_state=42)
best_val_acc = 0.0
bestHistory = None
bestFold = None
best_encoder = None
for fold, (train_index, val_index) in enumerate(kf.split(train_inputs['input_ids'])): # 3 folds
    fold_train_inputs = {
        'input_ids': tf.gather(train_inputs['input_ids'], train_index),
        'attention_mask': tf.gather(train_inputs['attention_mask'], train_index)
    }
    fold_val_inputs = {
        'input_ids': tf.gather(train_inputs['input_ids'], val_index),
        'attention_mask': tf.gather(train_inputs['attention_mask'], val_index)
    }
    y_train_fold = np.array(y_train)[train_index]
    y_val_fold = np.array(y_train)[val_index]

    model, bert_encoder = build_finetune_bert_model(FixedHP())
    history = model.fit(
        fold_train_inputs,
        y_train_fold,
        validation_data=(fold_val_inputs, y_val_fold),
        epochs=PRETRAINED_EPOCHS
    )

    val_acc = history.history["val_accuracy"][-1]
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_encoder = bert_encoder
        best_model_path = os.path.join(TRAINING_PATH, "models", f"BEST_fine_tuned_distilBERT_fold_{fold+1}")
        bestHistory = history
        bestFold = fold
    
    # Save encoder separately for embedding extraction later
    # "model" is just a keras wrapper for training, no need to save
    encoder_save_path = os.path.join(TRAINING_PATH, "models", f"model_deep_learning_distilBERT_finetuned_encoder_fold_{fold + 1}")
    bert_encoder.save_pretrained(encoder_save_path)
    tokenizer.save_pretrained(encoder_save_path)
    print(f"✅ Fine-tuned Hugging Face encoder and tokenizer saved at: {encoder_save_path}")

if best_model_path:
    print(f"Saving the best model from fold {bestFold + 1} with validation accuracy: {best_val_acc}")
    best_encoder.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"✅ Best model saved at: {best_model_path}")
    
    modelType = "Fine-tuned DistilBERT"
    plot(bestHistory, bestFold, modelType)
    print(f"✅ Convergence plots saved for {modelType} at fold {bestFold + 1}.")
else: # Error Handling
    best_model_path = os.path.join(TRAINING_PATH, "models", "BEST_fine_tuned_distilBERT")
    # If no best model was found, we can still save the last trained model
    if best_encoder:
        best_encoder.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"✅ Last trained model saved at: {best_model_path}")
    else:
        # No model was trained
        best_model_path = None
        print("⚠️ No model was trained, cannot save.")
    print("⚠️ No best model found to save.")

print("✅ Fine-tuning DistilBERT with hyperparameter tuning completed.")
step3EndTime = time.time()
# ============================
# STEP 4: Extract embeddings per field from best fine-tuned model
# ============================
step4StartTime = time.time()
def extract_field_embeddings(texts, tokenizer, model_path):
    encoded = tokenizer(    
        texts, 
        padding='max_length', 
        truncation=True, 
        max_length=32, 
        return_tensors='tf'
    )
    model = TFDistilBertModel.from_pretrained(model_path)
    model.trainable = False # Freeze for extraction
    # Inference mode
    outputs = model.distilbert(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'], training=False)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.numpy()

print("\n✅ Extracting embeddings from the best fine-tuned model...")

encoder_path = best_model_path if best_model_path is not None else os.path.join(TRAINING_PATH, "models", "BEST_fine_tuned_distilBERT")

train_title_embed = extract_field_embeddings(X_train_title, tokenizer, encoder_path)
train_content_embed = extract_field_embeddings(X_train_content, tokenizer, encoder_path)
train_hashtags_embed = extract_field_embeddings(X_train_hashtags, tokenizer, encoder_path)

val_title_embed = extract_field_embeddings(X_val_title, tokenizer, encoder_path)
val_content_embed = extract_field_embeddings(X_val_content, tokenizer, encoder_path)
val_hashtags_embed = extract_field_embeddings(X_val_hashtags, tokenizer, encoder_path)

test_title_embed = extract_field_embeddings(X_test_title, tokenizer, encoder_path)
test_content_embed = extract_field_embeddings(X_test_content, tokenizer, encoder_path)
test_hashtags_embed = extract_field_embeddings(X_test_hashtags, tokenizer, encoder_path)

train_embeddings = np.concatenate([train_title_embed, train_content_embed, train_hashtags_embed], axis=1)
val_embeddings = np.concatenate([val_title_embed, val_content_embed, val_hashtags_embed], axis=1)
test_embeddings = np.concatenate([test_title_embed, test_content_embed, test_hashtags_embed], axis=1)

print("✅ Embeddings extracted and concatenated.")
step4EndTime = time.time()
# ============================
# STEP 5: Apply SMOTENN (after embeddings)
# ============================
step5StartTime = time.time()
# SMOTENN - to get resulting X_train, X_val for training
sampling = SMOTEENN(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = sampling.fit_resample(train_embeddings, y_train)
X_val_resampled, y_val_resampled = sampling.fit_resample(val_embeddings, y_val)

print("✅ SMOTENN Resampling completed.")

# Convert to Numpy arrays
y_train_resampled = np.array(y_train_resampled)
y_val_resampled = np.array(y_val_resampled)

# Check shapes - ensuring (n, 3, 768)
print(f"Resampled X_train shape: {X_train_resampled.shape}")
print(f"Resampled y_train shape: {y_train_resampled.shape}")
print(f"Resampled X_val shape: {X_val_resampled.shape}")
print(f"Resampled y_val shape: {y_val_resampled.shape}")

# Inspect as well the class distribution again
unique_train_resampled, counts_train_resampled = np.unique(y_train_resampled, return_counts=True)
print("\nClass distribution in resampled y_train:")
for cls, count in zip(unique_train_resampled, counts_train_resampled):
    print(f"Class {cls}: {count} samples")
unique_val_resampled, counts_val_resampled = np.unique(y_val_resampled, return_counts=True)
print("\nClass distribution in y_val:")
for cls, count in zip(unique_val_resampled, counts_val_resampled):
    print(f"Class {cls}: {count} samples")
step5EndTime = time.time()
# ============================
# STEP 6: Prepare datasets
# ============================
step6StartTime = time.time()

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_resampled, y_train_resampled)).batch(4)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_resampled, y_val_resampled)).batch(4)

# ============================
# Save Data for Custom Layers
# ============================
if os.path.exists(os.path.join(TRAINING_PATH, "resampled data")):
    shutil.rmtree(os.path.join(TRAINING_PATH, "resampled data"))
os.makedirs(os.path.join(TRAINING_PATH, "resampled data"))
np.save(os.path.join(TRAINING_PATH, "resampled data", "X_train_resampled.npy"), X_train_resampled)
np.save(os.path.join(TRAINING_PATH, "resampled data", "y_train_resampled.npy"), y_train_resampled)
np.save(os.path.join(TRAINING_PATH, "resampled data", "X_val_resampled.npy"), X_val_resampled)
np.save(os.path.join(TRAINING_PATH, "resampled data", "y_val_resampled.npy"), y_val_resampled)
np.save(os.path.join(TRAINING_PATH, "resampled data", "X_test.npy"), test_embeddings)
print("✅ Resampled data are saved for custom classifier.")

step6EndTime = time.time()
# # ============================
# # STEP 7: Define custom classifier model
# # ============================
# step7StartTime = time.time()

# def build_model(hp: HyperParameters):
#     inputs = tf.keras.layers.Input(shape=(X_train_resampled.shape[1],))  # (None, embedding_size)
    
#     x = tf.keras.layers.Reshape((3, -1))(inputs)

#     # Bi-directional LSTM Layer
#     x = tf.keras.layers.Bidirectional(
#         tf.keras.layers.LSTM(
#             units = hp.Choice("lstm_units", values=[64, 128, 256]),
#             return_sequences=False
#         )
#     )(x)

#     # Dropout Layer
#     x = tf.keras.layers.Dropout(
#         rate=hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
#     )(x)

#     # Dense Layer
#     x = tf.keras.layers.Dense(
#         units=hp.Int("dense_units", min_value=32, max_value=256, step=32),
#         activation="relu"
#     )(x)

#     # Output
#     outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)

#     # Build the Model with dummy data
#     model(tf.zeros((1, X_train_resampled.shape[1])))

#     # Compile the Model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=hp.Float("learning_rate", 1e-5, 1e-3, sampling="LOG")
#         ),
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")] # for sigmoid outputs with non-integer predictions
#     )
#     return model

# step7EndTime = time.time()

# # ============================
# # STEP 8: Cross-validation and hyperparameter tuning for custom classifier
# # ============================
# step8StartTime = time.time()

# FOLDS = 5

# # Clear Cache from previous run
# def clean_tuner_cache(project_name):
#     tuner_root = os.path.join(TRAINING_PATH, "models", project_name)
#     if os.path.exists(tuner_root):
#         shutil.rmtree(tuner_root)
#         print(f"✅ Deleted tuner cache at: {tuner_root}")

#     for path in ["~/.keras_tuner", "~/.keras"]:
#         full_path = os.path.expanduser(path)
#         if os.path.exists(full_path):
#             shutil.rmtree(full_path)
#             print(f"✅ Deleted global tuner cache: {full_path}")

# print("Cleaning tuner directories for fresh tuning...")
# # Monkey patch to disable project saving
# oracle.Oracle._save_trial = lambda *a, **kw: None
# oracle.Oracle._save = lambda *a, **kw: None
# # Disable loading of previous state from ~/.keras_tuner
# os.environ["KERAS_TUNER_DISABLE_LOAD"] = "true"

# for fold in range(FOLDS):
#     project_name = f"model_2_custom_classifier_fold_{fold + 1}"
#     clean_tuner_cache(project_name)

# # cross validation folds
# kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

# best_val_acc = 0.0
# best_model_path = None
# bestHistory = None
# bestFold = None
# temp_dirs = []
# print("Starting cross-validation and hyperparameter tuning for custom classifier...")
# for fold, (train_index, val_index) in enumerate(kf.split(X_train_resampled)):
#     print(f"\nCustom Classifier Fold {fold + 1}")

#     X_train_fold, X_val_fold = X_train_resampled[train_index], X_train_resampled[val_index]
#     y_train_fold, y_val_fold = y_train_resampled[train_index], y_train_resampled[val_index]

#     train_fold_dataset = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold)).batch(4)
#     val_fold_dataset = tf.data.Dataset.from_tensor_slices((X_val_fold, y_val_fold)).batch(4)

#     tuner_dir = tempfile.mkdtemp()
#     temp_dirs.append(tuner_dir)

#     tuner = Hyperband(
#         build_model,
#         objective="val_accuracy",
#         max_epochs=CUSTOM_EPOCHS,
#         directory=tuner_dir,
#         project_name=f"model_2_custom_classifier_fold_{fold + 1}"
#     )

#     # Monkey patch to bypass incompatible Keras model check
#     def patched_validate_trial_model(self, model):
#         if not isinstance(model, tf.keras.Model):
#             print("⚠️ Model is not tf.keras.Model — bypassing check anyway")
#             return
#         return

#     # keras_tuner.engine.trial.Trial._validate_trial_model = patched_validate_trial_model
#     keras_tuner.engine.base_tuner.BaseTuner._validate_trial_model = patched_validate_trial_model


#     tuner.search(
#         train_fold_dataset,
#         validation_data=val_fold_dataset,
#         epochs=CUSTOM_EPOCHS
#     )

#     best_hp = tuner.get_best_hyperparameters(1)[0]
#     print(f"✅ Best hyperparameters for fold {fold + 1}: {best_hp.values}")

#     model = build_model(best_hp)

#     # print the model's summary after complex modifications
#     print("Model summary after hyperparameter tuning:")
#     model.summary()

#     history = model.fit(train_fold_dataset, validation_data=val_fold_dataset, epochs=CUSTOM_EPOCHS)

#     val_acc = history.history['val_accuracy'][-1]
#     model_save_path = os.path.join(TRAINING_PATH, "models", f"custom_classifier_fold_{fold + 1}")
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         best_model_path = model_save_path
#         bestHistory = history
#         bestFile = fold

#     model.save(os.path.join(TRAINING_PATH, "models", f"custom_classifier_fold_{fold + 1}.h5"))

# if best_model_path:
#     print(f"Saving the best model from fold with validation accuracy: {best_val_acc}")
#     best_model_path = os.path.join(TRAINING_PATH, "models", f"BEST_custom_classifier.h5")
#     model.save(best_model_path)
#     print(f"✅ Best model saved at: {best_model_path}")

#     modelType = "Custom Layers"
#     plot(bestHistory, fold, modelType)
#     print(f"✅ Convergence plots saved for {modelType} at fold {bestFold + 1}.")
# step8EndTime = time.time()

# # ============================
# # Clean up the temp directory
# # ============================
# for temp_dir in temp_dirs:
#     shutil.rmtree(temp_dir)

# ============================
# Final Logging
# ============================
endTime = time.time()
print(f"Total Execution Time: {endTime - startTime} seconds")
print("✅ Pipeline for fine-tuning DistilBERT model has been completed.")
print(f"\nExecution Time for Step 1 (data extraction and preprocessing): {step1EndTime - step1StartTime} seconds")
print(f"Execution Time for Step 2 (randomization and split): {step2EndTime - step2StartTime} seconds")
print(f"Execution Time for Step 3 (fine-tuning DistilBERT): {step3EndTime - step3StartTime} seconds")
print(f"Execution Time for Step 4 (embedding extraction): {step4EndTime - step4StartTime} seconds")
print(f"Execution Time for Step 5 (SMOTENN): {step5EndTime - step5StartTime} seconds")
print(f"Execution Time for Step 6 (dataset preparation): {step6EndTime - step6StartTime} seconds")
#print(f"Execution Time for Step 7 (custom classifier model): {step7EndTime - step7StartTime} seconds")
#print(f"Execution Time for Step 8 (cross-validation and hyperparameter tuning): {step8EndTime - step8StartTime} seconds")