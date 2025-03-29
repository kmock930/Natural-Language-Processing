import os
import pandas as pd
import numpy as np
from data_processing import normalize, vectorize, encode_labels
import joblib
import tensorflow as tf
from transformers import TFDistilBertModel
# Load the Model (via inference) to obtain embeddings - encoded input AND attention masks
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# ==== DIRECTORY SETUP ====
folders = [
    "Numpy Data/Text",
    "Numpy Data/Metadata/Full Dataset",
    "Numpy Data/Metadata/Sensitive",
    "Numpy Data/Metadata/Non-Sensitive",
    "Numpy Data/Encoders"
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ==== TEXT VECTORIZATION HELPERS ====
def vectorize_text_field(text: str) -> np.ndarray:
    if not isinstance(text, str):
        text = ""
    inputs = vectorize([text])  # returns input_ids and attention_mask
    outputs = model(inputs)
    mask = tf.cast(tf.expand_dims(inputs['attention_mask'], axis=-1), dtype=outputs.last_hidden_state.dtype)
    masked = outputs.last_hidden_state * mask
    pooled = tf.reduce_sum(masked, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1)
    return pooled[0].numpy()

def build_text_embedding_vector(title, content, hashtags):
    return np.stack([
        vectorize_text_field(normalize(title)),
        vectorize_text_field(normalize(content)),
        vectorize_text_field(normalize(hashtags))
    ])  # (3, 768)

# ==== METADATA ENCODING ====
def encode_metadata(df: pd.DataFrame, exclude_cols: list[str], encoder=None):
    encoded_rows = []
    for _, row in df.iterrows():
        row_data = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            val = row[col]
            if pd.api.types.is_numeric_dtype(df[col]):
                row_data.append(float(val) if pd.notna(val) else 0.0)
            else:
                label_array, encoder = encode_labels([val], encoder)
                row_data.append(float(label_array[0]))
        encoded_rows.append(row_data)
    metadata_array = np.array(encoded_rows)
    metadata_array = np.expand_dims(metadata_array, axis=-1)  # (n, num_features, 1)
    return metadata_array, encoder

# ==== PROCESS AND SAVE ====
def process_and_save(split: str, text_array, meta_full, meta_sensitive, meta_nonsensitive, labels):
    np.save(f"Numpy Data/Text/X_{split}_text.npy", text_array)
    np.save(f"Numpy Data/Metadata/Full Dataset/X_{split}_metadata.npy", meta_full)
    np.save(f"Numpy Data/Metadata/Sensitive/X_{split}_metadata.npy", meta_sensitive)
    np.save(f"Numpy Data/Metadata/Non-Sensitive/X_{split}_metadata.npy", meta_nonsensitive)
    if labels is not None:
        np.save(f"Numpy Data/y_{split}_text.npy", labels)
    else:
        print(f"⚠️ No labels provided for {split} set.")
    print(f"✅ Saved {split} set: text {text_array.shape}, metadata {meta_full.shape}, labels {labels.shape if labels is not None else None}")


# Defining Paths to Datasets
__file__ = os.path.abspath(os.curdir)
PATHS = [
    # TRAINING DATA 
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'Twitter_Suicidal_Data')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'Social_Media_Sentiments_Analysis_Dataset')),
    # VALIDATION DATA
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'Reddit_SuicideWatch')),
    # TEST DATA
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'Depression_Tweets'))
]

# TRAIN
# Text Embeddings
train_set_text = []
# Metadata
train_set_metadata_full = []
train_set_metadata_sensitive = []
train_set_metadata_nonsensitive = []
y_train = []

# VALIDATION
# Text Embeddings
val_set_text = []
# Metadata
val_set_metadata_full = []  
val_set_metadata_sensitive = []
val_set_metadata_nonsensitive = []
y_val = []

# TEST
# Text Embeddings
test_set_text = []
# Metadata
test_set_metadata_full = []
test_set_metadata_sensitive = []
test_set_metadata_nonsensitive = []
y_test = []


# Identify groups - train, validation, test sets
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
            text_vector = build_text_embedding_vector(title, post_content, hashtags)
            train_set_text.append(text_vector)

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
            text_vector = build_text_embedding_vector(title, post_content, hashtags)
            train_set_text.append(text_vector)

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
            text_vector = build_text_embedding_vector(title, post_content, hashtags)
            val_set_text.append(text_vector)

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

            text_vector = build_text_embedding_vector(title, post_content, hashtags)
            test_set_text.append(text_vector)

            # No Ground Truth Labels
                

# Extract Metadata features
# Social_Media_Sentiments_Analysis_Dataset
# Encode categorical columns
def process_columns(dataset: str):
    # Intializing structures
    encoder = None
    train_set_metadata_full = []
    train_set_metadata_sensitive = []
    train_set_metadata_nonsensitive = []
    val_set_metadata_full = []
    val_set_metadata_sensitive = []
    val_set_metadata_nonsensitive = []
    test_set_metadata_full = []
    test_set_metadata_sensitive = []
    test_set_metadata_nonsensitive = []

    match (dataset):
        case "Twitter_Suicidal_Data":
            # Exclude "tweet" column, encode all other categorical features
            excluded_columns = ['tweet']
            full_metadata, encoder = encode_metadata(df_full, exclude_cols=excluded_columns)
            sensitive_metadata, _ = encode_metadata(df_sensitive, exclude_cols=excluded_columns, encoder=encoder)
            nonsensitive_metadata, _ = encode_metadata(df_nonsensitive, exclude_cols=excluded_columns, encoder=encoder)

            train_set_metadata_full.append(full_metadata)
            train_set_metadata_sensitive.append(sensitive_metadata)
            train_set_metadata_nonsensitive.append(nonsensitive_metadata)
        case "Social_Media_Sentiments_Analysis_Dataset":
            # Exclude "Text" and "Hashtags" columns, encode all other categorical features
            excluded_columns = ['Text', 'Hashtags']
            full_metadata, encoder = encode_metadata(df_full, exclude_cols=excluded_columns)
            sensitive_metadata, _ = encode_metadata(df_sensitive, exclude_cols=excluded_columns, encoder=encoder)
            nonsensitive_metadata, _ = encode_metadata(df_nonsensitive, exclude_cols=excluded_columns, encoder=encoder)

            train_set_metadata_full.append(full_metadata)
            train_set_metadata_sensitive.append(sensitive_metadata)
            train_set_metadata_nonsensitive.append(nonsensitive_metadata)
        case "Reddit_SuicideWatch":
            # Exclude "selftext" and "title" columns, encode all other categorical features
            excluded_columns = ['title', 'selftext']
            full_metadata, encoder = encode_metadata(df_full, exclude_cols=excluded_columns)
            sensitive_metadata, _ = encode_metadata(df_sensitive, exclude_cols=excluded_columns, encoder=encoder)
            nonsensitive_metadata, _ = encode_metadata(df_nonsensitive, exclude_cols=excluded_columns, encoder=encoder)

            val_set_metadata_full.append(full_metadata)
            val_set_metadata_sensitive.append(sensitive_metadata)
            val_set_metadata_nonsensitive.append(nonsensitive_metadata)
        case "Depression_Tweets":
            # Exclude "content" column, encode all other categorical features
            excluded_columns = ['content']
            full_metadata, encoder = encode_metadata(df_full, exclude_cols=excluded_columns)
            sensitive_metadata, _ = encode_metadata(df_sensitive, exclude_cols=excluded_columns, encoder=encoder)
            nonsensitive_metadata, _ = encode_metadata(df_nonsensitive, exclude_cols=excluded_columns, encoder=encoder)
            
            test_set_metadata_full.append(full_metadata)
            test_set_metadata_sensitive.append(sensitive_metadata)
            test_set_metadata_nonsensitive.append(nonsensitive_metadata)

    # Save the LabelEncoder model
    if (encoder is not None):
        joblib.dump(encoder, os.path.join("Numpy Data", "Encoders", f"{dataset}_label_encoder.pkl"))
    else:
        # Won't save if encoder is None
        # No Exception will be raised
        print(f"No LabelEncoder model to save for dataset: {dataset}")
    
    print(f"Metadata Processing Completed at Dataset: {dataset}")

    match (dataset):
        case "Twitter_Suicidal_Data":
            return train_set_metadata_full, train_set_metadata_sensitive, train_set_metadata_nonsensitive
        case "Social_Media_Sentiments_Analysis_Dataset":
            return train_set_metadata_full, train_set_metadata_sensitive, train_set_metadata_nonsensitive
        case "Reddit_SuicideWatch":
            return val_set_metadata_full, val_set_metadata_sensitive, val_set_metadata_nonsensitive
        case "Depression_Tweets":
            return test_set_metadata_full, test_set_metadata_sensitive, test_set_metadata_nonsensitive

train_set_metadata_full, train_set_metadata_sensitive, train_set_metadata_nonsensitive = process_columns(
    dataset="Twitter_Suicidal_Data"
)
train_set_metadata_full += process_columns(
    dataset="Social_Media_Sentiments_Analysis_Dataset"
)[0]
train_set_metadata_sensitive += process_columns(
    dataset="Social_Media_Sentiments_Analysis_Dataset"
)[1]
train_set_metadata_nonsensitive += process_columns(
    dataset="Social_Media_Sentiments_Analysis_Dataset"
)[2]

val_set_metadata_full, val_set_metadata_sensitive, val_set_metadata_nonsensitive = process_columns(
    dataset="Reddit_SuicideWatch"
)

test_set_metadata_full, test_set_metadata_sensitive, test_set_metadata_nonsensitive = process_columns(
    dataset="Depression_Tweets"
)

# Convert lists to numpy arrays
train_set_text = np.array(train_set_text)
val_set_text = np.array(val_set_text)
test_set_text = np.array(test_set_text)

y_train = np.array(y_train)
y_val = np.array(y_val)

train_set_metadata_full = np.concatenate(train_set_metadata_full, axis=0) 
train_set_metadata_sensitive = np.concatenate(train_set_metadata_sensitive, axis=0)
train_set_metadata_nonsensitive = np.concatenate(train_set_metadata_nonsensitive, axis=0)

val_set_metadata_full = np.concatenate(val_set_metadata_full, axis=0)
val_set_metadata_sensitive = np.concatenate(val_set_metadata_sensitive, axis=0)
val_set_metadata_nonsensitive = np.concatenate(val_set_metadata_nonsensitive, axis=0)

test_set_metadata_full = np.concatenate(test_set_metadata_full, axis=0)
test_set_metadata_sensitive = np.concatenate(test_set_metadata_sensitive, axis=0)
test_set_metadata_nonsensitive = np.concatenate(test_set_metadata_nonsensitive, axis=0)

# Save the arrays to .npy files
# TRAIN: Twitter Suicidal Data, Social Media Sentiments Analysis Dataset
process_and_save(
    split="train",
    text_array=train_set_text,
    meta_full=train_set_metadata_full,
    meta_sensitive=train_set_metadata_sensitive,
    meta_nonsensitive=train_set_metadata_nonsensitive,
    labels=y_train
)

# Val: Reddit SuicideWatch
process_and_save(
    split="val",
    text_array=val_set_text,
    meta_full=val_set_metadata_full,
    meta_sensitive=val_set_metadata_sensitive,
    meta_nonsensitive=val_set_metadata_nonsensitive,
    labels=y_val
)

# Test: Depression Tweets
process_and_save(
    split="test",
    text_array=test_set_text,
    meta_full=test_set_metadata_full,
    meta_sensitive=test_set_metadata_sensitive,
    meta_nonsensitive=test_set_metadata_nonsensitive,
    labels=None
)

print("✅ All datasets processed and saved as NumPy arrays.")