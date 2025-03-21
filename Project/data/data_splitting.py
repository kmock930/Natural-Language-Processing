import os
import pandas as pd
import numpy as np
from data_processing import normalize, vectorize

if not os.path.exists("Numpy Data"):
    os.makedirs("Numpy Data")

folders = ["Sensitive", "Non-Sensitive", "Full Dataset"]
for folder in folders:
    folder_path = os.path.join("Numpy Data", folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Defining Paths
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

train_set_full: np.ndarray = np.array([])
train_set_sensitive: np.ndarray = np.array([])
train_set_nonsensitive: np.ndarray = np.array([])

val_set_full: np.ndarray = np.array([])
val_set_sensitive: np.ndarray = np.array([])
val_set_nonsensitive: np.ndarray = np.array([])

test_set_full: np.ndarray = np.array([])
test_set_sensitive: np.ndarray = np.array([])
test_set_nonsensitive: np.ndarray = np.array([])

# Identify groups - train, validation, test sets
for path in PATHS:
    if (path.endswith("Twitter_Suicidal_Data")):
        # Load the data into pandas
        df = pd.read_csv(os.path.join(path, "twitter-suicidal_data.csv"))
        # Preprocess Post content
        for index, row in df.iterrows():
            post_content = row['tweet']
            label = int(row['intention'])
            # Normalize and vectorize the post content
            normalized_content = normalize(post_content)
            vectorized_content = vectorize(normalized_content)
            input_ids = vectorized_content['input_ids']
            attention_mask = vectorized_content['attention_mask']
            combined_vector = np.concatenate((input_ids, attention_mask))
            # Store them in Numpy arrays
            train_set_full = np.append(train_set_full, combined_vector)
            train_set_sensitive = np.append(train_set_sensitive, combined_vector)
            train_set_nonsensitive = np.append(train_set_nonsensitive, combined_vector)
            # Store labels to Numpy arrays as well
            train_set_full = np.append(train_set_full, label)
            train_set_sensitive = np.append(train_set_sensitive, label)
            train_set_nonsensitive = np.append(train_set_nonsensitive, label)
    elif (path.endswith("Social_Media_Sentiments_Analysis_Dataset")):
        # Load the data into pandas
        df_full = pd.read_csv(os.path.join(path, "sentimentdataset_annotated_binary.csv"))
        df_sensitive = pd.read_csv(os.path.join(path, "sentimentdataset_binary_class_sensitive_attributes.csv"))
        df_nonsensitive = pd.read_csv(os.path.join(path, "sentimentdataset_binary_class_non_sensitive_attributes.csv"))
        # Preprocess Post content
        for index, row in df_full.iterrows():
            post_content = row['Text'] + " " + row['Hashtags']
            label = int(row['Annotation'])
            # Normalize and vectorize the post content
            normalized_content = normalize(post_content)
            vectorized_content = vectorize(normalized_content)
            input_ids = vectorized_content['input_ids']
            attention_mask = vectorized_content['attention_mask']
            combined_vector = np.concatenate((input_ids, attention_mask))
            # Store them in Numpy arrays
            train_set_full = np.append(train_set_full, combined_vector)
            train_set_sensitive = np.append(train_set_sensitive, combined_vector)
            train_set_nonsensitive = np.append(train_set_nonsensitive, combined_vector)
            # Store labels to Numpy arrays as well
            train_set_full = np.append(train_set_full, label)
            train_set_sensitive = np.append(train_set_sensitive, label)
            train_set_nonsensitive = np.append(train_set_nonsensitive, label)
    elif (path.endswith("Reddit_SuicideWatch")):
        # Load the data into pandas
        df_full = pd.read_csv(os.path.join(path, "reddit_suicidewatch.csv"))
        df_sensitive = pd.read_csv(os.path.join(path, "reddit_suicidewatch_sensitive_attribute.csv"))
        df_nonsensitive = pd.read_csv(os.path.join(path, "reddit_suicidewatch_non_sensitive_attribute.csv"))
        # Preprocess Post content
        for index, row in df_full.iterrows():
            post_content = row["title"] + " " + row['selftext']
            label = 1 # everything here is suicidal
            # Normalize and vectorize the post content
            normalized_content = normalize(post_content)
            vectorized_content = vectorize(normalized_content)
            input_ids = vectorized_content['input_ids']
            attention_mask = vectorized_content['attention_mask']
            combined_vector = np.concatenate((input_ids, attention_mask))
            # Store them in Numpy arrays
            val_set_full = np.append(val_set_full, combined_vector)
            val_set_sensitive = np.append(val_set_sensitive, combined_vector)
            val_set_nonsensitive = np.append(val_set_nonsensitive, combined_vector)
            # Store labels to Numpy arrays as well
            val_set_full = np.append(val_set_full, label)
            val_set_sensitive = np.append(val_set_sensitive, label)
            val_set_nonsensitive = np.append(val_set_nonsensitive, label)
    elif (path.endswith("Depression_Tweets")):
        # Load the data into pandas
        df_full = df_sensitive = df_nonsensitive = pd.read_json(os.path.join(path, "depression_json"))
        # Preprocess Post content
        for index, row in df_full.iterrows():
            post_content = row["content"]
            # No ground truth label is provided in the test set
            # Normalize and vectorize the post content
            normalized_content = normalize(post_content)
            vectorized_content = vectorize(normalized_content)
            input_ids = vectorized_content['input_ids']
            attention_mask = vectorized_content['attention_mask']
            combined_vector = np.concatenate((input_ids, attention_mask))
            # Store them in Numpy arrays
            test_set_full = np.append(test_set_full, combined_vector)
            test_set_sensitive = np.append(test_set_sensitive, combined_vector)
            test_set_nonsensitive = np.append(test_set_nonsensitive, combined_vector)

# Observe data sizes
print("TRAINING DATA")
print("Train Set Full: ", train_set_full.shape)
print("Train Set Sensitive: ", train_set_sensitive.shape)
print("Train Set Non-Sensitive: ", train_set_nonsensitive.shape)

print("VALIDATION DATA")
print("Validation Set Full: ", val_set_full.shape)
print("Validation Set Sensitive: ", val_set_sensitive.shape)
print("Validation Set Non-Sensitive: ", val_set_nonsensitive.shape)

print("TEST DATA")
print("Test Set Full: ", test_set_full.shape)
print("Test Set Sensitive: ", test_set_sensitive.shape)
print("Test Set Non-Sensitive: ", test_set_nonsensitive.shape)

# Save to corresponding folders as .npy files
# TRAIN DATA
TRAIN_FILENAME = "train.npy"
np.save(os.path.join("Numpy Data", "Full Dataset", TRAIN_FILENAME), train_set_full)
np.save(os.path.join("Numpy Data", "Sensitive", TRAIN_FILENAME), train_set_sensitive)
np.save(os.path.join("Numpy Data", "Non-Sensitive", TRAIN_FILENAME), train_set_nonsensitive)

# VALIDATION DATA
VAL_FILENAME = "validation.npy"
np.save(os.path.join("Numpy Data", "Full Dataset", VAL_FILENAME), val_set_full)
np.save(os.path.join("Numpy Data", "Sensitive", VAL_FILENAME), val_set_sensitive)
np.save(os.path.join("Numpy Data", "Non-Sensitive", VAL_FILENAME), val_set_nonsensitive)

# TEST DATA
TEST_FILENAME = "test.npy"
np.save(os.path.join("Numpy Data", "Full Dataset", TEST_FILENAME), test_set_full)
np.save(os.path.join("Numpy Data", "Sensitive", TEST_FILENAME), test_set_sensitive)
np.save(os.path.join("Numpy Data", "Non-Sensitive", TEST_FILENAME), test_set_nonsensitive)

# Check unique values in the second item of the numpy arrays (i.e., labels)
unique_train_full = np.unique(train_set_full[1::2])
unique_train_sensitive = np.unique(train_set_sensitive[1::2])
unique_train_nonsensitive = np.unique(train_set_nonsensitive[1::2])

unique_val_full = np.unique(val_set_full[1::2])
unique_val_sensitive = np.unique(val_set_sensitive[1::2])
unique_val_nonsensitive = np.unique(val_set_nonsensitive[1::2])

unique_test_full = np.unique(test_set_full[1::2])
unique_test_sensitive = np.unique(test_set_sensitive[1::2])
unique_test_nonsensitive = np.unique(test_set_nonsensitive[1::2])

print("Unique values in the second item (LABELS) of train_set_full:", unique_train_full)
print("Unique values in the second item (LABELS) of train_set_sensitive:", unique_train_sensitive)
print("Unique values in the second item (LABELS) of train_set_nonsensitive:", unique_train_nonsensitive)

print("Unique values in the second item (LABELS) of val_set_full:", unique_val_full)
print("Unique values in the second item (LABELS) of val_set_sensitive:", unique_val_sensitive)
print("Unique values in the second item (LABELS) of val_set_nonsensitive:", unique_val_nonsensitive)

print("Unique values in the second item (LABELS) of test_set_full:", unique_test_full)
print("Unique values in the second item (LABELS) of test_set_sensitive:", unique_test_sensitive)
print("Unique values in the second item (LABELS) of test_set_nonsensitive:", unique_test_nonsensitive)

TOTAL_RECORDS_TRAIN = 732 + 9119
TOTAL_RECORDS_VAL = 100
TOTAL_RECORDS_TEST = 18679
assert len(train_set_full) == len(train_set_sensitive) == len(train_set_nonsensitive) == TOTAL_RECORDS_TRAIN
assert len(val_set_full) == len(val_set_sensitive) == len(val_set_nonsensitive) == TOTAL_RECORDS_VAL
assert len(test_set_full) == len(test_set_sensitive) == len(test_set_nonsensitive) == TOTAL_RECORDS_TEST