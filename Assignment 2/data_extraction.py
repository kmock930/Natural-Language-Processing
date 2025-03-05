import json
import pandas as pd
import constants

def load_jsonl(file_path):
    """
    Loads a JSONL file into a pandas DataFrame while handling decode errors.
    
    Author: Jenifer Yu

    Returns: 
        pd.DataFrame: A DataFrame containing the JSONL file data.
    """
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding line {i}: {e}")
                continue
    return pd.DataFrame(data)

def get_raw_dataset(mode: str = 'train'):
    """
    Load the raw dataset from the JSONL file based on the specified mode.

    Author: Kelvin Mock
    
    Args:
        mode (str): The mode of the dataset to load. Options are 'train', 'dev', or 'test'.
    
    Returns:
        tuple: A tuple containing the dataset based on the mode:
            - For 'train': (X_train (pd.Series), y_train (pd.Series))
            - For 'dev': (X_dev (pd.Series), y_dev (pd.Series))
            - For 'test': (X_test (pd.Series), ids_test (pd.Series))
    """
    match (mode):
        case 'train':
            # Load training data
            df_train = load_jsonl(constants.TRAIN_FILEPATH)
            X_train = df_train['text']
            y_train = df_train['label']
            return X_train, y_train
        case 'dev':
            # Load dev data
            df_dev = load_jsonl(constants.DEV_FILEPATH)
            X_dev = df_dev['text']
            y_dev = df_dev['label']
            return X_dev, y_dev
        case 'test':
            # Load test data
            df_test = load_jsonl(constants.TEST_FILEPATH)
            X_test = df_test['text']
            ids_test = df_test['id']
            return X_test, ids_test
        case _: 
            return get_raw_dataset('train')