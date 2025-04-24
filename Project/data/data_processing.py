import emoji
import sys
import os
PATH_TO_ASSIGNMENT1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Assignment 1', 'src'))
sys.path.append(PATH_TO_ASSIGNMENT1)
from word_tokenizer import WordTokenizer
from lemmatizer import Lemmatizer
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder # for multi-class
import pandas as pd
import numpy as np

def encode_labels(labels: list[str] | pd.Series, encoder: LabelEncoder = None) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode the given labels to integers.

    Args:
        labels (list[str] | pandas.Series): The labels to be encoded.
        encoder (LabelEncoder, optional): The encoder to be used to encode the labels. Defaults to None.

    Returns:
        tuple[numpy.ndarray, LabelEncoder]: The encoded labels and the encoder used to encode the labels.
    
    Author:
        Kelvin Mock
    """
    encoder = LabelEncoder() if encoder is None else encoder
    labels_reshaped = np.array(labels)
    encoded_labels = encoder.fit_transform(labels_reshaped)
    return encoded_labels, encoder

def decode_labels(encoder: LabelEncoder, encoded_labels: np.ndarray) -> np.ndarray:
    """
    Decode the given labels from integers.

    Args:
        encoder: LabelEncoder: The FITTED encoder used to encode the labels.
        encoded_labels (numpy.ndarray): The encoded labels to be decoded.

    Returns:
        list[str]: The decoded labels.
    
    Author:
        Kelvin Mock
    """
    decoded_labels = encoder.inverse_transform(encoded_labels)
    return decoded_labels

# Code inspired by https://medium.com/@ebimsv/nlp-series-day-5-handling-emojis-strategies-and-code-implementation-0f8e77e3a25c
def normalize_emojis(text: str) -> str:
    """
    Normalize emojis in the given text.

    Args:
        text (str): The text containing emojis to be normalized.

    Returns:
        str: The text with emojis normalized to their textual representation.
    
    Author:
        Kelvin Mock
    """
    return emoji.demojize(text)

# Code inspired by https://ai.plainenglish.io/how-to-perform-hashtag-analysis-using-natural-language-processing-and-machine-learning-in-python-ea6da817b3a4
def normalize_symbols(text: str) -> str:
    """
    Normalize symbols in the given text by removing URLs, mentions, and hashtags.

    Args:
        text (str): The text containing symbols to be normalized.

    Returns:
        str: The text with URLs, mentions, and hashtags removed.

    Author:
        Kelvin Mock
    """
    return ' '.join(word for word in text.split() if not (word.startswith('http') or word.startswith('@') or word.startswith('#')))

# Code inspired by https://ai.plainenglish.io/how-to-perform-hashtag-analysis-using-natural-language-processing-and-machine-learning-in-python-ea6da817b3a4
def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation in the given text by removing all punctuation marks and returning the text in lowercase.

    Args:
        text (str): The text containing punctuation to be normalized.

    Returns:
        str: The text with all punctuation marks removed."
    
    Author:
        Kelvin Mock
    """
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    text = text.lower()
    return text

# Code inherited from Assignment 1
def normalize_stopwords(text: list[str]) -> str:
    """
    Normalize stopwords in the given text by removing all stopwords.

    Args:
        text (list[str]): The text containing stopwords to be normalized.

    Returns:
        list[str]: The text with all stopwords removed.
    
    Author:
        Kelvin Mock
    """
    stopwordsFilePath = os.path.join(PATH_TO_ASSIGNMENT1, 'StopWords.txt')
    stopwordsFile = open(stopwordsFilePath, 'r')
    stopwords:list = [line.strip() for line in stopwordsFile.readlines()]
    stopwordsFile.close()
    print(f"Number of Stopwords considered: {len(stopwords)}")
    filtered_text: list[str] = [word for word in text if word not in stopwords]
    return filtered_text

# Code inherited from Assignment 1
def lemmatize(text: str) -> str:
    """
        Lemmatizes the input text.

        Args:
            text (str): The text to be lemmatized.
        
        Returns:
            str: The lemmatized version of the input text.
        
        Author:
            Kelvin Mock
    """
    lemmatizer = Lemmatizer()
    return lemmatizer.lemmatize(text)

# Code inspired by Assignment 2 - Deep Learning Approach (DistilBERT)
def vectorize(list_of_texts: list[str]):
    """
    Vectorize the given list of texts.

    Args:
        list_of_texts (list[str]): The list of texts to be vectorized.

    Returns:
        numpy.ndarray: The vectorized list of texts.
    
    Author:
        Kelvin Mock
    """
    # Load the tokernizer of a pretrained model
    # https://huggingface.co/distilbert/distilbert-base-uncased
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_input = tokenizer(
        list_of_texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='tf'
    )
    return encoded_input

def normalize(text: str) -> str:
    """
    Normalize the given text with all possible methods.

    Args:
        text (str): The text to be normalized.

    Returns:
        str: The normalized text.
    
    Author:
        Kelvin Mock
    """
    text = normalize_emojis(text)
    text = normalize_symbols(text)
    text = normalize_punctuation(text)

    # Lemmatize: to replace abstract words with its base form
    text = lemmatize(text)

    # tokenization
    wordTokenizer = WordTokenizer()
    tokenized_text:list[str] = wordTokenizer.tokenize(text=text)

    tokenized_text = normalize_stopwords(tokenized_text)
    print(tokenized_text)
    return text