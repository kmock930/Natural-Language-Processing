import unittest
import sys
import os
import pandas as pd
import numpy as np

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(PATH)
from data_processing import normalize, normalize_emojis, normalize_symbols, normalize_punctuation, normalize_stopwords, vectorize, encode_labels, decode_labels
from sklearn import svm

class TestDataProcessing(unittest.TestCase):
    def check_matches(self, targetText, originalText):
        if type(targetText) == str:
            targetText = targetText.split()
        for word in targetText:
            self.assertIn(word, originalText)

    def test_normalize_emojis(self):
        self.check_matches(
            targetText="Hello", 
            originalText=normalize_emojis("Hello 😊")
        )
        self.check_matches(
            targetText="No emojis here!", 
            originalText=normalize_emojis("No emojis here!")
        )
        self.check_matches(
            targetText="Multiple emojis", 
            originalText=normalize_emojis("Multiple emojis 😂😂😂")
        )
        self.check_matches(
            targetText="Mixed content text", 
            originalText=normalize_emojis("Mixed content 😃 text 😢")
        )
        self.check_matches(
            targetText="", 
            originalText=normalize_emojis("")
        )
        self.check_matches(
            targetText="", 
            originalText=normalize_emojis("Only emojis 😜😜😜")
        )

    def test_normalize_symbols(self):
        self.check_matches(
            targetText="Hello", 
            originalText=normalize_symbols("Hello")
        )
        self.check_matches(
            targetText="No symbols here!", 
            originalText=normalize_symbols("No symbols here!")
        )
        self.check_matches(
            targetText="Multiple symbols", 
            originalText=normalize_symbols("Multiple symbols @ # http")
        )
        self.check_matches(
            targetText="Mixed content text", 
            originalText=normalize_symbols("Mixed content @ text #")
        )
        self.check_matches(
            targetText="", 
            originalText=normalize_symbols("")
        )
        self.check_matches(
            targetText="Only symbols", 
            originalText=normalize_symbols("Only symbols @ # http")
        )

        self.check_matches(
            targetText="Check URL parsing", 
            originalText=normalize_symbols("Check URL parsing http://example.com")
        )

    def test_normalize_punctuation(self):
        self.check_matches(
            targetText="hello world", 
            originalText=normalize_punctuation("Hello, world!")
        )
        self.check_matches(
            targetText="no punctuation here", 
            originalText=normalize_punctuation("No punctuation here")
        )
        self.check_matches(
            targetText="multiple punctuations", 
            originalText=normalize_punctuation("Multiple, punctuations!!!")
        )
        self.check_matches(
            targetText="mixed content text", 
            originalText=normalize_punctuation("Mixed content, text.")
        )
        self.check_matches(
            targetText="", 
            originalText=normalize_punctuation("")
        )
        self.check_matches(
            targetText="only alphanumeric characters 123", 
            originalText=normalize_punctuation("Only alphanumeric characters 123")
        )

    def test_hybrid_case(self):
        self.check_matches(
            originalText=normalize(
                "Hello, world! This is a test with emojis 😊, symbols, and punctuations@ # http://example.com"
            ),
            targetText=["hello", "world", "test", "emoji", "symbol", "and", "punctuation"]
        )

    def test_normalize_stopwords(self):
        # no stopwords
        self.check_matches(
            targetText="Hello World", 
            originalText=normalize_stopwords(["Hello", "World"])
        )
        # contains stopwords
        self.check_matches(
            targetText="Moon", 
            originalText=normalize_stopwords(["Above", "The", "Moon"])
        )
        # no words
        self.check_matches(
            targetText="", 
            originalText=normalize_stopwords([])
        )

    def test_vectorize(self):
        texts = ["Hello world!", "This is a test.", "Natural Language Processing with DistilBERT.", "😊"]
        vectorized_output = vectorize(texts)
        MAX_TEXT_LENGTH = 32 # max length should be no more than 32
        self.assertEqual(vectorized_output['input_ids'].shape[0], len(texts))
        self.assertEqual(vectorized_output['attention_mask'].shape[0], len(texts))
        self.assertLessEqual(vectorized_output['input_ids'].shape[1], MAX_TEXT_LENGTH)
        self.assertLessEqual(vectorized_output['attention_mask'].shape[1], MAX_TEXT_LENGTH)

    def test_encode_decode(self):
        PATH_TO_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Social_Media_Sentiments_Analysis_Dataset'))
        df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'sentimentdataset_annotated_binary.csv'))
        platform_series = df['Platform']
        self.assertIsInstance(platform_series, pd.Series)
        uniqueColumns: list[str] = platform_series.unique()

        # one hot encode process
        encoded_platform, encoder = encode_labels(platform_series)
        uniqueEncodedLabels = np.unique(encoded_platform)

        # decode labels
        decoded_platform = decode_labels(encoder=encoder, encoded_labels=encoded_platform)
        uniqueDecodedPlatforms = np.unique(decoded_platform)

        self.assertEqual(len(uniqueColumns), len(uniqueDecodedPlatforms))
        self.assertEqual(len(uniqueColumns), len(uniqueEncodedLabels))
        for column in uniqueColumns:
            self.assertIn(column, uniqueDecodedPlatforms)

    def test_reuse_fitted_encoder(self):
        PATH_TO_DATASET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'Social_Media_Sentiments_Analysis_Dataset'))
        df = pd.read_csv(os.path.join(PATH_TO_DATASET, 'sentimentdataset_annotated_binary.csv'))
        platform_series = df['Platform']
        self.assertIsInstance(platform_series, pd.Series)
        uniqueColumns: list[str] = platform_series.unique()

        # one hot encode process
        encoded_platform, encoder = encode_labels(platform_series)
        uniqueEncodedLabels = np.unique(encoded_platform)

        # decode labels
        decoded_platform = decode_labels(encoder=encoder, encoded_labels=encoded_platform)
        uniqueDecodedPlatforms = np.unique(decoded_platform)

        self.assertEqual(len(uniqueColumns), len(uniqueDecodedPlatforms))
        self.assertEqual(len(uniqueColumns), len(uniqueEncodedLabels))
        for column in uniqueColumns:
            self.assertIn(column, uniqueDecodedPlatforms)

        # reuse the fitted encoder
        encoded_platform, encoder = encode_labels(platform_series, encoder)
        uniqueEncodedLabels = np.unique(encoded_platform)

        # decode labels
        decoded_platform = decode_labels(encoder=encoder, encoded_labels=encoded_platform)
        uniqueDecodedPlatforms = np.unique(decoded_platform)

        self.assertEqual(len(uniqueColumns), len(uniqueDecodedPlatforms))
        self.assertEqual(len(uniqueColumns), len(uniqueEncodedLabels))
        for column in uniqueColumns:
            self.assertIn(column, uniqueDecodedPlatforms)

if __name__ == '__main__':
    unittest.main()