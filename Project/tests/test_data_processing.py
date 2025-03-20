import unittest
import sys
import os
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(PATH)
from data_processing import normalize, normalize_emojis, normalize_symbols, normalize_punctuation

class TestDataProcessing(unittest.TestCase):
    def check_matches(self, targetText, originalText):
        for word in targetText.split():
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
            targetText="hello world this is a test with emojis symbols and punctuations",
            originalText=normalize(
                "Hello, world! This is a test with emojis 😊, symbols, and punctuations@ # http://example.com"
            )
        )

if __name__ == '__main__':
    unittest.main()