import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from word_tokenizer import WordTokenizer

class TestWordTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = WordTokenizer()

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])

    def test_single_word(self):
        self.assertEqual(self.tokenizer.tokenize("hello"), ["hello"])

    def test_multiple_words(self):
        self.assertEqual(self.tokenizer.tokenize("hello world"), ["hello", "world"])

    def test_punctuation(self):
        self.assertEqual(self.tokenizer.tokenize("hello, world!"), ["hello", "world"])

    def test_numbers(self):
        self.assertEqual(self.tokenizer.tokenize("hello 123"), ["hello", "123"])

    def test_mixed_case(self):
        self.assertEqual(self.tokenizer.tokenize("Hello World"), ["Hello", "World"])

    def test_with_newlines(self):
        self.assertEqual(self.tokenizer.tokenize("hello\nworld"), ["hello", "world"])

    def test_with_tabs(self):
        self.assertEqual(self.tokenizer.tokenize("hello\tworld"), ["hello", "world"])

    def test_with_multiple_spaces(self):
        self.assertEqual(self.tokenizer.tokenize("hello    world"), ["hello", "world"])

    def test_with_apostrophes(self):
        self.assertEqual(self.tokenizer.tokenize("I'm fine thank-you!"), ["I'm", "fine", "thank", "you"])

if __name__ == '__main__':
    unittest.main()