import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from word_tokenizer import WordTokenizer

class TestWordTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = WordTokenizer("This is a test. This test is only a test.")

    def test_countOccurrences(self):
        text = "This is a test. This test is only a test."
        expected_output = {
            'This': 2,
            'is': 2,
            'a': 2,
            'test': 3,
            'only': 1
        }
        self.assertEqual(self.tokenizer.countOccurrences(text), expected_output)

    def test_count_word(self):
        text = "This is a test. This test is only a test."
        self.assertEqual(WordTokenizer.count_word(text, 'test'), 3)
        self.assertEqual(WordTokenizer.count_word(text, 'This'), 2)
        self.assertEqual(WordTokenizer.count_word(text, 'only'), 1)
        self.assertEqual(WordTokenizer.count_word(text, 'not_in_text'), 0)

if __name__ == '__main__':
    unittest.main()
