import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from lemmatizer import Lemmatizer

class TestLemmatizer(unittest.TestCase):

    def setUp(self):
        self.lemmatizer = Lemmatizer()

    def test_get_wordnet_pos(self):
        self.assertEqual(self.lemmatizer.get_wordnet_pos("running"), "v")
        self.assertEqual(self.lemmatizer.get_wordnet_pos("bats"), "n")
        self.assertEqual(self.lemmatizer.get_wordnet_pos("striped"), "v")
        self.assertEqual(self.lemmatizer.get_wordnet_pos("best"), "a")
    
    def test_lemmatize(self):
        corpus = "The striped bats are hanging on their feet for best"
        self.assertEqual(self.lemmatizer.lemmatize(corpus), "The strip bat be hang on their foot for best")


if __name__ == '__main__':
    unittest.main()
