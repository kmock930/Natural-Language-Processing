import re

class WordTokenizer:
    word_pattern: str = r"\b\w+(?:'\w+)?\b"

    corpus: str = ""

    def __init__(self, corpus: str = ""):
        self.corpus = corpus

    def tokenize(self, text):
        return re.findall(self.word_pattern, text)

    def countOccurrences(self, text: str = ""):
        '''
        Counts the number of occurrences of each token in a given corpus
        @param text: the text to tokenize
        @return: a dictionary with each token as the key and the number of occurrences as the value
        '''
        tokens: list = self.tokenize(text)
        occurrences: dict = {}
        for token in tokens:
            if token in occurrences:
                continue
            else:
                occurrences[token] = WordTokenizer.count_word(self.corpus, token)
        return occurrences

    @staticmethod
    def count_word(text, word):
        '''
        Counts the number of occurrences of a specific word in a given corpus
        @param text: the text to tokenize
        @param word: the word to count
        @return: the number of occurrences of the word
        '''
        tokens: list = WordTokenizer().tokenize(text)
        return tokens.count(word)