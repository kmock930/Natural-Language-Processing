import re

class WordTokenizer:
    word_pattern: str = r'\b\w+\b'

    def tokenize(self, text):
        return re.findall(self.word_pattern, text)