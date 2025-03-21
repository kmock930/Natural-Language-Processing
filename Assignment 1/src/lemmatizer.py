import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Run on Linux
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('stopwords')

        # Run on Windows
        #nltk.download('wordnet')
        #nltk.download('averaged_perceptron_tagger')
        #nltk.download('punkt_tab')

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag[0].upper(), wordnet.NOUN)

    def lemmatize(self, corpus):
        lemmatized_corpus = []
        for sentence in nltk.sent_tokenize(corpus):
            lemmatized_sentence = []
            for word in nltk.word_tokenize(sentence):
                lemmatized_word = self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                lemmatized_sentence.append(lemmatized_word)
            lemmatized_corpus.append(' '.join(lemmatized_sentence))
        return ' '.join(lemmatized_corpus)