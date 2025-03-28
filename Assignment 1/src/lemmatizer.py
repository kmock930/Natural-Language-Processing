import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        linux_packages = ['averaged_perceptron_tagger_eng', 'punkt', 'wordnet', 'omw-1.4', 'stopwords']
        windows_packages = ['wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
        for resource in linux_packages:
            try:
                nltk.data.find(f'taggers/{resource}') if resource.startswith('averaged_perceptron_tagger') else nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)

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