class Lemmatizer:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet

    lemmatizer = None

    nltk_packages_installed = False

    def install_nltk_packages(self, nltk_packages_installed):
        if not nltk_packages_installed:
            linux_packages = ['averaged_perceptron_tagger_eng', 'punkt', 'wordnet', 'omw-1.4', 'stopwords']
            windows_packages = ['wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
            for resource in linux_packages:
                try:
                    self.nltk.data.find(f'taggers/{resource}') if resource.startswith('averaged_perceptron_tagger') else self.nltk.data.find(f'corpora/{resource}')
                except LookupError:
                    self.nltk.download(resource)
            nltk_packages_installed = True

    def __init__(self):
        if (self.lemmatizer is None):
            self.lemmatizer = self.WordNetLemmatizer()
        
        if (self.nltk_packages_installed is False):
            self.install_nltk_packages(self.nltk_packages_installed)
            self.nltk_packages_installed = True

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = self.nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": self.wordnet.ADJ,
                    "N": self.wordnet.NOUN,
                    "V": self.wordnet.VERB,
                    "R": self.wordnet.ADV}
        return tag_dict.get(tag[0].upper(), self.wordnet.NOUN)

    def lemmatize(self, corpus):
        lemmatized_corpus = []
        for sentence in self.nltk.sent_tokenize(corpus):
            lemmatized_sentence = []
            for word in self.nltk.word_tokenize(sentence):
                lemmatized_word = self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word))
                lemmatized_sentence.append(lemmatized_word)
            lemmatized_corpus.append(' '.join(lemmatized_sentence))
        return ' '.join(lemmatized_corpus)