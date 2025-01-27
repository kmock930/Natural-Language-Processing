# Natural Language Processing Work
## Assignment 1: Corpus analysis and sentence embeddings
All codes are in the directory named `Assignment 1`.
### Prerequisite Files:
* Dataset - the given corpus: [The Atticus dataset of legal contacts](https://zenodo.org/record/4595826#.YyXT6HbMI2w)
* A text file containing a list of stopwords (to evaluate the lexical density): [`StopWords.txt`](http://www.site.uottawa.ca/~diana/csi5180/StopWords)
* Note: All the prerequisite files are placed the archive at `Assignment 1/src/` directory.
### Part 1: Corpus processing (legal text): tokenization and word counting
1. Implementation of our Word Tokenizer
2. Extract the given corpus.
3. Concatenate text files to form a corpus.
4. Analyze statistics about tokens in the corpus.
* Note: Unlike source codes which are stored at `src` directory, unit tests are stored under `tests` directory. 
### Execution of Codes
#### Analytical Results
Please run all cells in the Jupyter Notebook named `Part 1 - Corpus Processing.ipynb` sequentially.
#### Word Tokenizer
* Please run the test file named `tests/test_word_tokenizer.py`. 
* Please also run another test file named `tests/test_count_occurences.py`. 
#### Experimental Lemmantizer
Please run the test file named `tests/test_lemmatizer.py`. 