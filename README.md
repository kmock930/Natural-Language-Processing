# Natural Language Processing Work
## Assignment 1: Corpus analysis and sentence embeddings
All codes are in the directory named `Assignment 1`.

Check our our [Report](https://github.com/kmock930/Natural-Language-Processing/blob/main/Assignment%201/CSI5386%20NLP%20Assignment%201%20Report%20-%20Kelvin%2C%20Jenifer%2C%20Sabrina.pdf)
---

### Prerequisite Files:
* Dataset - the given corpus: [The Atticus dataset of legal contacts](https://zenodo.org/record/4595826#.YyXT6HbMI2w)
* A text file containing a list of stopwords (to evaluate the lexical density): [`StopWords.txt`](http://www.site.uottawa.ca/~diana/csi5180/StopWords)
* To install all dependencies, run this command with your terminal under `src` and `tests` directories respectively: `pip install -r requirements.txt`
* Note: All the prerequisite files are placed the archive at `Assignment 1/src/` directory.
---

### Part 1: Corpus processing (legal text): tokenization and word counting
1. Implementation of our Word Tokenizer
2. Extract the given corpus.
3. Concatenate text files to form a corpus.
4. Analyze statistics about tokens in the corpus.
* Note: Unlike source codes which are stored at `src` directory, unit tests are stored under `tests` directory. 
---

### Execution of Codes
#### Analytical Results
Please run all cells in the Jupyter Notebook named `Part 1 - Corpus Processing.ipynb` sequentially.
#### Word Tokenizer
* Please run the test file named `tests/test_word_tokenizer.py`. 
* Please also run another test file named `tests/test_count_occurences.py`. 
#### Experimental Lemmantizer
Please run the test file named `tests/test_lemmatizer.py`. 

---

### Part 2: Evaluation of Pre-trained Sentence Embedding Models

This directory also contains the code and data used for evaluating sentence embedding models for semantic textual similarity tasks.

---

#### Prerequisite Files

- **Dataset**: The Semeval 2016 Task-1 dataset (STS data) is provided in the directory `STS_Data/`.
- **Output Directory**: Results of the evaluation will be stored in `Part 2 - Output/`.
- **Requirements**: Dependencies are listed in `requirements.txt`. Install them by running:
  ```bash
  pip install -r requirements.txt
#### Instructions

#### 1. Dataset Preparation
The dataset is preloaded in the folder `STS_Data/` and contains multiple `.txt` files:
- `STS2016.input.answer-answer.txt`
- `STS2016.input.headlines.txt`
- `STS2016.input.plagiarism.txt`
- `STS2016.input.postediting.txt`
- `STS2016.input.question-question.txt`

Concatenate these files to form a single test dataset as required by your experiment.

---

#### 2. Execution of Code
##### Running the Evaluation:
1. Open the Jupyter Notebook `Part 2 - Sentence Similarity.ipynb`.
2. Run all cells sequentially to evaluate the pre-trained sentence embedding models.

The notebook includes:
- Loading the dataset.
- Applying pre-trained models such as SBERT and other LLM-based embeddings.
- Evaluating similarity and computing the Pearson correlation with ground truth scores.

##### Generating Results:
- Outputs of the experiments will be saved in the directory `Part 2 - Output/`.

---

#### 3. Models Used
Pre-trained sentence embedding models evaluated include:
- **SBERT (Sentence-BERT)**: A model designed for semantic textual similarity tasks.
- **Additional models** (e.g., models based on recent generative LLMs).