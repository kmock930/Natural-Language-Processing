# Part 2: Evaluation of Pre-trained Sentence Embedding Models

This directory contains the code and data used for evaluating sentence embedding models for semantic textual similarity tasks.

---

## Prerequisite Files

- **Dataset**: The Semeval 2016 Task-1 dataset (STS data) is provided in the directory `STS_Data/`.
- **Output Directory**: Results of the evaluation will be stored in `Part 2 - Output/`.
- **Requirements**: Dependencies are listed in `requirements.txt`. Install them by running:
  ```bash
  pip install -r requirements.txt
# Instructions

## 1. Dataset Preparation
The dataset is preloaded in the folder `STS_Data/` and contains multiple `.txt` files:
- `STS2016.input.answer-answer.txt`
- `STS2016.input.headlines.txt`
- `STS2016.input.plagiarism.txt`
- `STS2016.input.postediting.txt`
- `STS2016.input.question-question.txt`

Concatenate these files to form a single test dataset as required by your experiment.

---

## 2. Execution of Code
### Running the Evaluation:
1. Open the Jupyter Notebook `Part 2 - Sentence Similarity.ipynb`.
2. Run all cells sequentially to evaluate the pre-trained sentence embedding models.

The notebook includes:
- Loading the dataset.
- Applying pre-trained models such as SBERT and other LLM-based embeddings.
- Evaluating similarity and computing the Pearson correlation with ground truth scores.

### Generating Results:
- Outputs of the experiments will be saved in the directory `Part 2 - Output/`.

---

## 3. Models Used
Pre-trained sentence embedding models evaluated include:
- **SBERT (Sentence-BERT)**: A model designed for semantic textual similarity tasks.
- **Additional models** (e.g., models based on recent generative LLMs).
