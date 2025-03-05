# Assignment 2: Machine-Generated Text Detection
All codes are in the directory named `Assignment 2`.
---

## Prerequisite Files:
* Dataset - part of the [SemEval 2024 Task 8](https://github.com/mbzuai-nlp/SemEval2024-task8), which could also be accessed [here](https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc)
* `subtaskA_dev_monolingual.jsonl`
* `subtaskA_monolingual.jsonl`
* `subtaskA_train_monolingual.jsonl`
* To install all dependencies, run this command with your terminal: `pip install -r requirements.txt`
---

## Baseline Model - Logistic Regression
* Please run the Jupyter Notebook named [`part_1.ipynb`](./part_1.ipynb) sequentially to check out the training and prediction processes, as well as evaluation results. 
* A model is dumped into the [`models` direcotory](./models).
* Train set, Dev set, and test set data are dumped into `.npy` files in `data` directory. Please request from owners and unzip the [`data.7z`](https://uottawa-my.sharepoint.com/personal/kmock073_uottawa_ca/Documents/CSI5386%20NLP/data.7z?csf=1&web=1&e=m4fJkY).
* Results of this model is saved into the [`content` directory](./content/).