# Assignment 2: Machine-Generated Text Detection
All codes are in the directory named `Assignment 2`.
---

## Prerequisite Files:
* Dataset - part of the [SemEval 2024 Task 8](https://github.com/mbzuai-nlp/SemEval2024-task8), which could also be accessed [here](https://drive.google.com/drive/folders/1CAbb3DjrOPBNm0ozVBfhvrEh9P9rAppc)
* `subtaskA_dev_monolingual.jsonl`
* `subtaskA_monolingual.jsonl`
* `subtaskA_train_monolingual.jsonl`
* To install all dependencies, run this command with your terminal: `pip install -r requirements.txt`
* Run this alternative command if you are training the deep learning model on a Linux-based system: `requirements_for_deep_learning_linux.txt`.
* Setup [conda](https://stackoverflow.com/questions/44515769/conda-is-not-recognized-as-internal-or-external-command) and [cuda](https://www.tensorflow.org/install/pip) for tensorflow and configure your environment such that codes run on your **GPU** instead. 
* A script for system's [health check](../health_check.py) is available at the root level of the repository. 
---

## Baseline Model - Logistic Regression
* Please run the Jupyter Notebook named [`part_1.ipynb`](./part_1.ipynb) sequentially to check out the training and prediction processes, as well as evaluation results. 
* A model is dumped into the [`models` direcotory](./models).
* Train set, Dev set, and test set data are dumped into `.npy` files in `data` directory. Please request from owners and unzip the [`data.7z`](https://uottawa-my.sharepoint.com/personal/kmock073_uottawa_ca/Documents/CSI5386%20NLP/data.7z?csf=1&web=1&e=m4fJkY).
* Results of this model is saved into the [`content` directory](./content/).

## Deep Learning based Transformer Model - Fine Tuning DistilBERT
* This part consists of 2 notebooks: 
1. To execute the fine tuning process, if you have sufficient computational resources, you may run this notebook sequentially: [`part_2_deep_learning_training.ipynb`](./part_2_deep_learning_training.ipynb).
2. To execute the prediction & evaluation process, you may run this notebook sequentially: [`part_2_deep_learning_evaluation.ipynb`](./part_2_deep_learning_evaluation.ipynb).
* The computational resources required is outlined in the notebook. 
* The pre-trained model is loaded and dumped into the [`models` direcotory](./models).
* The fine-tuned model is also dumped into the same [`models directory`](./models/).
* It uses a tokenizer which is different from the baseline model - `TFDistilBertForSequenceClassification`.
* It uses the entire training set and the entire dev set as the validation set during the training process.
* The model is further trained with the tokenizer's `max_depth=32`, batch size = 4, Adam's optimizer, in 5 epochs, with Tensorflow. 
* Results of this model is saved into the [`content` directory](./content/).

## Zero-Shot Classification with Large Language Model
* This part uses a pre-trained LLM model for zero-shot classification of machine-generated text.
* To execute the zero-shot classification process, you may run this notebook: [`part3-llm.ipynb`](./part3-llm.ipynb).
* It leverages the `facebook/bart-large-mnli` model with zero-shot classification capabilities.
* The model classifies text as either "human-written" or "machine-generated" without requiring any task-specific training.
* Text inputs are truncated to 500 tokens to manage computational requirements.
* Unlike the other models, this approach demonstrates how pre-trained language models can be applied to classification tasks without fine-tuning.
* Dependencies for this implementation are in `requirements_for_llm.txt`. Install with: `pip install -r requirements_for_llm.txt`
* Results of this model are saved into the [`content` directory](./content/).
* The model achieved an accuracy of 50.85% on the test set and 52.06% on the validation set.
* Performance metrics:
  - Test Set: Accuracy: 0.5085, Macro F1: 0.4358, Micro F1: 0.5085
  - Validation Set: Accuracy: 0.5206, Macro F1: 0.4245, Micro F1: 0.5206


## Check the Best Model and its Results
* After executing all the programs covering the training and evaluation processes for all models, run [`ModelSummarizer.py`](./ModelSummarizer.py) to check whichever is the best model. It is determined by the combination of accuracies, macro and micro F1 scores. The results of the best model is copied to [`Results.jsonl`](./content/Results.jsonl).