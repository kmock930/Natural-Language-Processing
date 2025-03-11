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
---

## Baseline Model - Logistic Regression
* Please run the Jupyter Notebook named [`part_1.ipynb`](./part_1.ipynb) sequentially to check out the training and prediction processes, as well as evaluation results. 
* A model is dumped into the [`models` direcotory](./models).
* Train set, Dev set, and test set data are dumped into `.npy` files in `data` directory. Please request from owners and unzip the [`data.7z`](https://uottawa-my.sharepoint.com/personal/kmock073_uottawa_ca/Documents/CSI5386%20NLP/data.7z?csf=1&web=1&e=m4fJkY).
* Results of this model is saved into the [`content` directory](./content/).

## Deep Learning based Transformer Model - Fine Tuning DistilBERT
* This part consists of 2 notebooks. To execute the fine tuning process, if you have sufficient computational resources, you may run this notebook sequentially: [`part_2_deep_learning_training.ipynb`](./part_2_deep_learning_training.ipynb).
* The computational resources required is outlined in the notebook. 
* The pre-trained model is loaded and dumped into the [`models` direcotory](./models).
* The fine-tuned model is also dumped into the same [`models directory`](./models/).
* It uses a tokenizer which is different from the baseline model - `TFDistilBertForSequenceClassification`.
* It uses the entire training set and the entire dev set as the validation set during the training process.
* The model is further trained with the tokenizer's `max_depth=32`, batch size = 4, Adam's optimizer, in 5 epochs, with Tensorflow. 