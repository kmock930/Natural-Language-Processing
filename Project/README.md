# Suicide Detection - Project Guide
## [`data` directory](./data/README.md)
It includes 4 datasets, in separate subdirectories. Each of which contains various code files for processes from loading the raw datasets to exploring, analyzing, and further data preprocessing/cleansing. 
Check the [README.md](./data/README.md) file inside the directory for further information about the datasets.

## [N-Gram Analysis](./N-Gram%20Analysis/)
It includes all the exported results from all models. It adds fields related to sentiment analysis in .json files, including polarity, subjectivity ratio, postivity / negativity / neutrality ratios. 

## [NLP Training](./NLP%20Training/)
It is the main directory containing training and evaluation codes of different models. 
### The Baseline Model
Please run this file: `model_1_baseline.ipynb`.
### Deep Learning based Model
To run the experimentaion of only the custom layers, please run the following Jupyter Notebooks in sequence:
1. `model_2_deep_learning_customized_training.ipynb`
2. `model_2_deep_learning_customized_evaluation.ipynb`

To run the full pipeline, please run these Python files in sequence:
1. `fine-tuning-distilBERT.py`
2. `training-added-layers-distilBERT.py`

Alternatively, you may directly run this file for the entire pipeline: `model_2_deep_learning_pipeline.py`.
### LLM-based Model
It relies on the recent generative LLM - deepseek. Please run these files in sequence:
1. `DeepSeek Data Processing.ipynb`
2. `DeepSeek Model Implementation.ipynb` (Note: You may need an API key to access the LLM with prompts.)
3. `parse-test-results.py`: this file exports prediction results into a readable and comparable .json format from an exported .csv file.

Sub-directories like `deepseek_model` contains the trained model and its corresponding information from an evaluation; and `processed_data_deepseek` includes pre-processed data from raw datasets which are compatible with the LLM. 

## [Reference Paper](./Reference%20Paper/)
This directory includes all the academic papers as an inspirational source of our project.

## [tests](./tests/)
This directory contains unit tests, particularly the implementation of data processing codes. 

## [`sentiment-analysis.ipynb`](./sentiment-analysis.ipynb)
This notebook contains an analysis of N-Grams from the test set as well as sentiment analysis using TextBlob and VADER. Results are exported to the `N-Gram Analysis` directory. 

## Please run `pip install -r requirements.txt` by changing directory into `Project` directory in order to obtain all the dependencies prior to executing codes. 