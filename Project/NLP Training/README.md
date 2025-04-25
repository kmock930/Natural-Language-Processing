# Installation Guidelines
* Use Python 3.10
* Use Cuda 11.5 with Tensorflow 2.12
* Under `Project` directory, run this command on your terminal: `pip install -r requirements.txt` in order to install all necessary dependencies.
# Baseline Model
* Run this notebook: `model_1_baseline.ipynb`.
# Fine Tuning DistilBERT
## Execution Guidelines
### Full Pipeline
* As suggested in `model_2_deep_learning_pipeline.py`, the pipeline for training our 2nd deep-learning-based model is in the sequence of firstly running `fine-tuning-distilBERT.py` and secondly `training-added-layers-distilBERT.py`.
* Run this notebook for evaluating the model: `model_2_deep_learning_OVERALL_evaluation.ipynb`.
### Experimental Codes of the Custom Layers
* Run this notebook to train standalone custom layers: `model_2_deep_learning_customized_training.ipynb`.
* Then, run this notebook to evaluate the standalone layers: `model_2_deep_learning_OVERALL_evaluation.ipynb`.
## Files and Directories
* `resampled data` is a directory containing resmapled data arrays from training, validation and test set. Those text embeddings are also vectorized using the fine-tuned DistilBERT model. 
* If you intend to use embeddings produced from the **raw** pretrained DistilBERT model, please check this directory: `data/Numpy Data/Text`. It is produced from `data/data_splitting_DistilBERT.py` and used in `data/data_processing.py` the preprocessing logic and the baseline model. 
## Model Architecture
### DistilBERT Base Model
| Layer (type)                | Output Shape           | Param #   |
|-----------------------------|------------------------|-----------|
| distilbert (TFDistilBertMainLayer) | multiple         | 66,362,880 |

* **Total params:** 66,362,880 (253.15 MB)  
* **Trainable params:** 66,362,880 (253.15 MB)  
* **Non-trainable params:** 0 (0.00 Byte)  

### Custom Layers
| Layer (type)                  | Output Shape           | Param #   |
|-------------------------------|------------------------|-----------|
| input_layer_361 (InputLayer)  | (None, 2304)           | 0         |
| reshape_361 (Reshape)         | (None, 3, 768)         | 0         |
| bidirectional_361             | (None, 128)            | 426,496   |
| (Bidirectional)               |                        |           |
| dropout_361 (Dropout)         | (None, 128)            | 0         |
| dense_722 (Dense)             | (None, 32)             | 4,128     |
| dense_723 (Dense)             | (None, 1)              | 33        |

* **Total params**: 430,657 (1.64 MB)
* **Trainable params**: 430,657 (1.64 MB)
* **Non-trainable params**: 0 (0.00 B)
# LLM-based Model
* Run these notebooks in sequence: 
1. `DeepSeek Data Processing.ipynb`
2. `DeepSeek Model Implementation.ipynbDeepSeek Model Implementation.ipynb`