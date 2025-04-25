# Natural Language Processing Work
## [Assignment 1](./Assignment%201/README.md) - Corpus analysis and sentence embeddings

![Assignment 1 Preview](asm1-preview.png)

## [Assignment 2](./Assignment%202/README.md) - Machine-Generated Text Detection

![Assignment 2 Preview](./Assignment%202/models_comparison.png)

## [Seminar Research](./Seminar%20Paper/Paper%20Presentation%20-%20Group%202.pdf) - Depression Detection
Given the rising popularity of social media, there is a risk of negative impacts such as cyberbullying, causing mental health distress to some users. As a result, we dived into an exploration of depression detection with the **DORIS framework** proposed by Lan X., Cheng Y., Sheng L., Gao C., and Li Y. It also forms a basis for our project which aims to perform a NLP-based model targetting suicide detection.

## [Project](./Project/README.md)
### Summary of Our Work
* [Project's Proposal](./Project/CSI5386_Natural_Language_Processing_Project_Proposal.pdf)
* [Presenting from the NLP's Perspective](./Project/Project%20Presentation%20-%20NLP%20Aspects.pdf)
Our project analyzes suicidal intentions from popular social media platforms, and trains the best model for suicidal detection. Here are the models that we've used. 

![Summary of Models](./Project/models_comparison.png)

### Baseline Model
![Project - Baseline Model](./Project/NLP%20Training/Results/baseline_auc_curve.png)

### Fine-Tuning a Deep Learning based Transformer - DistilBERT
![Project - Deep Learning based Fine-Tuning DistilBERT Model's Results](./Project/NLP%20Training/Results/Fine-tuned%20DistilBERT%20accuracy_fold_2.png)
### Added Custom Layers on top of Fine-Tuned DistilBERT
![Project - Deep Learning based Custom Layers](./Project/NLP%20Training/Results/Custom%20Layers_accuracy_fold_5.png)

![Project - Deep Learning based model resulting AUC](./Project/NLP%20Training/Results/model_2_deep_learning_auc_curve.png)

### LLM-based Model
![Project - LLM-Based Model - ROC-AUC](./Project/NLP%20Training/Results/ROC-deepseek.png)

![Project - LLM-based Model - Confusion Matrix](./Project/NLP%20Training/Results/deepseek_confusion_matrix.png)

## Execution Guide
* [**TMUX**](tmux.md) for idling long executions
