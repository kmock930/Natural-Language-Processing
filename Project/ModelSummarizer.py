import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, auc
import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd

def evaluate(models: list[dict]):
    """
    Evaluate each model in the list of models and add the evaluation metrics to the model dictionary.

    Args:
        models (list[dict]): A list of dictionaries where each dictionary contains details of a model.
        [
            {
                "modelName": "model1",
                "predArray": [1, 2, 3, 4, 5],
                "trueArray": [1, 2, 3, 4, 5],
            }
        ]

    Returns:
        list[dict]: The list of models with added evaluation metrics.
    
    Author: Kelvin Mock
    """
    for model in models:
        model["accuracy"] = accuracy_score(model["trueArray"], model["predArray"])
        model["macroF1"] = f1_score(model["trueArray"], model["predArray"], average="macro")
        model["microF1"] = f1_score(model["trueArray"], model["predArray"], average="micro")
        model["recall"] = recall_score(model["trueArray"], model["predArray"], average="macro")
        model["precision"] = precision_score(model["trueArray"], model["predArray"], average="macro")
        try:
            model["auc"] = auc(model["trueArray"], model["predArray"])
        except Exception as e:
            model["auc"] = 0.0
    return models

def summarize(models: list[dict]):
    """
    Summarize the best model from a list of models.

    Parameters:
    models (list[dict]): A list of dictionaries where each dictionary contains details of a model.
    [
        {
            "modelName": "model1",
            "predArray": [1, 2, 3, 4, 5],
            "trueArray": [1, 2, 3, 4, 5],
            "accuracy": 0.5,
            "macroF1": 0.5,
            "microF1": 0.5
        }
    ]

    Returns:
        dict: The best model from the list of models.
    
    Author: Kelvin Mock
    """
    bestModel = max(models, key=lambda x: (x["accuracy"], x["macroF1"], x["microF1"], x["recall"], x["precision"], x["auc"]))
    return bestModel

def plotComparison(models: list[dict]):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    metrics = ["accuracy", "macroF1", "microF1", "recall", "precision", "auc"]
    data = {metric: [model[metric] for model in models] for metric in metrics}
    data["modelName"] = [model["modelName"] for model in models]
    
    df = pd.DataFrame(data)
    df.set_index("modelName", inplace=True)
    
    plt.figure(figsize=(10, 6))
    df.plot(kind="bar")
    plt.title("Model Comparison")
    plt.ylabel("Scores")
    plt.xlabel("Models")
    plt.xticks(rotation=0)
    plt.gca().set_xticklabels([label.get_text().replace(" ", "\n") for label in plt.gca().get_xticklabels()])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "models_comparison.png"))
    plt.show()

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.abspath(__file__))
    TRAINING_PATH = os.path.join(ROOT, "NLP Training")
    models = [
        {
            "modelName": "Baseline: Logistic Regression",
            "predArray": np.load(os.path.join(TRAINING_PATH, "Results", "baseline_y_pred.npy")),
            "trueArray": np.load(os.path.join(TRAINING_PATH, "Results", "baseline_y_true.npy")),
        },
        {
            "modelName": "Deep Learning: DistilBERT",
            "predArray": np.load(os.path.join(TRAINING_PATH, "Results", "DistilBERT_y_pred_val.npy")),
            "trueArray": np.load(os.path.join(TRAINING_PATH, "resampled data", "y_val_resampled.npy")),
        },
        {
            "modelName": "Deep Learning: LLM",
            "predArray": pd.read_csv(os.path.join(TRAINING_PATH, "model_3_deepseek", "deepseek_model", "test_predictions.csv"))["predicted_label"].to_numpy(),
            "trueArray": pd.read_csv(os.path.join(TRAINING_PATH, "model_3_deepseek", "deepseek_model", "test_predictions.csv"))["label"].to_numpy()
        },
        {
            "modelName": "Hybrid Model",
            "predArray": np.load(os.path.join(TRAINING_PATH, 'Results', 'model_4_hybrid_distilBERT_predictions.npy')),
            "trueArray": np.load(os.path.join(TRAINING_PATH, "Results", "model_4_hybrid_distilBERT_true_labels.npy"))
        },
    ]
    models = evaluate(models)
    bestModel = summarize(models)
    print("Best Model is: ", bestModel["modelName"])

    # process model name
    match (bestModel["modelName"]) :
        case "Baseline - Logistic Regression":
            best_model_name = "baseline_logistic_regression"
        case "Deep Learning - DistilBERT":
            best_model_name = "deep_learning_DistilBERT"
        case "Deep Learning - LLM":
            best_model_name = "llm_deepseek"
        case "Hybrid Model":
            best_model_name = "hybrid_model"
        case _:
            best_model_name = "baseline_logistic_regression"
    
    # copy the best model's results to Result.jsonl as an output
    source_file = os.path.join(ROOT, "NLP Training", "Results", f"Result_{best_model_name}.jsonl")
    destination_file = os.path.join(ROOT, "NLP Training", "Results", "BEST_Results.jsonl")

    shutil.copyfile(source_file, destination_file)
    print(f"Copied {source_file} to {destination_file}")

    # Plot Comparison
    plotComparison(models)