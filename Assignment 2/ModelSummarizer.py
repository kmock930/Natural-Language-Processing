import numpy as np
from sklearn.metrics import accuracy_score, f1_score
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
    # Author: Your Name
    for model in models:
        model["accuracy"] = accuracy_score(model["trueArray"], model["predArray"])
        model["macroF1"] = f1_score(model["trueArray"], model["predArray"], average="macro")
        model["microF1"] = f1_score(model["trueArray"], model["predArray"], average="micro")
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
    bestModel = max(models, key=lambda x: (x["accuracy"], x["macroF1"], x["microF1"]))
    return bestModel

def plotComparison(models: list[dict]):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    metrics = ["accuracy", "macroF1", "microF1"]
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
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(ROOT, "models_comparison.png"))

if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.abspath(__file__))
    models = [
        {
            "modelName": "Baseline - Logistic Regression",
            "predArray": np.load(os.path.join(ROOT, "predictions","BASELINE_y_dev_pred.npy")),
            "trueArray": np.load(os.path.join(ROOT, "predictions", "BASELINE_y_dev.npy")),
        },
        {
            "modelName": "Deep Learning - DistilBERT",
            "predArray": np.load(os.path.join(ROOT, "predictions", "DISTILBERT_pred_dev.npy")),
            "trueArray": np.load(os.path.join(ROOT, "predictions", "DISTILBERT_y_dev.npy")),
        },
    ]
    models = evaluate(models)
    bestModel = summarize(models)
    print("Best Model is: ", bestModel["modelName"])

    # process model name
    match (bestModel["modelName"]) :
        case "Baseline - Logistic Regression":
            best_model_name = "baseline"
        case "Deep Learning - DistilBERT":
            best_model_name = "distilBERT"
        case _:
            best_model_name = "baseline"
    
    # copy the best model's results to Result.jsonl as an output
    source_file = os.path.join(ROOT, "content", f"Result_{best_model_name}.jsonl")
    destination_file = os.path.join(ROOT, "content", "Results.jsonl")

    shutil.copyfile(source_file, destination_file)
    print(f"Copied {source_file} to {destination_file}")

    # Plot Comparison
    plotComparison(models)