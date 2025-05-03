from fastapi import FastAPI

import os
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..", "Project")
TRAINING_PATH = os.path.join(PROJECT_PATH, "NLP Training")
MODELS_PATH = os.path.join(TRAINING_PATH, "models")
import sys
sys.path.append(TRAINING_PATH)

from model_4_hybrid import CustomVotingClassifier
from models import getModels, predict as customPredict, getDeepSeekModel
import numpy as np

# Logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
app = FastAPI()

def is_local_request(request: Request):
    return request.client.host in ["127.0.0.1", "localhost"]

@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/getModelNames")
async def get_model_names(request: Request):
    models = getModels(nameOnly=True, isLocal=is_local_request(request))
    return list(models.keys())

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: Request):
    startTime = time.time()

    models = getModels(nameOnly=False, isLocal=is_local_request(request))
    if model_name in models or model_name.lower() in ['ensemble', 'deepseek']:
        body = await request.json()
        if 'baseline' in model_name.lower():
            if not all(key in body for key in ['title', 'content', 'hashtags']):
                return JSONResponse(status_code=400, content={"error": "Invalid input format. Expected keys: title, content, hashtags."})
            title = body['title']
            content = body['content']
            hashtags = body['hashtags']

            tokenizer = models["Model deep learning distilbert finetuned encoder"]
            distilBERT_encoder = models["Fine tuned distilbert fold 2"]
            baseline_model = models["Baseline model logisticregression"]

            prediction = customPredict(
                modelName="baseline",
                inputData={
                    'title': title,
                    'content': content,
                    'hashtags': hashtags
                },
                tokenizer=tokenizer,
                encoder=distilBERT_encoder,
                model=baseline_model
            )
        if 'distilbert' in model_name.lower():
            if not all(key in body for key in ['content']):
                return JSONResponse(status_code=400, content={"error": "Invalid input format. Expected keys: content."})
            title = body['title'] if 'title' in body else ""
            content = body['content']  # mandatory
            hashtags = body['hashtags'] if 'hashtags' in body else ""
            print(f"Content: {content}")
            tokenizer = models["Model deep learning distilbert finetuned encoder"]
            print("Tokenizer is loaded.")
            encoder = models["Fine tuned distilbert fold 2"]
            print("Encoder is loaded.")
            model = models["Custom classifier.keras"]
            print("Model is loaded.")
            # Prediction
            prediction = customPredict(
                modelName="distilBERT",
                inputData={
                    'title': title,
                    'content': content,
                    'hashtags': hashtags
                },
                tokenizer=tokenizer,
                encoder=encoder,
                model=model
            )
        if 'ensemble' in model_name.lower():
            if not all(key in body for key in ['content']):
                return JSONResponse(status_code=400, content={"error": "Invalid input format. Expected keys: content."})
            title = body['title'] if 'title' in body else ""
            content = body['content']  # mandatory
            hashtags = body['hashtags'] if 'hashtags' in body else ""

            tokenizer = models["Model deep learning distilbert finetuned encoder"]
            distilBERT_encoder = models["Fine tuned distilbert fold 2"]
            model_names = ["Ensemble hard model", "Ensemble soft model"]

            predictions = []
            for model_name in model_names:
                ensemble_model = models[model_name]

                prediction = customPredict(
                    modelName=model_name,
                    inputData={
                        'title': title,
                        'content': content,
                        'hashtags': hashtags
                    },
                    tokenizer=tokenizer,
                    encoder=distilBERT_encoder,
                    model=ensemble_model
                )
                predictions.append(prediction)
                print(f"Prediction of {model_name}: {prediction}")

            # Combine predictions from both models via Majority Voting
            final_prediction = (np.sum(predictions) >= 1).astype(int)
            print(f"Final prediction: {final_prediction}")
            prediction = final_prediction
        if 'deepseek' in model_name.lower():
            if not all(key in body for key in ['content']):
                return JSONResponse(status_code=400, content={"error": "Invalid input format. Expected keys: content."})
            title = body['title'] if 'title' in body else ""
            content = body['content']
            hashtags = body['hashtags'] if 'hashtags' in body else ""

            model, tokenizer = getDeepSeekModel()
            encoder = None # No encoder needed for DeepSeek model
            prediction = customPredict(
                modelName="deepseek",
                inputData={
                    'title': title,
                    'content': content,
                    'hashtags': hashtags
                },
                tokenizer=tokenizer,
                encoder=encoder,
                model=model
            )
            print(f"Prediction of DeepSeek model: {prediction}")

        # Convert prediction to a more readable format
        ret_content = {}
        if isinstance(prediction, float):
            prediction = 1 if prediction >= 0.5 else 0
            ret_content = {"Prediction": 'suicidal' if prediction == 1 else 'non-suicidal'}
        elif isinstance(prediction, str):
            ret_content = {"Prediction": prediction}
        else:
            try:
                prediction = int(prediction)
                ret_content = {"Prediction": 'suicidal' if prediction == 1 else 'non-suicidal'}
            except ValueError:
                ret_content = {"error": f"Unexpected prediction format: {type(prediction)}", "Prediction": prediction}

        endTime = time.time()
        print(f"Elapsed runtime: {endTime - startTime} seconds")
        return JSONResponse(
            status_code=200 if 'error' not in ret_content else 400, 
            content=ret_content
        )
    else:
        endTime = time.time()
        print(f"Elapsed runtime: {endTime - startTime} seconds")
        return JSONResponse(status_code=400, content={"error": "Model not found"})



def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()