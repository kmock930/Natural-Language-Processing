import os
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..", "Project")
TRAINING_PATH = os.path.join(PROJECT_PATH, "NLP Training")
MODELS_PATH = os.path.join(TRAINING_PATH, "models")
import sys
sys.path.append(TRAINING_PATH)

from model_4_hybrid import CustomVotingClassifier

# For Loading Models
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel
import joblib
import numpy as np

def getModels(nameOnly=False, isLocal=False):
    models = {}
    if isLocal:
        # From local dir
        for modelName in os.listdir(MODELS_PATH):
            modelPath = os.path.join(MODELS_PATH, modelName)
            if modelName.lower().startswith('best'):
                if modelName.lower().endswith('.h5'):
                    modelName = " ".join(modelName.lower().replace('best_', '').replace('.h5', '').split('_')).capitalize()
                    if 'baseline' in modelName.lower():
                        models[modelName] = joblib.load(modelPath) if not nameOnly else modelName
                elif modelName.lower().endswith('.keras'):
                    modelName = " ".join(modelName.lower().replace('best_', '').replace('.h5', '').split('_')).capitalize()
                    models[modelName] = tf.keras.models.load_model(modelPath, compile=False) if not nameOnly else modelName
            if os.path.isdir(modelPath):
                if 'encoder' in modelName.lower():
                    modelName = " ".join(modelName[:-1].split('_')).replace('fold', '').rstrip().capitalize()
                    models[modelName] = DistilBertTokenizer.from_pretrained(modelPath) if not nameOnly else modelName
                elif 'best' in modelName.lower():
                    modelName = " ".join(modelName.lower().replace('best_', '').split('_')).strip().capitalize()
                    models[modelName] = TFDistilBertModel.from_pretrained(modelPath) if not nameOnly else modelName
            if modelName.lower().endswith('.pkl') and 'ensemble' in modelName.lower():
                modelName = " ".join(modelName.lower().replace('.pkl', '').split('_')).strip().capitalize()
                models[modelName] = joblib.load(modelPath) if not nameOnly else modelName
    return models

def extract_cls_from_embeddings(encoder, encoded_inputs):
    output = encoder(
        input_ids=encoded_inputs['input_ids'], 
        attention_mask=encoded_inputs['attention_mask'],
        # Ensure the model is in inference mode
        **({'trainable': False} if 'trainable' in encoder.__call__.__code__.co_varnames else {})
    )
    # Extract the [CLS] embeddings
    cls_embeddings = output.last_hidden_state[:, 0, :].numpy()
    return cls_embeddings

def predict(modelName, inputData, tokenizer, encoder, model):
    prediction = 0 # initialize:non-suicidal

    match modelName:
        case 'baseline' | 'distilBERT':
            if isinstance(inputData, dict):
                encoded_inputs = []
                for key, value in inputData.items():
                    encoded = tokenizer(
                        value, 
                        truncation=True, 
                        padding="max_length", 
                        max_length=32, 
                        return_tensors="tf"
                    )
                    if hasattr(encoder, 'distilbert'):
                        encoder = encoder.distilbert
                    cls_embeddings = extract_cls_from_embeddings(encoder, encoded)
                    encoded_inputs.append(cls_embeddings)
                encoded_inputs = np.concatenate(encoded_inputs, axis=1)
            else:
                raise ValueError("Input data must be a dictionary with keys: title, content, hashtags.")
            prediction = model.predict(encoded_inputs)
            while isinstance(prediction, (list, np.ndarray)) and len(prediction) > 0:
                prediction = prediction[0]
            if isinstance(prediction, (np.float32, np.float64, float)):
                prediction = float(prediction)
    
    return prediction