import os
PROJECT_PATH = os.path.join(os.path.dirname(__file__), "..", "Project")
TRAINING_PATH = os.path.join(PROJECT_PATH, "NLP Training")
MODELS_PATH = os.path.join(TRAINING_PATH, "models")
import sys
sys.path.append(TRAINING_PATH)

from model_4_hybrid import CustomVotingClassifier

# For Loading Models
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel, AutoTokenizer, AutoModelForCausalLM
import joblib
import numpy as np

# ignore warnings
import warnings
import torch
warnings.filterwarnings("ignore")

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
        case 'ensemble':
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
            proba = model.predict_proba({
                'X1': encoded_inputs,
                'X2': encoded_inputs
            })
            labels = (proba[:, 1] >= 0.5).astype(int)
            if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
                prediction = labels[0]
            if isinstance(prediction, (np.float32, np.float64, float)):
                prediction = float(prediction)
            print(f"[DEBUG] Final Prediction from Ensemble: {prediction}")
        case 'deepseek':
            if isinstance(inputData, dict):
                for key, value in inputData.items():
                    system_prompt = f"You are an AI mental health assistant. Classify the following text from a social media post's {key} and return only 'suicidal' or 'non-suicidal':\n"
                    encoded = tokenizer(
                        system_prompt + value,
                        return_tensors="pt"
                    )
            else:
                raise ValueError("Input data must be a dictionary with keys: title, content, hashtags.")
            # Generate predictions using the model
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoded['input_ids'].to(model.device),
                    attention_mask=encoded['attention_mask'].to(model.device),
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            # Decode the generated output
            decoded_output = tokenizer.decode(outputs[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            # Convert the decoded output to a float prediction (0 or 1)
            pred_label = None
            if "suicidal" in decoded_output:
                pred_label = 1
            elif "non-suicidal" in decoded_output:
                pred_label = 0
            if pred_label is not None:
                prediction = pred_label
            else:
                # Confidence
                prediction = 0.9 if "definitely" in decoded_output or "clearly" in decoded_output else 0.7

    return prediction

def getDeepSeekModel():
    MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
    OUTPUT_DIR = os.path.join(MODELS_PATH, "deepseek_model")
    if not os.path.exists(os.path.join(OUTPUT_DIR, "pretrained_autotokenizer")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Set special tokens
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "pretrained_autotokenizer"))
        print(f"✅ Tokenizer saved to {os.path.join(OUTPUT_DIR, 'pretrained_autotokenizer')}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(OUTPUT_DIR, "pretrained_autotokenizer"))
        print(f"✅ Tokenizer loaded from {os.path.join(OUTPUT_DIR, 'pretrained_autotokenizer')}")
    if not os.path.exists(os.path.join(OUTPUT_DIR, "pretrained_llm_model")):
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map= "auto" if torch.cuda.is_available() else None,
            load_in_8bit=True # Enable 8-bit quantization for memory efficiency
        )
        model.save_pretrained(os.path.join(OUTPUT_DIR, "pretrained_llm_model"))
        print(f"✅ Model saved to {MODEL_NAME}")
    else:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(OUTPUT_DIR, "pretrained_llm_model"))
        print(f"✅ Model loaded from {os.path.join(OUTPUT_DIR, 'pretrained_llm_model')}")
    return model, tokenizer