import unittest
import sys
import os
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROJECT_PATH = os.path.join(BACKEND_PATH, "..", "Project")
TRAINING_PATH = os.path.join(PROJECT_PATH, "NLP Training")
MODELS_PATH = os.path.join(TRAINING_PATH, "models")
import sys
sys.path.append(TRAINING_PATH)
sys.path.append(BACKEND_PATH)
from model_4_hybrid import CustomVotingClassifier
from models import getModels, predict, getDeepSeekModel
from transformers import LlamaForCausalLM, LlamaTokenizerFast
import numpy as np

class TestModelPredictions(unittest.TestCase):
    def setUp(self):
        self.models = getModels(nameOnly=False, isLocal=True)

    def test_predict_distilbert(self):
        tokenizer_name = "Model deep learning distilbert finetuned encoder" 
        encoder_name = "Fine tuned distilbert fold 2"
        model_name = "Custom classifier.keras"
        input_data = {
            'title': "", 
            'content': "This is a test input.", 
            'hashtags': ""
        }
        tokenizer = self.models[tokenizer_name]
        encoder = self.models[encoder_name]
        classifier = self.models[model_name]
        prediction = predict(
            'distilBERT', # alias
            input_data, 
            tokenizer, 
            encoder,
            classifier
        )
        print(f"Prediction of DistilBERT model: {prediction}")
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)

    def test_predict_baseline(self):
        tokenizer_name = "Model deep learning distilbert finetuned encoder"
        encoder_name = "Fine tuned distilbert fold 2"
        model_name = "Baseline model logisticregression"
        input_data = {
            'title': "I want to die!", 
            'content': "I am going through a terrible week. I would rather kill myself!", 
            'hashtags': "#die #life #kill"
        }
        tokenizer = self.models[tokenizer_name]
        encoder = self.models[encoder_name]
        classifier = self.models[model_name]
        prediction = predict(
            'baseline', # alias
            input_data, 
            tokenizer, 
            encoder,
            classifier
        )
        print(f"Prediction of Baseline model: {prediction}")
        self.assertTrue(isinstance(prediction, (np.int64, int)))
        self.assertIn(prediction, [0, 1])

    def test_predict_ensemble(self):
        model_names = ["Ensemble hard model", "Ensemble soft model"]
        predictions = []
        for model_name in model_names:
            input_data = {
                'title': "I want to die!",
                'content': "I am going through a terrible week. I would rather kill myself!",
                'hashtags': "#die #life #kill"
            }
            tokenizer = self.models["Model deep learning distilbert finetuned encoder"]
            encoder = self.models["Fine tuned distilbert fold 2"]
            classifier = self.models[model_name]
            prediction = predict(
                'ensemble', # alias
                input_data, 
                tokenizer, 
                encoder,
                classifier
            )
            print(f"Prediction of {model_name}: {prediction}")
            self.assertTrue(isinstance(prediction, (np.int64, np.int32, int)))
            self.assertIn(prediction, [0, 1])
            predictions.append(prediction)
        # Majority Voting
        final_prediction = (np.sum(predictions) >= 1).astype(int)
        print(f"Final prediction from ensemble: {final_prediction}")
        self.assertTrue(isinstance(final_prediction, (np.int64, np.int32, int)))
        self.assertIn(final_prediction, [0, 1])

    def test_getDeepSeekModel(self):
        model, tokenizer = getDeepSeekModel()
        self.assertIsNotNone(model, "DeepSeek model is None")
        self.assertIsNotNone(tokenizer, "Tokenizer is None")
        self.assertTrue(isinstance(model, LlamaForCausalLM))
        self.assertTrue(isinstance(tokenizer, LlamaTokenizerFast))
    
    def test_predict_deepseek(self):
        model, tokenizer = getDeepSeekModel()
        input_data = {
            'title': "I want to die!",
            'content': "I am going through a terrible week. I would rather kill myself!",
            'hashtags': "#die #life #kill"
        }
        encoder = None  # No encoder needed for DeepSeek model
        prediction = predict(
            'deepseek', # alias
            input_data, 
            tokenizer, 
            encoder,
            model
        )
        print(f"Prediction of DeepSeek model: {prediction}")
        self.assertTrue(isinstance(prediction, int))
        self.assertIn(prediction, [0, 1])


if __name__ == '__main__':
    unittest.main()