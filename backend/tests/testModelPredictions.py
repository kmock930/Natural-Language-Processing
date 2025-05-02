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
from models import getModels, predict
import numpy as np

class TestModelPredictions(unittest.TestCase):
    def test_predict_distilbert(self):
        tokenizer_name = "Model deep learning distilbert finetuned encoder" 
        encoder_name = "Fine tuned distilbert fold 2"
        model_name = "Custom classifier.keras"
        models = getModels(nameOnly=False, isLocal=True)
        input_data = {
            'title': "", 
            'content': "This is a test input.", 
            'hashtags': ""
        }
        tokenizer = models[tokenizer_name]
        encoder = models[encoder_name]
        classifier = models[model_name]
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
        models = getModels(nameOnly=False, isLocal=True)
        input_data = {
            'title': "I want to die!", 
            'content': "I am going through a terrible week. I would rather kill myself!", 
            'hashtags': "#die #life #kill"
        }
        tokenizer = models[tokenizer_name]
        encoder = models[encoder_name]
        classifier = models[model_name]
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

if __name__ == '__main__':
    unittest.main()