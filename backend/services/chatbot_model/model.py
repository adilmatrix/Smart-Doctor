import os
import pickle
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from services.chatbot_model.preprocessor import clean_symptoms
from services.chatbot_model.tokenizer import symptom_tokenizer

class ChatbotModel:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None
        self.classes = None
        self.load_model()

    def load_model(self) -> None:
        """Load the trained model and vectorizer."""
        try:
            vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
            model_path = os.path.join(self.model_dir, "model.pkl")
            
            if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
                raise FileNotFoundError("Model files not found. Please train the model first.")
            
            # Create a new vectorizer with the same parameters
            self.vectorizer = TfidfVectorizer(
                tokenizer=symptom_tokenizer,
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            # Load the vocabulary from the saved vectorizer
            with open(vectorizer_path, "rb") as f:
                saved_vectorizer = pickle.load(f)
                self.vectorizer.vocabulary_ = saved_vectorizer.vocabulary_
                self.vectorizer.idf_ = saved_vectorizer.idf_
            
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
                self.classes = self.model.classes_
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def predict(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        Make predictions for a given query.
        
        Args:
            query: Comma-separated list of symptoms
            top_k: Number of top predictions to return
            
        Returns:
            List of dictionaries containing responses and their probabilities
        """
        try:
            if not self.model or not self.vectorizer:
                raise ValueError("Model not loaded")
            
            # Clean and preprocess the query
            cleaned_query = clean_symptoms(query)
            
            # Transform the query
            query_vec = self.vectorizer.transform([cleaned_query])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(query_vec)[0]
            
            # Get top k predictions
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            predictions = []
            for idx in top_indices:
                predictions.append({
                    "response": self.classes[idx],
                    "confidence": float(probabilities[idx])
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return []

    def get_available_responses(self) -> List[str]:
        """Return list of all possible responses the model can predict."""
        if not self.classes:
            raise ValueError("Model not loaded")
        return list(self.classes) 