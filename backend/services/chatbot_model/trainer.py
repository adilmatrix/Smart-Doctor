import os
import json
import pickle
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from .preprocessor import clean_symptoms
from .tokenizer import symptom_tokenizer

class ModelTrainer:
    def __init__(self, data_path: str = "dataset/chatbot_data.json", model_dir: str = "models"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.vectorizer = None
        self.model = None
        self.classes = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

    def load_data(self) -> Tuple[List[str], List[str]]:
        """Load and preprocess the training data."""
        print("Loading and preprocessing data...")
        
        # Load the JSON file
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract queries and responses
        print("Extracting queries and responses...")
        queries = []
        responses = []
        
        # Process each item with a progress bar
        for item in tqdm(data, desc="Processing data"):
            # Clean and preprocess the symptoms
            symptoms = clean_symptoms(item['query'])
            queries.append(symptoms)
            responses.append(item['response'])
        
        print(f"Loaded {len(queries)} training samples\n")
        return queries, responses

    def train(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train the model and return metrics."""
        try:
            # Load and preprocess data
            queries, responses = self.load_data()
            
            # Split data into train and test sets
            print("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                queries, responses, test_size=test_size, random_state=random_state
            )
            print(f"Training set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}\n")
            
            # Initialize and fit vectorizer
            print("Vectorizing text data...")
            self.vectorizer = TfidfVectorizer(
                tokenizer=symptom_tokenizer,
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            print("Fitting vectorizer...")
            X_train_vec = self.vectorizer.fit_transform(tqdm(X_train, desc="Vectorizing training data"))
            X_test_vec = self.vectorizer.transform(tqdm(X_test, desc="Vectorizing test data"))
            
            # Train model
            print("\nTraining model...")
            start_time = time.time()
            self.model = LogisticRegression(max_iter=1000, n_jobs=-1)
            self.model.fit(X_train_vec, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            print("Making predictions...")
            y_pred = self.model.predict(X_test_vec)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Get top 5 most accurate classes
            class_accuracies = []
            for class_name in self.model.classes_:
                # Get indices where true label matches this class
                class_indices = np.where(y_test == class_name)[0]
                if len(class_indices) > 0:  # Only consider classes with test samples
                    class_pred = y_pred[class_indices]
                    class_true = y_test[class_indices]
                    class_acc = accuracy_score(class_true, class_pred)
                    class_accuracies.append((class_name, class_acc))
            
            top_classes = sorted(class_accuracies, key=lambda x: x[1], reverse=True)[:5]
            
            # Prepare metrics
            metrics = {
                "accuracy": accuracy,
                "training_time": training_time,
                "vocabulary_size": len(self.vectorizer.vocabulary_),
                "num_classes": len(self.model.classes_),
                "top_classes": {name: acc for name, acc in top_classes},
                "classification_report": report
            }
            
            # Save model and vectorizer
            print("\nSaving model and vectorizer...")
            self.save_model()
            
            # Save metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = os.path.join(self.model_dir, f"metrics_{timestamp}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            
            print(f"\nModel saved to {os.path.join(self.model_dir, 'model.pkl')}")
            print(f"Vectorizer saved to {os.path.join(self.model_dir, 'vectorizer.pkl')}")
            print(f"Metrics saved to {metrics_path}")
            
            return metrics
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            raise

    def save_model(self) -> None:
        """Save the trained model and vectorizer."""
        if not self.model or not self.vectorizer:
            raise ValueError("No model or vectorizer to save")
            
        model_path = os.path.join(self.model_dir, "model.pkl")
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
            
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

if __name__ == "__main__":
    trainer = ModelTrainer()
    metrics = trainer.train()
    
    # Print summary
    print("\nTraining Results:")
    print("-" * 50)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")
    print(f"Vocabulary Size: {metrics['vocabulary_size']}")
    print(f"Number of Classes: {metrics['num_classes']}")
    
    print("\nTop 5 Most Accurate Classes:")
    for condition, accuracy in metrics['top_classes'].items():
        print(f"- {condition}: {accuracy:.2%}") 