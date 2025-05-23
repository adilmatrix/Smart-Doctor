import os
import sys
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)

from services.chatbot_model.model import ChatbotModel

def test_query(query: str, top_k: int = 5):
    """Test the model with a given query."""
    try:
        # Initialize the model
        model = ChatbotModel()
        
        # Make prediction
        predictions = model.predict(query, top_k=top_k)
        
        # Print results
        print("\nQuery:", query)
        print("\nTop", top_k, "responses:")
        print("-" * 50)
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['response']}")
            print(f"   Confidence: {pred['confidence']:.2%}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing model: {str(e)}")

if __name__ == "__main__":
    # Test query
    query = "white discharge from eye ,diminished vision ,pain in eye ,eye redness ,fever"
    test_query(query) 