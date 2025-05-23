import json
from typing import Dict, List, Tuple
from collections import Counter
import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)

from config import CHATBOT_DATA_PATH

def analyze_dataset() -> Dict:
    """Analyze the dataset and return statistics."""
    try:
        with open(CHATBOT_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        # Basic statistics
        total_samples = len(data)
        
        # Analyze queries
        queries = [item['query'] for item in data]
        responses = [item['response'] for item in data]
        
        # Check for empty or invalid entries
        empty_queries = sum(1 for q in queries if not q.strip())
        empty_responses = sum(1 for r in responses if not r.strip())
        
        # Analyze response distribution
        response_counter = Counter(responses)
        unique_responses = len(response_counter)
        
        # Check for special characters and formatting
        special_chars = set()
        for query in queries:
            special_chars.update(char for char in query if not char.isalnum() and not char.isspace())
        
        # Print first few examples
        print("\nFirst 5 examples:")
        for i, item in enumerate(data[:5]):
            print(f"\nExample {i+1}:")
            print(f"Query: {item['query']}")
            print(f"Response: {item['response']}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Unique responses: {unique_responses}")
        print(f"Empty queries: {empty_queries}")
        print(f"Empty responses: {empty_responses}")
        print(f"Special characters found: {sorted(list(special_chars))}")
        
        # Print most common responses
        print("\nTop 5 most common responses:")
        for response, count in response_counter.most_common(5):
            print(f"{response}: {count} occurrences")
        
        return {
            "total_samples": total_samples,
            "unique_responses": unique_responses,
            "empty_queries": empty_queries,
            "empty_responses": empty_responses,
            "special_chars": sorted(list(special_chars))
        }
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return {}

if __name__ == "__main__":
    analyze_dataset() 