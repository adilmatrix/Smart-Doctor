from typing import List, Dict
import json
import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)

from config import CHATBOT_DATA_PATH

def clean_symptoms(symptoms: str) -> str:
    """
    Clean a comma-separated list of symptoms.
    - Strips whitespace
    - Converts to lowercase
    - Removes duplicates
    """
    # Split by comma and clean each symptom
    symptom_list = [s.strip().lower() for s in symptoms.split(',')]
    # Remove duplicates while preserving order
    unique_symptoms = list(dict.fromkeys(symptom_list))
    # Join back with commas
    return ','.join(unique_symptoms)

def preprocess_data(data: List[Dict]) -> List[Dict]:
    """
    Preprocess the entire dataset.
    Returns a list of cleaned query-response pairs.
    """
    cleaned_data = []
    for item in data:
        cleaned_query = clean_symptoms(item['query'])
        cleaned_response = item['response'].strip().lower()
        cleaned_data.append({
            'query': cleaned_query,
            'response': cleaned_response
        })
    return cleaned_data

def load_and_preprocess_data() -> List[Dict]:
    """
    Load the dataset and apply preprocessing.
    Returns the cleaned dataset.
    """
    try:
        with open(CHATBOT_DATA_PATH, 'r') as f:
            data = json.load(f)
        return preprocess_data(data)
    except Exception as e:
        print(f"Error loading and preprocessing data: {str(e)}")
        return [] 