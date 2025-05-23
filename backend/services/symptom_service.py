import pandas as pd
import os

def get_all_symptoms():
    """Extract all unique symptoms from the dataset."""
    try:
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        
        # Load the dataset
        df = pd.read_csv(os.path.join(project_root, 'dataset/diseases_symptoms.csv'))
        
        # Extract all symptoms and split them
        all_symptoms = []
        for symptoms_str in df['Symptoms'].dropna():
            # Split on commas and semicolons
            symptoms = [s.strip().lower() for s in symptoms_str.replace(';', ',').split(',')]
            # Remove any parenthetical descriptions
            symptoms = [s.split('(')[0].strip() for s in symptoms]
            all_symptoms.extend(symptoms)
        
        # Remove duplicates and sort
        unique_symptoms = sorted(list(set(all_symptoms)))
        
        return {
            "success": True,
            "symptoms": unique_symptoms
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 