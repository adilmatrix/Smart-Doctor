# model_training.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from .symptom_utils import preprocess_symptoms, tokenize_symptoms

def train_model():
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # Load the data
    df = pd.read_csv(os.path.join(project_root, 'dataset/diseases_symptoms.csv'))
    
    # Create TF-IDF vectorizer for symptoms with improved parameters
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=tokenize_symptoms,
        preprocessor=preprocess_symptoms,
        token_pattern=None,
        min_df=1,  # Include all terms
        max_df=0.95,  # Maximum document frequency
        ngram_range=(1, 3)  # Use unigrams, bigrams, and trigrams
    )
    
    # Transform symptoms to TF-IDF matrix
    symptoms_matrix = vectorizer.fit_transform(df['Symptoms'])
    
    # Calculate similarity matrix between all diseases
    similarity_matrix = cosine_similarity(symptoms_matrix)
    
    # Save the model components in the models directory
    model_dir = os.path.dirname(__file__)
    
    # Save the model components
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
    joblib.dump(symptoms_matrix, os.path.join(model_dir, 'symptoms_matrix.joblib'))
    joblib.dump(df, os.path.join(model_dir, 'disease_data.joblib'))
    joblib.dump(similarity_matrix, os.path.join(model_dir, 'similarity_matrix.joblib'))
    
    # Print some statistics about the model
    print("\nModel training completed successfully!")
    print(f"Total diseases in dataset: {len(df)}")
    print(f"Total unique symptoms: {len(vectorizer.get_feature_names_out())}")
    
    # Print some example symptom matches
    print("\nExample symptom matches:")
    test_symptoms = [
        "fever, headache, fatigue",
        "joint pain, swelling, stiffness",
        "chest pain, shortness of breath"
    ]
    
    for test_symptom in test_symptoms:
        # Transform test symptoms
        test_vector = vectorizer.transform([test_symptom])
        # Calculate similarity with all diseases
        similarities = cosine_similarity(test_vector, symptoms_matrix).flatten()
        # Get top 3 matches
        top_indices = similarities.argsort()[-3:][::-1]
        
        print(f"\nInput symptoms: {test_symptom}")
        for idx in top_indices:
            print(f"Matched disease: {df['Name'].iloc[idx]} (similarity: {similarities[idx]:.2f})")
            print(f"Actual symptoms: {df['Symptoms'].iloc[idx]}")

if __name__ == "__main__":
    train_model() 