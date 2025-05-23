import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# Import preprocessing/tokenization from symptom_utils
try:
    from .symptom_utils import preprocess_symptoms, tokenize_symptoms
except ImportError:
    from symptom_utils import preprocess_symptoms, tokenize_symptoms

def main():
    # Load the full dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    df = pd.read_csv(os.path.join(project_root, 'dataset/diseases_symptoms.csv'))

    # Remove diseases with only one sample
    disease_counts = df['Name'].value_counts()
    df = df[df['Name'].isin(disease_counts[disease_counts > 1].index)]

    # Try stratified split, fall back to random split if needed
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Name'])
    except ValueError as e:
        print(f"[Warning] Stratified split failed: {e}\nFalling back to random split without stratification.")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=None)

    # Vectorizer setup
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=tokenize_symptoms,
        preprocessor=preprocess_symptoms,
        token_pattern=None,
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 3)
    )

    # Fit on train symptoms
    symptoms_matrix = vectorizer.fit_transform(train_df['Symptoms'])
    train_disease_data = train_df.reset_index(drop=True)

    # Evaluate on test set
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        symptoms = row['Symptoms']
        true_label = row['Name']
        test_vector = vectorizer.transform([symptoms])
        similarities = cosine_similarity(test_vector, symptoms_matrix).flatten()
        top_idx = similarities.argmax()
        predicted_label = train_disease_data.iloc[top_idx]['Name']
        y_true.append(true_label)
        y_pred.append(predicted_label)

    # Metrics
    print("Test set size:", len(test_df))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average='weighted', zero_division=0))
    print("F1-score:", f1_score(y_true, y_pred, average='weighted', zero_division=0))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main() 