import os
import sys
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def tokenize_symptoms(text):
    """Tokenize symptoms by splitting on commas and handling special cases"""
    if not isinstance(text, str):
        return []
    # Split on commas and handle potential semicolons
    symptoms = []
    for part in text.split(','):
        # Further split on semicolons if present
        subparts = part.split(';')
        for subpart in subparts:
            # Clean up the symptom text
            symptom = subpart.strip().lower()
            # Remove parenthetical descriptions
            if '(' in symptom:
                symptom = symptom.split('(')[0].strip()
            if symptom:  # Only add non-empty symptoms
                symptoms.append(symptom)
    return symptoms

def preprocess_symptoms(text):
    """Preprocess symptoms by converting to lowercase and stripping whitespace"""
    if not isinstance(text, str):
        return ''
    return text.lower().strip()

def load_model():
    """Load the trained model components"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, 'models')
    
    try:
        vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        symptoms_matrix = joblib.load(os.path.join(model_dir, 'symptoms_matrix.joblib'))
        disease_data = joblib.load(os.path.join(model_dir, 'disease_data.joblib'))
        return vectorizer, symptoms_matrix, disease_data
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def predict_disease(symptoms):
    """Predict diseases based on input symptoms"""
    vectorizer, symptoms_matrix, disease_data = load_model()
    
    # Transform input symptoms
    test_vector = vectorizer.transform([symptoms])
    
    # Calculate similarity with all diseases
    similarities = cosine_similarity(test_vector, symptoms_matrix).flatten()
    
    # Get top 5 matches with similarity > 0.1
    top_indices = similarities.argsort()[-5:][::-1]
    top_matches = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.1]
    
    if not top_matches:
        return "No matching diseases found for the given symptoms."
    
    # Format the results
    results = []
    for idx, similarity in top_matches:
        disease = disease_data.iloc[idx]
        result = {
            'disease': disease['Name'],
            'similarity': f"{similarity:.2f}",
            'symptoms': disease['Symptoms'],
            'treatments': disease['Treatments']
        }
        results.append(result)
    
    return results

def main():
    print("\nDisease Prediction System")
    print("Enter symptoms (comma-separated) or 'quit' to exit")
    
    while True:
        print("\n" + "="*50)
        symptoms = input("\nEnter symptoms: ").strip()
        
        if symptoms.lower() == 'quit':
            break
        
        if not symptoms:
            print("Please enter at least one symptom.")
            continue
        
        results = predict_disease(symptoms)
        
        if isinstance(results, str):
            print(f"\n{results}")
            continue
        
        print("\nTop Matching Diseases:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['disease']} (Similarity: {result['similarity']})")
            print(f"   Symptoms: {result['symptoms']}")
            print(f"   Treatments: {result['treatments']}")
            print("-" * 50)

if __name__ == "__main__":
    main() 