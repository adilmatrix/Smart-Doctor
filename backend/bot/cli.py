import os
import sys
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from healthcare_bot import HealthcareBot

def show_available_symptoms(bot: HealthcareBot):
    """
    Display all available symptoms that the model recognizes.
    
    Args:
        bot (HealthcareBot): The healthcare bot instance
    """
    print("\nAvailable symptoms:")
    symptoms = sorted(bot.get_available_symptoms())
    for i in range(0, len(symptoms), 3):
        row = symptoms[i:i+3]
        print("".join(f"{s:<30}" for s in row))
    print()

def get_symptoms() -> List[str]:
    """
    Get symptoms from user input.
    
    Returns:
        List[str]: List of symptoms
    """
    print("\nEnter your symptoms (one per line). Press Enter twice when done:")
    symptoms = []
    while True:
        symptom = input().strip().lower()
        if not symptom:
            break
        if symptom == 'quit':
            sys.exit(0)
        if symptom == 'help':
            print("\nCommands:")
            print("- Enter a symptom (one per line)")
            print("- Press Enter twice when done")
            print("- Type 'quit' to exit")
            print("- Type 'list' to see available symptoms")
            print("- Type 'help' to see this message")
            continue
        if symptom == 'list':
            show_available_symptoms(bot)
            print("\nEnter your symptoms (one per line):")
            continue
        symptoms.append(symptom)
    return symptoms

def test_model_accuracy(bot: HealthcareBot, dataset_path: str, n_samples: int = 100):
    """Test model accuracy on a sample of the dataset."""
    print(f"\nTesting model accuracy on {n_samples} samples from: {dataset_path}")
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        
        # Get available symptoms from the model
        available_symptoms = set(s.lower() for s in bot.get_available_symptoms())
        
        # Group by disease to ensure diversity
        disease_groups = df.groupby('disease')
        samples_per_disease = max(1, n_samples // len(disease_groups))
        
        X = []
        y_true = []
        for disease, group in disease_groups:
            disease_samples = 0
            for i, row in group.iterrows():
                try:
                    # Get symptoms for this disease
                    symptoms = row['symptoms'].lower().split(',')
                    symptoms = [s.strip() for s in symptoms if s.strip()]
                    
                    # Only use symptoms that are in our model's vocabulary
                    valid_symptoms = [s for s in symptoms if s in available_symptoms]
                    
                    if valid_symptoms:
                        # Use only a subset of symptoms (2-4 symptoms) to make it more realistic
                        num_symptoms = min(len(valid_symptoms), np.random.randint(2, 5))
                        selected_symptoms = np.random.choice(valid_symptoms, num_symptoms, replace=False)
                        
                        X.append(list(selected_symptoms))
                        y_true.append(row['disease'])
                        disease_samples += 1
                        if disease_samples >= samples_per_disease:
                            break
                except Exception as e:
                    print(f"Warning: Skipping row {i} due to error: {str(e)}")
                    continue
                
        if not X:
            print("No valid samples found in the dataset!")
            return
            
        print(f"Successfully loaded {len(X)} samples from {len(disease_groups)} different diseases")
        print(f"Average symptoms per sample: {np.mean([len(x) for x in X]):.1f}")
        
        y_pred = []
        for symptoms in X:
            try:
                result = bot.process_symptoms(symptoms)
                if 'disease' in result:
                    y_pred.append(result['disease'])
                else:
                    y_pred.append('unknown')
            except Exception as e:
                print(f"Warning: Error processing symptoms {symptoms}: {str(e)}")
                y_pred.append('unknown')
                
        if y_pred:
            acc = accuracy_score(y_true, y_pred)
            print(f"\nModel accuracy on {len(X)} samples: {acc:.2%}")
            
            # Show predictions by disease
            print("\nPredictions by disease:")
            disease_results = {}
            for true, pred in zip(y_true, y_pred):
                if true not in disease_results:
                    disease_results[true] = {'correct': 0, 'total': 0}
                disease_results[true]['total'] += 1
                if true == pred:
                    disease_results[true]['correct'] += 1
            
            for disease, results in disease_results.items():
                acc = results['correct'] / results['total']
                print(f"{disease}: {results['correct']}/{results['total']} correct ({acc:.2%})")
            
            # Show some example predictions
            print("\nSample predictions:")
            for i in range(min(5, len(X))):
                print(f"Symptoms: {X[i]} | True: {y_true[i]} | Predicted: {y_pred[i]}")
                
            # Show confusion matrix for top 10 diseases
            print("\nConfusion matrix for top 10 diseases:")
            top_diseases = sorted(disease_results.items(), key=lambda x: x[1]['total'], reverse=True)[:10]
            top_disease_names = [d[0] for d in top_diseases]
            
            # Create confusion matrix
            cm = np.zeros((len(top_disease_names), len(top_disease_names)))
            for true, pred in zip(y_true, y_pred):
                if true in top_disease_names and pred in top_disease_names:
                    i = top_disease_names.index(true)
                    j = top_disease_names.index(pred)
                    cm[i, j] += 1
            
            # Print confusion matrix
            print("\nTrue Disease (rows) vs Predicted Disease (columns):")
            print("Disease".ljust(30), "|", " ".join(f"{d[:10]:>10}" for d in top_disease_names))
            print("-" * (30 + 11 * len(top_disease_names)))
            for i, disease in enumerate(top_disease_names):
                print(f"{disease[:30]:<30} |", " ".join(f"{int(cm[i,j]):>10}" for j in range(len(top_disease_names))))
        else:
            print("No predictions were made successfully!")
            
    except Exception as e:
        print(f"Error testing model accuracy: {str(e)}")
        print("Please check if the dataset file exists and has the correct format.")

def main():
    # Initialize bot with the correct model path
    global bot  # Make bot accessible to get_symptoms()
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    bot = HealthcareBot(model_path)
    
    print("Welcome to the Healthcare Bot!")
    print("I can help you predict possible diseases based on your symptoms.")
    print("\nCommands:")
    print("- Enter symptoms one per line")
    print("- Press Enter twice when done")
    print("- Type 'quit' to exit")
    print("- Type 'list' to see available symptoms")
    print("- Type 'help' to see this message")
    print("- Type 'test' to test model accuracy on the dataset")
    
    while True:
        user_input = input("\nType 'test' to test model accuracy, or press Enter to continue: ").strip().lower()
        if user_input == 'test':
            # Use the dataset from Hugging Face
            dataset_path = "hf://datasets/shanover/disease_symptoms_prec_full/disease_sympts_prec_full.csv"
            test_model_accuracy(bot, dataset_path, n_samples=100)
            continue
            
        # Get symptoms
        symptoms = get_symptoms()
        if not symptoms:
            print("No symptoms provided. Please try again or type 'quit' to exit.")
            continue
            
        # Process symptoms and generate response
        try:
            prediction = bot.process_symptoms(symptoms)
            response = bot.generate_response(prediction)
            print("\n" + response)
        except Exception as e:
            print(f"\nError processing symptoms: {str(e)}")
            show_available_symptoms(bot)
            continue
            
        # Ask if user wants to continue
        while True:
            choice = input("\nWould you like to analyze more symptoms? (yes/no): ").lower()
            if choice in ['yes', 'no']:
                break
            print("Please enter 'yes' or 'no'")
            
        if choice != 'yes':
            break
            
    print("\nThank you for using the Healthcare Bot. Take care!")

if __name__ == "__main__":
    main() 