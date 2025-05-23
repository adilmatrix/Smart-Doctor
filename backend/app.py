import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from services.chatbot_service import ChatbotService
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(
    title="Healthcare AI Assistant",
    description="An AI-powered healthcare assistant that predicts diseases based on symptoms",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_logger")

# Middleware to log each API call
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"API call: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def refit_vectorizer(disease_data):
    """Refit the vectorizer with the disease data."""
    try:
        # Create a new vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit the vectorizer with all symptoms
        symptoms_texts = disease_data['Symptoms'].fillna('').astype(str)
        vectorizer.fit(symptoms_texts)
        
        # Transform all symptoms to create the matrix
        symptoms_matrix = vectorizer.transform(symptoms_texts)
        
        return vectorizer, symptoms_matrix
    except Exception as e:
        logger.error(f"Error refitting vectorizer: {str(e)}")
        return None, None

# Load model components
try:
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Load predictor model components
    predictor_model_dir = os.path.join(project_root, 'model')
    logger.info(f"Loading predictor model components from: {predictor_model_dir}")
    
    # Load disease data first
    disease_data = joblib.load(os.path.join(predictor_model_dir, 'disease_data.joblib'))
    logger.info("Disease data loaded successfully")
    
    # Verify the data
    if not isinstance(disease_data, pd.DataFrame):
        raise ValueError("Disease data is not a pandas DataFrame")
    if 'Symptoms' not in disease_data.columns:
        raise ValueError("Symptoms column not found in disease data")
    
    # Refit the vectorizer with the disease data
    vectorizer, symptoms_matrix = refit_vectorizer(disease_data)
    if vectorizer is None or symptoms_matrix is None:
        raise ValueError("Failed to refit vectorizer")
    
    logger.info("Vectorizer refitted successfully")
    logger.info(f"Loaded {len(disease_data)} diseases from dataset")
    
except Exception as e:
    logger.error(f"Error loading predictor model components: {str(e)}")
    vectorizer = None
    symptoms_matrix = None
    disease_data = None

# Initialize chatbot service for chat endpoint
chatbot = ChatbotService()

class SymptomRequest(BaseModel):
    symptoms: List[str]

class ChatRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    predictions: List[dict]
    error: Optional[str] = None

class ChatResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

class SymptomsResponse(BaseModel):
    success: bool
    symptoms: Optional[List[str]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "Healthcare AI Assistant API"}

@app.get("/api/symptoms", response_model=SymptomsResponse)
async def get_symptoms():
    """Get the list of all available symptoms from the dataset."""
    try:
        if disease_data is None:
            logger.error("Disease data is None")
            return SymptomsResponse(
                success=False,
                error="Model not loaded. Please try again later."
            )
        
        logger.info("Extracting symptoms from disease data")
        
        # Extract all unique symptoms from the dataset
        all_symptoms = set()
        for symptoms in disease_data['Symptoms']:
            if isinstance(symptoms, str):
                # Split on commas and clean up
                symptom_list = [s.strip().lower() for s in symptoms.split(',')]
                all_symptoms.update(symptom_list)
        
        symptoms_list = sorted(list(all_symptoms))
        logger.info(f"Found {len(symptoms_list)} unique symptoms")
        
        if not symptoms_list:
            logger.error("No symptoms found in dataset")
            return SymptomsResponse(
                success=False,
                error="No symptoms found in dataset"
            )
        
        return SymptomsResponse(
            success=True,
            symptoms=symptoms_list
        )
    except Exception as e:
        logger.error(f"Error getting symptoms: {str(e)}")
        return SymptomsResponse(
            success=False,
            error=f"Error getting symptoms: {str(e)}"
        )

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_disease(request: SymptomRequest):
    """Predict diseases based on symptoms using the old model."""
    if vectorizer is None or symptoms_matrix is None or disease_data is None:
        logger.error("Model components not loaded")
        return PredictionResponse(
            predictions=[],
            error="Model not loaded. Please try again later."
        )

    try:
        # Join symptoms into a single string
        symptoms_text = ', '.join(request.symptoms)
        logger.info(f"Processing symptoms: {symptoms_text}")
        
        # Transform input symptoms - ensure we're using the correct method
        input_vector = vectorizer.transform([symptoms_text.lower()])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(input_vector, symptoms_matrix).flatten()
        
        # Get top 5 matches with similarity > 0.1
        top_indices = np.argsort(similarity_scores)[::-1][:5]
        top_scores = similarity_scores[top_indices]
        
        predictions = []
        for idx, score in zip(top_indices, top_scores):
            if score > 0.1:  # Only include matches with reasonable similarity
                disease_name = disease_data.iloc[idx]['Name']
                disease_symptoms = disease_data.iloc[idx]['Symptoms']
                disease_treatments = disease_data.iloc[idx]['Treatments']
                
                predictions.append({
                    "disease": disease_name,
                    "similarity_score": float(score),
                    "symptoms": disease_symptoms,
                    "treatments": disease_treatments
                })

        if not predictions:
            logger.warning("No matching diseases found")
            return PredictionResponse(
                predictions=[],
                error="No matching diseases found for the given symptoms"
            )

        logger.info(f"Found {len(predictions)} matching diseases")
        return PredictionResponse(predictions=predictions)

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return PredictionResponse(
            predictions=[],
            error=f"Error making prediction: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Get a response from the medical chatbot using the new model."""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Please enter a message")
        
        response = await chatbot.get_response(request.message)
        
        if not response["success"]:
            logger.error(f"Chatbot error: {response['error']}")
            raise HTTPException(status_code=500, detail=response["error"])
        
        return {
            "success": True,
            "message": response["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000) 