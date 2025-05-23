import json
from typing import Dict, Optional, List
import os
from .chatbot_model.model import ChatbotModel

class ChatbotService:
    def __init__(self):
        """Initialize the chatbot service."""
        # Get the project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        model_dir = os.path.join(project_root, 'models')
        
        # Initialize the model with the correct path
        self.model = ChatbotModel(model_dir=model_dir)
        self.greetings = {
            "hi": "Hello! I'm your healthcare AI assistant. How can I help you today?",
            "hello": "Hi there! I'm here to help with your health concerns. What symptoms are you experiencing?",
            "hey": "Hello! I'm your medical assistant. How can I assist you today?",
            "greetings": "Hello! I'm here to help with your health questions. What would you like to know?",
            "good morning": "Good morning! How can I help you with your health today?",
            "good afternoon": "Good afternoon! What health concerns can I help you with?",
            "good evening": "Good evening! How can I assist you with your health today?"
        }

    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting."""
        message = message.lower().strip()
        return message in self.greetings

    def _get_greeting_response(self, message: str) -> str:
        """Get a response for a greeting."""
        message = message.lower().strip()
        return self.greetings.get(message, "Hello! How can I help you today?")

    def _is_symptom_query(self, message: str) -> bool:
        """Check if the message contains symptoms."""
        # Common symptom-related keywords
        symptom_keywords = [
            "symptom", "symptoms", "feeling", "feel", "experiencing", "having",
            "pain", "ache", "hurts", "hurting", "sick", "ill", "unwell"
        ]
        message = message.lower()
        return any(keyword in message for keyword in symptom_keywords)

    async def get_response(self, user_message: str) -> Dict:
        """Get a response based on the user's message."""
        try:
            if not user_message.strip():
                return {
                    "success": False,
                    "message": None,
                    "error": "Please enter a message"
                }

            # Handle greetings
            if self._is_greeting(user_message):
                return {
                    "success": True,
                    "message": self._get_greeting_response(user_message),
                    "error": None
                }

            # Handle symptom queries
            if self._is_symptom_query(user_message):
                # Get predictions from model
                predictions = self.model.predict(user_message, top_k=3)
                
                if not predictions:
                    return {
                        "success": True,
                        "message": "I understand you're describing symptoms, but I couldn't find a clear match. Could you please provide more specific details about your symptoms?",
                        "error": None
                    }

                # Format the response
                response = "Based on your symptoms, here are the possible diagnoses:\n\n"
                
                for i, pred in enumerate(predictions, 1):
                    response += f"{i}. {pred['response']}\n"
                    response += f"   Confidence: {pred['confidence']:.1%}\n\n"
                
                response += "Note: This is not a medical diagnosis. Please consult a healthcare professional for proper medical advice and treatment."

                return {
                    "success": True,
                    "message": response,
                    "error": None
                }

            # Handle general queries
            return {
                "success": True,
                "message": "I'm here to help with your health concerns. Could you please describe your symptoms or ask a specific health-related question?",
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "message": None,
                "error": f"An error occurred: {str(e)}"
            } 