import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model Configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
CHATBOT_MODEL_PATH = os.path.join(MODEL_DIR, 'chatbot_model')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.joblib')

# Dataset Configuration
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
CHATBOT_DATA_PATH = os.path.join(DATASET_DIR, 'chatbot_data.json')

# Chatbot Configuration
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in chat history

# HuggingFace API Configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')  # Get your free API key from huggingface.co
HUGGINGFACE_MODEL = "facebook/blenderbot-400M-distill"  # Free model for chat

# Chatbot Configuration
CHATBOT_SYSTEM_PROMPT = """You are a medical assistant chatbot that helps users understand their symptoms and provides general medical information. 
You should:
1. Be empathetic and professional
2. Provide clear, accurate medical information
3. Always remind users that you're not a replacement for professional medical advice
4. Ask clarifying questions when needed
5. Use simple, understandable language
6. Include relevant precautions and next steps when appropriate

Remember:
- Never make definitive diagnoses
- Always encourage consulting healthcare professionals for serious concerns
- Be cautious with emergency situations
- Maintain patient privacy and confidentiality
- Use evidence-based information
""" 