# Smart Doctor AI

An intelligent medical assistant system that combines symptom prediction with an AI chatbot for comprehensive medical guidance.

## Features

- Symptom-based disease prediction
- AI-powered medical chatbot
- Real-time symptom analysis
- Interactive user interface

## Project Structure

```
Smart-Doctor/
├── backend/           # FastAPI backend
│   ├── services/     # Service implementations
│   └── app.py        # Main application
├── frontend/         # React frontend
├── model/           # Predictor model files
├── models/          # Chatbot model files
└── dataset/         # Training datasets
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend:
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

4. Run the frontend:
```bash
cd frontend
npm install
npm start
```

## Technologies Used

- Backend: FastAPI, Python
- Frontend: React.js
- AI Models: 
  - Predictor: TF-IDF, Cosine Similarity
  - Chatbot: Logistic Regression
- Data Processing: Pandas, NumPy
- Machine Learning: Scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.