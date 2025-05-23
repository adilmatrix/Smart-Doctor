import React, { useState, useEffect, useRef } from 'react';
import './SymptomForm.css';

const SymptomForm = () => {
  const [symptoms, setSymptoms] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    // Fetch available symptoms when component mounts
    fetch('http://localhost:8000/api/symptoms')
      .then(response => response.json())
      .then(data => setAvailableSymptoms(data.symptoms))
      .catch(err => console.error('Error fetching symptoms:', err));
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms: symptoms }),
      });

      const data = await response.json();
      console.log('API Response:', data);  // Debug log

      if (!response.ok) {
        throw new Error(data.error || 'Something went wrong');
      }

      // Transform the backend response to match frontend expectations
      const transformedPrediction = {
        predictions: data.predictions.map(pred => ({
          disease: pred.disease,
          probability: pred.similarity_score,
          description: pred.symptoms,
          precautions: pred.treatments.split(',').map(t => t.trim())
        })),
        severity_analysis: data.predictions.map(pred => ({
          symptom: pred.symptoms,
          severity: pred.similarity_score > 0.8 ? 'High' : pred.similarity_score > 0.5 ? 'Medium' : 'Low'
        })),
        emergency_warning: data.predictions.some(pred => pred.similarity_score > 0.9)
      };

      setPrediction(transformedPrediction);
      console.log('Set prediction:', transformedPrediction);  // Debug log
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    setDropdownOpen(true);
  };

  const handleSymptomSelect = (symptom) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
      setInputValue('');
      setDropdownOpen(false);
      inputRef.current.focus();
    }
  };

  const handleRemoveSymptom = (symptom) => {
    setSymptoms(symptoms.filter(s => s !== symptom));
  };

  const filteredSymptoms = availableSymptoms.filter(
    s => s.toLowerCase().includes(inputValue.toLowerCase()) && !symptoms.includes(s)
  );

  const handleInputFocus = () => setDropdownOpen(true);
  const handleInputBlur = () => setTimeout(() => setDropdownOpen(false), 150);

  const renderPrediction = () => {
    console.log('Current prediction state:', prediction);  // Debug log
    if (!prediction) return null;

    if (prediction.error) {
      return (
        <div className="error-message">
          <span role="img" aria-label="error">⚠️</span> {prediction.error}
          {prediction.invalid_symptoms && prediction.invalid_symptoms.length > 0 && (
            <div className="invalid-symptoms">
              <strong>Unrecognized symptoms:</strong>
              <ul>
                {prediction.invalid_symptoms.map((sym, idx) => (
                  <li key={idx}>{sym}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      );
    }

    // Most likely disease (first in predictions)
    const topPrediction = prediction.predictions[0];
    console.log('Top prediction:', topPrediction);  // Debug log
    const shareText = encodeURIComponent(
      `My AI health assistant suggests: ${topPrediction.disease} (Confidence: ${(topPrediction.probability * 100).toFixed(1)}%).\nThis is not a diagnosis. Please consult a healthcare professional.`
    );
    const shareUrl = encodeURIComponent(window.location.href);

    return (
      <div className="prediction-results fade-in">
        {/* Summary Card */}
        <div className={`summary-card${prediction.emergency_warning ? ' emergency' : ''}`}> 
          <div className="top-disease-header">
            <div className="disease-icon-block">
              <span className="summary-icon" role="img" aria-label="diagnosis">🩺</span>
            </div>
            <div className="disease-info-block">
              <h2>{topPrediction.disease}</h2>
              <div className="confidence">Confidence: {(topPrediction.probability * 100).toFixed(1)}%</div>
              <div className="description">{topPrediction.description}</div>
            </div>
            <div className="precautions-block">
              <h4>Recommended Precautions</h4>
              <ul>
                {topPrediction.precautions.map((precaution, i) => (
                  <li key={i}>{precaution}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
        {prediction.emergency_warning && (
          <div className="emergency-warning">
            <span role="img" aria-label="emergency">🚨</span> Emergency Warning: Immediate medical attention may be required!
          </div>
        )}

        {/* Invalid Symptoms */}
        {prediction.invalid_symptoms && prediction.invalid_symptoms.length > 0 && (
          <div className="invalid-symptoms-card">
            <span role="img" aria-label="warning">⚠️</span> <strong>Unrecognized symptoms:</strong>
            <ul>
              {prediction.invalid_symptoms.map((sym, idx) => (
                <li key={idx}>{sym}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Severity Analysis */}
        <div className="severity-analysis">
          <h3>Symptom Severity Analysis</h3>
          <ul>
            {prediction.severity_analysis.map((item, index) => (
              <li key={index} className={`severity-${item.severity.toLowerCase()}`}>
                <span className="severity-dot" />
                {item.symptom} - {item.severity} Severity
              </li>
            ))}
          </ul>
        </div>
        <h3>Other Possible Conditions</h3>
        {/* Disease Predictions */}
        <div className="disease-predictions">
          {prediction.predictions.slice(1).map((pred, index) => (
            <div key={index} className="prediction-card"> 
              <div className="disease-header">
                <span className="disease-icon" role="img" aria-label="disease">🦠</span>
                <h4>{pred.disease}</h4>
                <span className="confidence">{(pred.probability * 100).toFixed(1)}%</span>
              </div>
              <p className="description">{pred.description}</p>
              <div className="precautions">
                <h5>Recommended Precautions:</h5>
                <ul>
                  {pred.precautions.map((precaution, i) => (
                    <li key={i}>{precaution}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>

        <div className="disclaimer">
          <p>⚠️ This is not a medical diagnosis. Please consult a healthcare professional for proper medical advice.</p>
        </div>
        {/* Share Buttons */}
        <div className="share-section">
          <span>Share your result:</span>
          <a
            className="share-btn fb"
            href={`https://www.facebook.com/sharer/sharer.php?u=${shareUrl}&quote=${shareText}`}
            target="_blank"
            rel="noopener noreferrer"
            title="Share on Facebook"
          >
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/facebook.svg" alt="Facebook" style={{width: 24, height: 24, verticalAlign: 'middle'}} />
          </a>
          <a
            className="share-btn wa"
            href={`https://wa.me/?text=${shareText}%20${shareUrl}`}
            target="_blank"
            rel="noopener noreferrer"
            title="Share on WhatsApp"
          >
            <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/whatsapp.svg" alt="WhatsApp" style={{width: 24, height: 24, verticalAlign: 'middle'}} />
          </a>
        </div>
      </div>
    );
  };

  return (
    <div className="symptom-form-container">
      <h2>Healthcare AI Assistant</h2>
      <form onSubmit={handleSubmit} autoComplete="off">
        <div className="input-group">
          <label htmlFor="symptoms">Search or select your symptoms:</label>
          <div className="symptom-searchbox">
            {symptoms.map(symptom => (
              <span className="selected-symptom" key={symptom}>
                {symptom}
                <button type="button" onClick={() => handleRemoveSymptom(symptom)}>&times;</button>
              </span>
            ))}
            <input
              type="text"
              id="symptoms"
              ref={inputRef}
              value={inputValue}
              onChange={handleInputChange}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
              placeholder="Type or select symptoms"
            />
            {dropdownOpen && filteredSymptoms.length > 0 && (
              <ul className="symptom-dropdown">
                {filteredSymptoms.map(symptom => (
                  <li key={symptom} onMouseDown={() => handleSymptomSelect(symptom)}>
                    {symptom}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
        <button type="submit" disabled={loading || symptoms.length === 0}>
          {loading ? 'Analyzing...' : 'Get Prediction'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}
      {renderPrediction()}
    </div>
  );
};

export default SymptomForm; 