def preprocess_symptoms(text):
    """Preprocess symptoms by converting to lowercase and stripping whitespace"""
    if not isinstance(text, str):
        return ''
    return text.lower().strip()

def tokenize_symptoms(text):
    """Tokenize symptoms by splitting on commas and handling special cases"""
    if not isinstance(text, str):
        return []
    symptoms = []
    for part in text.split(','):
        subparts = part.split(';')
        for subpart in subparts:
            symptom = subpart.strip().lower()
            if '(' in symptom:
                symptom = symptom.split('(')[0].strip()
            if symptom:
                symptoms.append(symptom)
    return symptoms 