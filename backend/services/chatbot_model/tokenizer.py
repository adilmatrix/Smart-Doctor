from typing import List
 
def symptom_tokenizer(text: str) -> List[str]:
    """Tokenize symptoms by splitting on commas and cleaning."""
    return [s.strip().lower() for s in text.split(',')] 