from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from hpoExtractorv2 import HPODataExtractor

# Initialize FastAPI app and extractor
app = FastAPI(title="HPO Extractor API", version="1.0")
extractor = HPODataExtractor()

# Enable CORS for local React and production domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class TextInput(BaseModel):
    text: str

class TermsInput(BaseModel):
    terms: List[str]

@app.post("/extract")
def extract_from_text(payload: TextInput) -> List[Dict]:
    """
    Extract HPO IDs from a text string (terms can be separated by '/').
    """
    return extractor.process_terms_from_text(payload.text)

@app.post("/extract_terms")
def extract_from_list(payload: TermsInput) -> List[Dict]:
    """
    Extract HPO IDs from a list of terms.
    """
    results = []
    for term in payload.terms:
        result = extractor.process_term(term)
        if result:
            results.append(result)
    return results
