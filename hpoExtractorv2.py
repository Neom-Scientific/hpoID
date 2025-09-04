try:
    import streamlit as st  # type: ignore[import-not-found]
except ImportError:
    st = None  # Streamlit is optional for API usage
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import logging
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional
from urllib.parse import quote
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load .env
load_dotenv()

# Initialize Groq API client directly
openai = OpenAI(
    api_key=os.getenv("HPO_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)



class HPODataExtractor:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1 Safari/605.1.15"
        ]
        self.abbreviation_map = {
            'HTN': 'Hypertension',
            'DM': 'Diabetes Mellitus',
            'CKD': 'Chronic Kidney Disease',
            'PIVD': 'Herniation of intervertebral nuclei',
            'HLD': 'Hyperlipidemia',
            'CAD': 'Coronary Artery Disease',
            'COPD': 'Chronic Obstructive Pulmonary Disease',
            'CHF': 'Congestive Heart Failure',
            'MI': 'Myocardial Infarction',
            'AF': 'Atrial Fibrillation'
        }
        self.corrections = {
            'encephlopathy': 'Encephalopathy',
            'hypothyroid': 'Hypothyroidism',
            'non dm': 'Diabetes Mellitus',
            'old d 12 fracture': 'Fractured thoracic vertebra',
            'ckd': 'Chronic Kidney Disease',
            'pivd': 'Herniation of intervertebral nuclei'
        }

    def get_random_header(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://hpo.jax.org/'
        }

    def map_abbreviation(self, term: str) -> str:
        """Map medical abbreviations to full forms and correct misspellings."""
        term_clean = term.strip().lower()
        term_clean = self.corrections.get(term_clean, term_clean)
        return self.abbreviation_map.get(term_clean.upper(), term_clean.capitalize())

    def get_hpo_id(self, term: str) -> Optional[Dict]:
        """Use Grok's API (llama-3.3-70b-versatile) to find the most relevant HPO ID."""
        term_lower = term.lower()
        mapped_term = self.map_abbreviation(term)

        # Handle non-phenotypic terms
        if term_lower == 'non dm':
            return {'Term': term, 'HPO ID': 'Not applicable'}

        try:
            # Construct prompt for Grok API
            prompt = (
                f"Given the medical term '{term}' (mapped to '{mapped_term}'), "
                "find the most relevant Human Phenotype Ontology (HPO) ID. "
                "Return only the HPO ID (format: HP:XXXXXXX). "
                "If no HPO term applies, return 'Not found'."
            )
            response = openai.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a medical ontology expert with access to the Human Phenotype Ontology (HPO)."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            result = response.choices[0].message.content.strip()

            # Extract HPO ID
            hpo_id = "Not found"
            match = re.search(r"(HP:\d{7})", result)
            if match:
                hpo_id = match.group(1)

            return {'Term': term, 'HPO ID': hpo_id}
        except Exception as e:
            logger.error(f"Grok API error for {term}: {e}")
            return {'Term': term, 'HPO ID': 'Error'}

    def process_term(self, term: str) -> Optional[Dict]:
        """Process a single term to get its HPO ID."""
        if not term or term.lower() in ['nan', 'null', '']:
            return None
        term = term.strip()
        return self.get_hpo_id(term)

    def process_terms_from_text(self, text: str) -> List[Dict]:
            """
            Process a text string containing multiple terms separated by common delimiters.
            Splits on , . - ? / \ | and whitespace.
            """
            # Split on multiple delimiters using regex
            terms = re.split(r'[,\.\-?\/\\|\s]+', text)
            # Remove empty strings
            terms = [t.strip() for t in terms if t.strip()]

            results = []
            for term in terms:
                result = self.process_term(term)
                if result:
                    results.append(result)
            return results

def create_hpo_analysis_interface():
    """Create the HPO analysis interface."""
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install it to use the UI.")
    st.header(" HPO ID Extractor")
    
    # Initialize HPO extractor
    if 'hpo_extractor' not in st.session_state:
        st.session_state.hpo_extractor = HPODataExtractor()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a CSV, Excel, or Text file", type=['csv', 'xlsx', 'txt'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                content = uploaded_file.read().decode('utf-8')
                terms = [line.strip() for line in content.split('\n') if line.strip()]
                df = pd.DataFrame({'terms': terms})
            
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Analyze HPO IDs"):
                if len(df) > 0:
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, row in df.iterrows():
                        terms = None
                        for col in df.columns:
                            if pd.notna(row[col]) and str(row[col]).strip():
                                terms = str(row[col]).strip()
                                break
                        
                        if terms:
                            status_text.text(f"Analyzing: {terms}")
                            term_results = st.session_state.hpo_extractor.process_terms_from_text(terms)
                            results.extend(term_results)
                            for result in term_results:
                                st.write(f"✓ {result['Term']} -> {result['HPO ID']}")
                            
                            progress_bar.progress((i + 1) / len(df))
                            time.sleep(0.5)  # Delay to prevent rate limiting
                        
                    if results:
                        st.session_state.hpo_results = results
                        st.success(f"Successfully analyzed {len(results)} terms!")
                        
                        # Generate and offer CSV download
                        results_df = pd.DataFrame(results, columns=['Term', 'HPO ID'])
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"hpo_analysis_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                    else:
                        st.warning("No valid terms found in the file")
                else:
                    st.warning("No data found in the uploaded file")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Display results
    if 'hpo_results' in st.session_state and st.session_state.hpo_results:
        st.subheader("HPO Analysis Results")
        results_df = pd.DataFrame(st.session_state.hpo_results, columns=['Term', 'HPO ID'])
        st.dataframe(results_df)

def main():
    """Main application."""
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install it to use the UI.")
    st.set_page_config(
        page_title="HPO ID Extractor",
        page_icon="🧬",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("<h1 class='main-header'>🧬 HPO ID Extractor</h1>", unsafe_allow_html=True)
    
    create_hpo_analysis_interface()

if __name__ == "__main__":
    if st is None:
        print("Streamlit is not installed. Run `pip install streamlit` to use the UI.")
    else:
        main()
