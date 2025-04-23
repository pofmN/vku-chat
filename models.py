import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from sentence_transformers import SentenceTransformer
#from config.secretKey import GEMINI_API_KEY
from config.secretKey import GEMINI_API_KEY
import logging

#GEMINI_API_KEY = "AIzaSyAni4v1XkTytjIZtjl1-wov-DER9QpwSHs"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_embedding_models():
    """Load and cache embedding models"""
    return SentenceTransformer('keepitreal/vietnamese-sbert')

@st.cache_resource
def initialize_gemini():
    """Initialize and cache Gemini client"""
    logger.debug(f"Initializing Gemini with API key: {GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-5:]}")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-2.0-flash-lite')