# Configuration settings for the VKU Document Q&A Assistant
import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_api_key_here")

# App settings
APP_TITLE = "Document Q&A VKU Assistant"
APP_ICON = "🤖"
DEFAULT_LANGUAGE = "Vietnamese"

# Document processing settings
MAX_CHUNK_SIZE = 850
CHUNK_OVERLAP = 300
ALLOWED_FILE_TYPES = ["txt", "pdf", "docx", "json"]

# AI model settings
TEMPERATURE = 0.9
TOP_P = 0.2
MAX_OUTPUT_TOKENS = 8192

# Search settings
USE_INTERNET_DEFAULT = False
MAX_SEARCH_RESULTS = 3