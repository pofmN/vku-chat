import requests
import logging
import streamlit as st
from bs4 import BeautifulSoup

# Maximum number of search results to consider
MAX_RESULTS = 3
# Maximum length of context extracted from each source
MAX_CONTEXT_LENGTH = 2000

def get_online_context(query):
    """Search online for relevant information and extract context"""
    try:
        # Add "VKU university" to the query for more relevant results
        enhanced_query = f"{query} VKU university admissions"
        
        # Search using a search API (this is a placeholder - replace with actual search API)
        search_results = perform_search(enhanced_query)
        
        if not search_results:
            return "No relevant online information found."
        
        # Extract content from top search results
        context = ""
        for result in search_results[:MAX_RESULTS]:
            try:
                content = extract_content_from_url(result.get('url', ''))
                if content:
                    context += content[:MAX_CONTEXT_LENGTH] + "\n\n"
            except Exception as e:
                logging.error(f"Error extracting content from {result.get('url', '')}: {str(e)}")
                continue
        
        if not context:
            return "Failed to extract relevant content from online sources."
        
        return context.strip()
    
    except Exception as e:
        logging.error(f"Error searching online: {str(e)}")
        return f"Error searching online: {str(e)}"

def perform_search(query):
    """Perform a web search (placeholder - implement with actual search API)"""
    # This is a placeholder implementation
    # In a real application, you would use a search API like Google Custom Search API,
    # Bing Search API, or another search service
    
    # For the purposes of this stub, we'll return a sample result structure
    # pointing to the official VKU admissions page
    
    return [
        {
            'title': 'VKU Admissions',
            'url': 'https://tuyensinh.vku.udn.vn/',
            'snippet': 'Official admissions website for Vietnam-Korea University of Information and Communication Technology'
        },
        {
            'title': 'VKU - University Information',
            'url': 'https://vku.udn.vn/',
            'snippet': 'Vietnam-Korea University of Information and Communication Technology'
        }
    ]

def extract_content_from_url(url):
    """Extract and clean content from a URL"""
    try:
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Request the page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for error status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text from the page
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    except Exception as e:
        logging.error(f"Error extracting content from {url}: {str(e)}")
        return ""