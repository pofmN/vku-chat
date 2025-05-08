# To install: pip install tavily-python
import os
import tavily
import dotenv
import re
from dotenv import load_dotenv
from utils import clean_text
from typing import List, Dict
from tavily import TavilyClient

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
print("TAVILY_API_KEY:", TAVILY_API_KEY)
client = TavilyClient(api_key=TAVILY_API_KEY)

def preprocess_tavily_response(result: List[dict]) -> List[dict]:
    """Process the response from Tavily API to extract relevant information."""
    processed_response = []
    for result in result:
        processed_result = {
            'title': result.get('title', ''),
            'content': result.get('content', ''),
            'url': result.get('url', ''),
            'source': result.get('source', ''),
            'snippet': result.get('snippet', '')
        }
        # Filter out empty content
        if processed_result['content'].strip():
            processed_response.append(processed_result)
    return processed_response


def get_tavily_response(query: str) -> List[dict]:
    """
    Get search results from Tavily API and preprocess them.
    
    Args:
        query (str): Search query
        
    Returns:
        List[Dict]: List of processed search results
    """
    try:
        response = client.search(
            query=query+'vku',
            search_type="web",
            num_results=5,
            num_results_per_page=5,
            region="vn",
            language="vi"
        )
        if 'results' in response:
            processed_results = preprocess_tavily_response(response['results'])
            result_texts = []
            for result in processed_results:
                # Include the most important fields
                result_text = f"{result['content']} {result['url']} {result['source']}"
                # Apply special cleaning for online text
                #result_text = clean_online_text(result_text)
                result_texts.append(result_text)
            combined_result = "\n".join(result_texts)
            #print(f"Combined result: {combined_result}")
            return combined_result
        else:
            print("No results found in Tavily API response.")
            return []
    except Exception as e:
        print(f"Error fetching data from Tavily API: {e}")
        return []
    
def clean_online_text(text: str) -> str:
    """Clean text retrieved from online sources with extra spacing issues."""
    # Remove extra spaces between characters (when every character is separated)
    if '  ' in text and len(text.split()) > len(text) / 3:
        # This pattern matches text where most characters are separated by spaces
        text = text.replace('  ', ' ')
        # Join characters that are separated by single spaces when they shouldn't be
        text = ''.join(text.split())
    
    # Normal cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text