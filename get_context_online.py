from serpapi import GoogleSearch
import dotenv
from dotenv import load_dotenv
import os

load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
SERP_API_KEY = os.getenv("SERP_API_KEY")
print("SERPAPI_API_KEY:", SERP_API_KEY)

def get_online_context(query):
    params = {
        "engine": "google",
        "q": query+":vku",
        "api_key": SERP_API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    
    # Extract snippets and combine them
    context = ""
    for result in organic_results:
        if "snippet" in result:
            context += result["snippet"] + " "
    
    return context.strip()

# Example usage
# if __name__ == "__main__":
#     query = 'who is Ho Chi Minh?'
#     context = get_online_context(query)
#     print("Retrieved context:")
#     print(context)
