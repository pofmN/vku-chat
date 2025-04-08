from serpapi import GoogleSearch

def get_online_context(query):
    params = {
        "engine": "google",
        "q": query+":vku",
        "api_key": "91dcf285233d247f855756a7bc64eaad448f186d9e0f321a3044fd2eab278922"
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
