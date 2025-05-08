import os
import sys
import getpass
from dotenv import load_dotenv

# Add parent directory to sys.path to allow imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules using absolute imports
from utils import clean_text
from storage import get_relevant_chunks
from get_context_online import get_online_context
from get_context_online_2 import get_tavily_response
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Debug: Print if we found the key
if os.environ.get("GOOGLE_API_KEY"):
    api_key = os.environ.get("GOOGLE_API_KEY")
    # Remove any quotes that might be in the API key
    api_key = api_key.strip('"\'')
    os.environ["GOOGLE_API_KEY"] = api_key
    print(f"Google API key loaded successfully (starts with {api_key[:5]}...)")
else:
    print("No API key found in environment variables. Prompting for input...")
    api_key = getpass.getpass("Enter your Google AI API key: ")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("API key set manually.")

os.environ["GOOGLE_API_KEY"] = "AIzaSyCLzMyymhb2UF7vc2lKswSFgj1fVYw3EcY"
# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None
)

# Create relevance checking prompt
research_prompt = PromptTemplate.from_template(
    """Đánh giá sự liên quan giữa "{query}" từ người dùng và ngữ cảnh sau:
    "{context}" sau đó trả về kết quả là 1 trong 2 giá trị sau:
    'relevant' hoặc 'not_relevant'.
    """
)
research_chain = research_prompt | llm | StrOutputParser()

def test_relevance_detection(query):
    print(f"\n--- Testing relevance detection for: '{query}' ---")
    
    # Get chunks from database
    print("Getting chunks from database...")
    relevant_chunks = get_relevant_chunks(query)
    cleaned_chunks = [clean_text(chunk) for chunk in relevant_chunks]
    context = " ".join(cleaned_chunks)
    print(f"Retrieved {len(relevant_chunks)} chunks from database.")
    
    # Check relevance
    print("Evaluating relevance...")
    research_result = research_chain.invoke({"query": query, "context": context}).strip()
    print(f"Relevance result: {research_result}")
    
    # If not relevant, search online
    if research_result == "not_relevant":
        print("No relevant context found in database.")
        print("Searching online for context...")
        online_context_2 = get_tavily_response(query)
        #online_context_1 = get_online_context(query)
        combined_context =  online_context_2
        #combined_context = clean_text(combined_context)
        print(f"Combined online context length: {len(combined_context.split())} words")
        print(f"Retrieved {len(combined_context.split())} words from online sources.")
        # You could analyze the online content here if needed
        print('Online context:', combined_context)
        return combined_context
    else:
        print("Relevant context found in database.")
        return context

if __name__ == "__main__":
    # Test with queries that should find relevant info in database
    test_queries_likely_in_db = [
        "Ở vku đào tạo những ngành gì?",
    ]
    
    # Test with queries that likely need online search
    test_queries_likely_online = [
        "Có chương trình trao đổi sinh viên quốc tế không?",
        "Ông Nguyễn Thanh Bình là ai?"
    ]
    
    # Run tests
    print("=== Testing queries likely in database ===")
    for query in test_queries_likely_in_db:
        context = test_relevance_detection(query)
        print(f"Context length: {len(context.split())} words")
    
    print("\n=== Testing queries likely needing online search ===")
    for query in test_queries_likely_online:
        context = test_relevance_detection(query)
        print(f"Context length: {len(context.split())} words")