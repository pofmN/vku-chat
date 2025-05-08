import getpass
import os
import dotenv
import requests
from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
else:
    os.environ["GOOGLE_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    print("Google API key loaded from .env file.")


prompt = [
        ("system", "You are a helpful assistant."),
        ("user", "What is the capital of France?"),
    ]




llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None
)
def dispatch_intent(intent: str, user_input:str):
    if intent == "research_university":
        research_prompt = PromptTemplate.from_template(
            """You are a helpful assistant providing information about universities.
            Answer the user's question based on the following input: "{query}"
            """
        )
        retrieval_docs = get_relevant_chunks(user_input)
        query = user_input + " " + retrieval_docs
        research_chain = research_prompt | llm | StrOutputParser()
        return {"output": [{"text": research_chain.invoke({"query": query})}]}
    elif intent == "write_email":
        return generate_university_email(user_input)

# query = "What is the capital of France?"
# chain = prompt | llm
response = llm.invoke(prompt)
print("Response from Google Gemini API:", response)