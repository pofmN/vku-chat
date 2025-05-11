from typing import Literal
import os
import dotenv
from dotenv import load_dotenv
import getpass
import webbrowser
from urllib.parse import quote
from utils import clean_text
from storage import get_relevant_chunks
from get_context_online import get_online_context
from get_context_online_2 import get_tavily_response
from response_generator import generate_response
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Debug: Print if we found the key
if os.environ.get("GOOGLE_API_KEY"):
    api_key = os.environ.get("GOOGLE_API_KEY")
    print(f"Google API key loaded successfully (starts with {api_key[:5]}...)")
else:
    print("No API key found in environment variables. Prompting for input...")
    api_key = getpass.getpass("Enter your Google AI API key: ")
    os.environ["GOOGLE_API_KEY"] = api_key
    print("API key set manually.")

intent_prompt = PromptTemplate.from_template(
    """Classify the user's intent into one of the following: 
    'research_university', 'write_email', or 'normal_chatting'.
    
    User input: {user_input}
    
    Respond with only the intent label.
    """
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None
)


def detect_intent(user_input: str) -> str:
    intent_chain = intent_prompt | llm | StrOutputParser()
    return intent_chain.invoke({"user_input": user_input}).strip()
 
def dispatch_intent(intent: str, user_input:str):
    if intent == "research_university":
        context = ""
        research_prompt = PromptTemplate.from_template(
            """Đánh giá sự liên quan giữa "{query}" từ người dùng và ngữ cảnh sau:
            "{context}" sau đó trả về kết quả là 1 trong 2 giá trị sau:
            'relevant' hoặc 'not_relevant'.
            """
        )
        research_chain = research_prompt | llm | StrOutputParser()
        relevant_chunks = get_relevant_chunks(user_input)
        cleaned_chunks = [clean_text(chunk) for chunk in relevant_chunks]
        context_database = " ".join(cleaned_chunks)
        research_result = research_chain.invoke({"query": user_input, "context": context_database}).strip()
        if research_result == "relevant":
            print("Relevant context found in database.")
            context = context_database
        else:
            print("No relevant context found in database.")
            print("Searching online for context...")
            context = get_online_context(user_input) + " " + get_tavily_response(user_input)
            print("-- Online context retrieved --")
            print("complete context is: " + context)
            print("-- Online context retrieved --")
        response = generate_response(user_input, context)
        return response
    elif intent == "write_email":
        return generate_university_email(user_input)
    elif intent == "normal_chatting":
        normal_chat_prompt = PromptTemplate.from_template(
            """Bạn là một chatbot trò chuyện vui nhộn, hài hước, dí dỏm. Hãy trả lời câu hỏi sau đây từ người dùng
              "{user_input}" một cách thân thiện và vui tươi nhất có thể.
            """
        )
        normal_chat_chain = normal_chat_prompt | llm | StrOutputParser()
        response = normal_chat_chain.invoke({"user_input": user_input}).strip()
        return response
    
def generate_university_email(user_input: str):
    # Generate email content
    email_prompt = PromptTemplate.from_template(
    """Bạn là một chatbot hổ trợ tuyển sinh, hãy tạo cho tôi nội dung của một email 
     để gửi cho trường đại học dựa trên nội dung hỏi mà user muốn hỏi như sau: "{user_input}"
    """
    )
    email_chain = email_prompt | llm | StrOutputParser()
    email_text = email_chain.invoke({"user_input": user_input})
    
    # Email parameters
    recipient = "congtactuyensinh.vku.udn.vn"  # Replace with actual email address
    subject = "University Inquiry"  # You can make this dynamic based on user_input
    body = email_text
    
    # Create mailto URL
    mailto_url = f"mailto:{recipient}?subject={quote(subject)}&body={quote(body)}"
    
    # Open default email client
    webbrowser.open(mailto_url)
    
    response_text = f"Email generated and opened in your mail client:\n\nTo: {recipient}\nSubject: {subject}\n\n{email_text}"
    return response_text


