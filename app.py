import streamlit as st
import re
from streamlit_chat import message

st.set_page_config(
    page_title="Document Q&A VKU Assistant",
    page_icon="🤖",
    layout="wide"
)
    
# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "Vietnamese"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "embeding_model" not in st.session_state:
    st.session_state.embedding_model = "SentenceTransformer('keepitreal/vietnamese-sbert')"

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import docx
import json 
from storage import store_document_chunks, get_relevant_chunks
from config.secretKey import GEMINI_API_KEY
import io
import hashlib
from get_context_online import get_online_context
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

# Page configuration


def compute_file_hash(file_content):
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

@st.cache_resource
def load_embedding_models():
    """Load and cache embedding models"""
    #st.session_state.embedding_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
    st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
    # try:
    #     return {
    #         "English": SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'),  # Changed to SentenceTransformer
    #         "Vietnamese": SentenceTransformer('keepitreal/vietnamese-sbert')  # Changed to SentenceTransformer
    #     }
    # except Exception as e:
    #     st.error(f"Error loading embedding models: {str(e)}")
    #     return None

@st.cache_resource
def initialize_gemini():
    """Initialize and cache Gemini client"""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-2.0-flash-lite')

# Load models
embedding_models = load_embedding_models()
model = initialize_gemini()

# Update the model loading section
# if embedding_models:
#     st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
# else:
#     st.error("Failed to load embedding models")

def clean_text(text: str) -> str:
    """Clean text by removing unnecessary characters and formatting."""
    # Remove special characters and unnecessary whitespace
    cleaned = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with single space
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)  # Remove numbered lists (e.g., "1. ", "2. ")
    cleaned = re.sub(r'[^\w\s.,?!-:]', ' ', cleaned)  # Remove special characters except basic punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Clean up any resulting multiple spaces
    return cleaned.strip()

import json
import io
import pdfplumber
import docx
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)

def clean_text(text):
    """Helper function to clean text (e.g., remove extra spaces, special characters)"""
    return text.strip()

def clean_text(text):
    """Clean text by removing unwanted characters and extra spaces"""
    # Remove special characters like {}, (), [], etc.
    text = re.sub(r'[{}()\[\],.]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_document(document):
    """Split document into chunks with caching"""
    try:
        # Validate input
        if not document or not hasattr(document, 'name'):
            raise ValueError("Invalid document object")

        # Extract file extension
        file_extension = document.name.split('.')[-1].lower()
        allowed_extensions = ['txt', 'pdf', 'docx', 'json']
        if file_extension not in allowed_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")

        text = ""
        
        # Handle .txt files
        if file_extension == 'txt':
            for encoding in ['utf-8', 'latin-1', 'ascii']:
                try:
                    text = document.getvalue().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                raise ValueError("Failed to decode the text file with supported encodings")
            
        # Handle .pdf files
        elif file_extension == 'pdf':
            with pdfplumber.open(io.BytesIO(document.getvalue())) as pdf:
                text = " ".join([page.extract_text() or "" for page in pdf.pages])
            if not text:
                raise ValueError("No text could be extracted from the PDF")
            
        # Handle .docx files
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(document.getvalue()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            if not text:
                raise ValueError("No text could be extracted from the DOCX file")
            
        # Handle .json files
        elif file_extension == 'json':
            try:
                json_data = json.loads(document.getvalue().decode('utf-8'))
                # Convert JSON to string representation
                text = json.dumps(json_data, indent=2)
                # Alternative: extract only values if you don't want the structure
                # text = " ".join(str(value) for value in json_data.values() if isinstance(value, (str, int, float, bool)))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {str(e)}")
        
        # Validate extracted text
        if not text:
            raise ValueError("No text could be extracted from the document")
        
        # Clean the text before splitting
        text = clean_text(text)
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Clean chunks before returning
        return [clean_text(chunk) for chunk in chunks]
    
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        return []
# # UI Components
# with st.sidebar:
#     st.title("Configuration")
    
#     Language selection
#     new_language = st.selectbox(
#         "Choose Language",
#         ["English", "Vietnamese"]
#     )
    
#     # Handle language change
#     if new_language != st.session_state.language:
#         st.session_state.language = new_language
#         st.session_state.embedding_model = embedding_models[new_language]
#         # Clear processed files if language changes
#         st.session_state.processed_files = set()
#         st.rerun()
#     else:
#         st.session_state.embedding_model = embedding_models[st.session_state.language]
    
#     st.markdown("---")
#     st.subheader("Model Information")
#     st.info("""
#     - English: MiniLM-L6-v2
#     - Vietnamese: vietnamese-sbert
#     - LLM: llama3.2Q80, llama3.2Q5KM
#     """)

# st.title("📚 Document Q&A Assistant")

# File upload section
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["txt", "pdf", "docx"],
        help="Supported formats: TXT, PDF, DOCX"
    )

# Process uploaded file
if uploaded_file is not None:
    # Compute file hash
    file_content = uploaded_file.getvalue()
    current_hash = compute_file_hash(file_content)
    
    # Check if file needs processing
    if current_hash != st.session_state.current_file_hash:
        with st.spinner("Processing new document..."):
            chunks = split_document(uploaded_file)
            if chunks:
                try:
                    store_document_chunks(chunks)
                    st.session_state.current_file_hash = current_hash   
                    st.session_state.processed_files.add(current_hash)
                    st.success(f"✨ Document processed and ready for questions using {st.session_state.language} models")
                except Exception as e:
                    st.error(f"Error storing embeddings: {str(e)}")
    else:
        st.success("✨ Document already processed and ready for questions!")

    with col2:
        st.success("✅ File uploaded successfully!")
        st.info(f"📄 Filename: {uploaded_file.name}")
else:
    with col2:
        st.warning("⚠️ Please upload a document to begin")

with col2:
    st.session_state.use_internet = st.checkbox(
        "Use Internet for Online Search",
        value=False,
        key="use_internet_toggle"
    )

# Chat interface
st.markdown("---")
st.subheader("💬 Chat Interface")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input and response generation
if prompt := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Try to get context from different sources
                context = ""
                
                # First try to get context from database if available
                try:
                    relevant_chunks = get_relevant_chunks(prompt)
                    if relevant_chunks:
                        # Clean each chunk before joining
                        cleaned_chunks = [clean_text(chunk) for chunk in relevant_chunks]
                        context = " ".join(cleaned_chunks)
                except Exception as e:
                    print(f"Database search error: {str(e)}")
                
                # If no context from database and internet is enabled, try online search
                if not context and st.session_state.use_internet:
                    try:
                        context = get_online_context(prompt)
                    except Exception as e:
                        print(f"Online search error: {str(e)}")
                
                # If still no context, proceed with just the question
                if not context:
                    context = "No specific context available. Providing a general response."

                # Prepare prompt based on language
                if st.session_state.language == "English":
                    enhanced_prompt = f"""Based on the following context (if available), provide a comprehensive, accurate, and user-focused answer. If the context is insufficient or unclear, follow the steps below to ensure a helpful response.

                        Context: {context}

                        Question: {prompt}

                        Instructions:
                        - If the context provides sufficient and relevant information, use it to craft a detailed and accurate answer, citing specific details from the context when appropriate.
                        - If the context is limited, unclear, or irrelevant:
                          1. Acknowledge the limitation (e.g., 'The provided context does not fully address this question').
                          2. Provide a general response based on common knowledge or reasonable assumptions relevant to the question (e.g., typical recruitment processes for education-related queries).
                          3. Suggest how the user can refine their question or where they might find more specific information (e.g., 'Please specify the school or check their official website').
                        - Structure the answer clearly and logically, addressing all relevant aspects of the question.
                        - Keep the response concise, easy to understand, and tailored to the user’s likely needs.
                        - Avoid speculation or inaccurate claims; if uncertain, state this explicitly.
                        - If the question has multiple parts or implications, address each one systematically.

                        Please provide your response:"""
                else:
                    enhanced_prompt = f"""Dựa trên ngữ cảnh được cung cấp, hãy đưa ra câu trả lời toàn diện, chính xác giải thích mềm mại dài dòng văn chương nhiều nhất có thể
                    và phù hợp với mục đích tư vấn tuyển sinh. đây là đường trang tuyển sinh của trường https://tuyensinh.vku.udn.vn/, hãy khuyên người dùng truy cập trang này
                    nếu ngữ cảnh chưa đầy đủ, Chú ý tới những liên kết được đề cập, sau đó hãy xử lý theo các bước được hướng dẫn dưới đây.

                        Ngữ cảnh: {context}

                        Câu hỏi: {prompt}

                        Hướng dẫn:
                        - Tập trung vào thông tin có trong ngữ cảnh được cung cấp để trả lời câu hỏi.
                        - Nếu thông tin trong ngữ cảnh đầy đủ, trích dẫn cụ thể và cấu trúc câu trả lời rõ ràng, logic, bao gồm tất cả các điểm liên quan 
                        và diễn dãi thêm nội dung để câu trả lời ý nghĩa hơn, không đề cập việc đã sử dụng ngữ cảnh để trả lờilời
                        - Nếu thông tin trong ngữ cảnh không đủ hoặc mơ hồ:
                          1. Thừa nhận rằng thông tin hiện tại từ dữ liệu hiện tạitại không đầy đủ để trả lời toàn diện.
                          2. Dựa trên kiến thức chung về tuyển sinh của riêng bạn(ví dụ: quy trình đăng ký, tiêu chí xét tuyển, lịch trình thông thường), đưa ra câu trả lời hợp lý, uyển chuyển 
                          hướng người dùng tới https://tuyensinh.vku.udn.vn/ để được tư vấn.
                          3. Đề xuất người dùng cung cấp thêm chi tiết hoặc tích vào ô sử dụng Internet để có câu trả lời chính xác hơn.
                        - Tránh đưa ra thông tin sai lệch hoặc suy đoán không có căn cứ; nếu không chắc chắn, hãy nêu rõ điều đó.
                        - Đảm bảo câu trả lời sinh động, tự nhiên, dễ hiểu, và phù hợp với nhu cầu của học sinh/sinh viên trong bối cảnh tư vấn tuyển sinh.
                        - Nếu có nhiều khía cạnh liên quan trong câu hỏi, phân tích từng khía cạnh một cách có tổ chức.

                        Vui lòng cung cấp câu trả lời của bạn một cách văn chương, dài nhất có thể :"""
                
                # Generate response
                print("PROMPT IS: " + enhanced_prompt)
                try:
                    response = model.generate_content(
                        enhanced_prompt,
                        generation_config={
                            'temperature': 0.9,
                            'top_p': 0.2,
                            'max_output_tokens': 8192,
                        },
                        safety_settings={
                            HarmCategory.HARASSMENT: HarmBlockThreshold.LOW,
                            HarmCategory.HATE_SPEECH: HarmBlockThreshold.LOW,
                            HarmCategory.SEXUALLY_EXPLICIT: HarmBlockThreshold.LOW,
                            HarmCategory.DANGEROUS_CONTENT: HarmBlockThreshold.LOW,
                        }
                    )
                except Exception as e:
                    if "HARASSMENT" in str(e):
                        # Try again with more permissive settings
                        response = model.generate_content(
                            enhanced_prompt,
                            safety_settings=None  # Disable safety settings
                        )
                
                assistant_response = response.text
                print("Answer is: "+assistant_response)
                st.markdown(assistant_response)
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Built with Streamlit • Powered by Nam-Giang</small>
    </div>
    """,
    unsafe_allow_html=True
)