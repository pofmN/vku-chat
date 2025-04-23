import streamlit as st
import re
# Page configuration
st.set_page_config(
    page_title="Document Q&A VKU Assistant",
    page_icon="🤖",
    layout="wide"
)
import logging
import hashlib
import io
import json
import torch
from streamlit_chat import message
import pdfplumber
import docx
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Import custom modules
from storage import store_document_chunks, get_relevant_chunks
from config.secretKey import GEMINI_API_KEY
from get_context_online import get_online_context
from process_data import extract_text_from_file, clean_text, split_document

# Configure logging
logging.basicConfig(level=logging.INFO)


# Apply custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: yellow;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .card {
        background-color: #2037b1;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Status indicators */
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
    
    /* Chat styling */
    .stContainer {
        border-radius: 10px;
        background-color: yellow;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4b6cb7;
        color: yellow;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #182848;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #796e1d;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: yellow;
        border-bottom: 2px solid #4b6cb7;
    }
    
    /* File uploader */
    .stFileUploader > div > label {
        font-weight: bold;
    }
    
    /* Chat message styling */
    .chat-message-container {
        display: flex;
        margin: 10px 0;
        width: 100%;
    }

    .chat-message-container.user {
        justify-content: flex-end;
    }

    .chat-message-container.assistant {
        justify-content: flex-start;
    }

    .chat-message {
        padding: 10px 15px;
        border-radius: 20px;
        max-width: 80%;
        word-wrap: break-word;
    }

    .chat-message.user {
        background-color: #136a1f;
    }

    .chat-message.assistant {
        background-color: #12214a;
    }
    /* Features section */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 24px 0;
    }
    
    .feature-card {
        background-color: yellow;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .feature-icon {
        font-size: 24px;
        margin-bottom: 12px;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .badge-primary {
        background-color: #4b6cb7;
        color: yellow;
    }
    
    .badge-success {
        background-color: #28a745;
        color: yellow;
    }
    .watermark {
        text-align: center        
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "Vietnamese"
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Chat"

# Helper functions
def compute_file_hash(file_content):
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

@st.cache_resource
def load_embedding_models():
    """Load and cache embedding models"""
    return SentenceTransformer('keepitreal/vietnamese-sbert')

@st.cache_resource
def initialize_gemini():
    """Initialize and cache Gemini client"""
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-2.0-flash-lite')

# def clean_text(text):
#     """Clean text by removing unwanted characters and extra spaces"""
#     if not text:
#         return ""
        
#     # Remove special characters and standardize yellowspace
#     text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with space
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
#     text = re.sub(r'^\d+\.\s*', '', text)  # Remove numbered lists (e.g., "1. ", "2. ")
    
#     # Remove extra spaces before and after text
#     return text.strip()

# def extract_text_from_file(document):
#     """Extract text from various file formats"""
#     if not document or not hasattr(document, 'name'):
#         raise ValueError("Invalid document object")

#     file_extension = document.name.split('.')[-1].lower()
#     allowed_extensions = ['txt', 'pdf', 'docx', 'json']
    
#     if file_extension not in allowed_extensions:
#         raise ValueError(f"Unsupported file type: {file_extension}")
    
#     file_content = document.getvalue()
    
#     # Handle different file types
#     if file_extension == 'txt':
#         for encoding in ['utf-8', 'latin-1', 'ascii']:
#             try:
#                 return file_content.decode(encoding)
#             except UnicodeDecodeError:
#                 continue
#         raise ValueError("Failed to decode the text file with supported encodings")
    
#     elif file_extension == 'pdf':
#         with pdfplumber.open(io.BytesIO(file_content)) as pdf:
#             text = " ".join([page.extract_text() or "" for page in pdf.pages])
#         if not text:
#             raise ValueError("No text could be extracted from the PDF")
#         return text
    
#     elif file_extension == 'docx':
#         doc = docx.Document(io.BytesIO(file_content))
#         text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
#         if not text:
#             raise ValueError("No text could be extracted from the DOCX file")
#         return text
    
#     elif file_extension == 'json':
#         try:
#             json_data = json.loads(file_content.decode('utf-8'))
#             return json.dumps(json_data, indent=2)
#         except json.JSONDecodeError as e:
#             raise ValueError(f"Invalid JSON file: {str(e)}")

# def split_document(document):
#     """Split document into chunks with caching"""
#     try:
#         text = extract_text_from_file(document)
        
#         # Validate extracted text
#         if not text:
#             raise ValueError("No text could be extracted from the document")
        
#         # Clean the text before splitting
#         text = clean_text(text)
            
#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=850,
#             chunk_overlap=300,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         chunks = text_splitter.split_text(text)
        
#         # Clean chunks before returning
#         return [clean_text(chunk) for chunk in chunks]
    
#     except Exception as e:
#         logging.error(f"Error processing document: {str(e)}")
#         raise

def generate_enhanced_prompt(language, context, prompt):
    """Generate the enhanced prompt based on language"""
    if language == "English":
        return f"""Based on the following context (if available), provide a comprehensive, accurate, and user-focused answer. If the context is insufficient or unclear, follow the steps below to ensure a helpful response.

            Context: {context}

            Question: {prompt}

            Instructions:
            - If the context provides sufficient and relevant information, use it to craft a detailed and accurate answer, citing specific details from the context when appropriate.
            - If the context is limited, unclear, or irrelevant:
              1. Acknowledge the limitation (e.g., 'The provided context does not fully address this question').
              2. Provide a general response based on common knowledge or reasonable assumptions relevant to the question (e.g., typical recruitment processes for education-related queries).
              3. Suggest how the user can refine their question or where they might find more specific information (e.g., 'Please specify the school or check their official website').
            - Structure the answer clearly and logically, addressing all relevant aspects of the question.
            - Keep the response concise, easy to understand, and tailored to the user's likely needs.
            - Avoid speculation or inaccurate claims; if uncertain, state this explicitly.
            - If the question has multiple parts or implications, address each one systematically.

            Please provide your response:"""
    else:
        return f"""Bạn là một chatbot tư vấn tuyển sinh sử dụng kiến trúc RAG. Dựa trên ngữ cảnh được cung cấp, hãy đưa ra câu trả lời chính xác, giải thích mềm mại, chi tiết kết hợp một tí dí dỏm.
        và phù hợp với mục đích tư vấn tuyển sinh. đây là đường trang tuyển sinh của trường https://tuyensinh.vku.udn.vn/, hãy khuyên người dùng truy cập trang này
        nếu ngữ cảnh chưa đầy đủ, Chú ý tới những liên kết, con số, liên hệ được đề cập, sau đó hãy xử lý theo các bước được hướng dẫn dưới đây.

            Ngữ cảnh: {context}

            Câu hỏi: {prompt}

            Hướng dẫn:
            - Tập trung vào thông tin có trong ngữ cảnh được cung cấp để trả lời câu hỏi.
            - Nếu thông tin trong ngữ cảnh đầy đủ, trích dẫn cụ thể và cấu trúc câu trả lời rõ ràng, logic, bao gồm tất cả các điểm liên quan 
            và diễn dãi thêm nội dung để câu trả lời ý nghĩa, vui nhộn hơn, không đề cập việc đã sử dụng ngữ cảnh để trả lời
            - Nếu thông tin trong ngữ cảnh không đủ hoặc mơ hồ:
              1. Thừa nhận rằng thông tin hiện tại từ dữ liệu hiện tại không đầy đủ để trả lời toàn diện.
              2. Dựa trên kiến thức chung về tuyển sinh của riêng bạn(ví dụ: quy trình đăng ký, tiêu chí xét tuyển, lịch trình thông thường), đưa ra câu trả lời hợp lý, uyển chuyển 
              hướng người dùng tới https://tuyensinh.vku.udn.vn/ để được tư vấn.
              3. Đề xuất người dùng cung cấp thêm chi tiết hoặc tích vào ô sử dụng Internet để có câu trả lời chính xác hơn.
            - Tránh đưa ra thông tin sai lệch hoặc suy đoán không có căn cứ; nếu không chắc chắn, hãy nêu rõ điều đó.
            - Đảm bảo câu trả lời sinh động, tự nhiên, dễ hiểu, vui nhộn, hoạt ngôn và phù hợp với nhu cầu, lứa tuổi của học sinh/sinh viên trong bối cảnh tư vấn tuyển sinh.
            - Nếu có nhiều khía cạnh liên quan trong câu hỏi, phân tích từng khía cạnh một cách có tổ chức.

            Vui lòng cung cấp câu trả lời của bạn một cách văn chương, dài nhất có thể:"""

def generate_response(prompt, context):
    """Generate response using the AI model"""
    try:
        enhanced_prompt = generate_enhanced_prompt(st.session_state.language, context, prompt)
        print("Complete promt is: "+ enhanced_prompt)
        
        model = initialize_gemini()
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
        
        return response.text
    except Exception as e:
        if "HARASSMENT" in str(e):
            # Try again with more permissive settings
            response = model.generate_content(enhanced_prompt, safety_settings=None)
            return response.text
        else:
            raise

# Load models
try:
    st.session_state.embedding_model = load_embedding_models()
    model = initialize_gemini()
except Exception as e:
    st.error(f"Error initializing models: {str(e)}")

# Create main layout
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("📚 Document Q&A VKU Assistant")
st.markdown("Upload your document and get AI-powered answers to your questions")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🛠️ Settings")
    
    st.session_state.language = st.selectbox(
        "💬 Language / Ngôn ngữ",
        options=["Vietnamese", "English"],
        index=0
    )
    
    st.markdown("### 📊 Statistics")
    if st.session_state.current_file_hash:
        st.markdown('<div class="status-success">', unsafe_allow_html=True)
        st.markdown("✅ **Document status:** Processed")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">', unsafe_allow_html=True)
        st.markdown("⚠️ **Document status:** No document loaded")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🧹 Clear Data")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    if st.button("Clear Document Data"):
        st.session_state.current_file_hash = None
        st.session_state.processed_files = set()
        st.session_state.document_chunks = []
        st.success("Document data cleared!")
    
    st.markdown("### ℹ️ About")
    st.info(
        "This assistant helps you interact with your document using AI. "
        "It can answer questions based on the document content and optionally "
        "search the internet for additional information."
    )

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📑 Documents", "ℹ️ Help"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("💬 Chat with your Document")
    
    # Create a scrollable chat container
    chat_container = st.container(height=500)  # Fixed height for scrollable area
    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            # Use divs with custom CSS for user and assistant messages
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="chat-message-container user">
                        <div class="chat-message user">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="chat-message-container assistant">
                        <div class="chat-message assistant">{msg["content"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # JavaScript to auto-scroll to the bottom of the chat container
    st.markdown(
        """
        <script>
            const chatContainer = window.parent.document.querySelector('.stContainer');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
        """,
        unsafe_allow_html=True
    )
    
    # Context sources section
    context_col1, context_col2 = st.columns([3, 1])
    
    with context_col2:
        st.session_state.use_internet = st.checkbox(
            "🌐 Use Internet Search",
            value=False,
            key="use_internet_toggle",
            help="Enable internet search for additional context"
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message directly in the container
        with chat_container:
            st.markdown(
                f"""
                <div class="chat-message-container user">
                    <div class="chat-message user">{prompt}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Handle response generation
        try:
            with st.spinner("Generating answer..."):
                context = ""
                
                # Try to get context from database
                try:
                    relevant_chunks = get_relevant_chunks(prompt)
                    if relevant_chunks:
                        cleaned_chunks = [clean_text(chunk) for chunk in relevant_chunks]
                        context = " ".join(cleaned_chunks)
                except Exception as e:
                    logging.error(f"Database search error: {str(e)}")
                
                # If no context and internet is enabled, try online search
                if not context and st.session_state.use_internet:
                    try:
                        context = get_online_context(prompt)
                    except Exception as e:
                        logging.error(f"Online search error: {str(e)}")
                
                if not context:
                    context = "No specific context available. Providing a general response."
                
                # Generate response
                print("Context from database is: " + context)
                assistant_response = generate_response(prompt, context)
                
                # Add assistant message to state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_response
                })
                
                # Display assistant message
                with chat_container:
                    st.markdown(
                        f"""
                        <div class="chat-message-container assistant">
                            <div class="chat-message assistant">{assistant_response}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Document upload section
    st.subheader("📄 Document Upload")
    
    upload_col1, upload_col2 = st.columns([2, 1])
    
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=["txt", "pdf", "docx", "json"],
            help="Supported formats: TXT, PDF, DOCX, JSON"
        )
    
    # Process uploaded file
    if uploaded_file is not None:
        # Compute file hash
        file_content = uploaded_file.getvalue()
        current_hash = compute_file_hash(file_content)
        
        # Check if file needs processing
        if current_hash != st.session_state.current_file_hash:
            try:
                with st.spinner("Processing document..."):
                    chunks = split_document(uploaded_file)
                    if chunks:
                        # Store chunks in session state for display
                        st.session_state.document_chunks = chunks
                        
                        # Store chunks in database
                        store_document_chunks(chunks)
                        
                        # Update session state
                        st.session_state.current_file_hash = current_hash   
                        st.session_state.processed_files.add(current_hash)
                        
                        st.markdown('<div class="status-success">', unsafe_allow_html=True)
                        st.success(f"✨ Document processed successfully! Ready for questions using {st.session_state.language} models")
                        st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown('<div class="status-error">', unsafe_allow_html=True)
                st.error(f"Error processing document: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-success">', unsafe_allow_html=True)
            st.success("✨ Document already processed and ready for questions!")
            st.markdown('</div>', unsafe_allow_html=True)

        # Display file info
        with upload_col2:
            st.markdown('<div class="status-info">', unsafe_allow_html=True)
            st.markdown(f"**File Details:**")
            st.markdown(f"📄 **Name:** {uploaded_file.name}")
            st.markdown(f"📏 **Size:** {len(file_content) / 1024:.1f} KB")
            st.markdown(f"🧩 **Chunks:** {len(st.session_state.document_chunks)}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        with upload_col2:
            st.markdown('<div class="status-warning">', unsafe_allow_html=True)
            st.warning("⚠️ Please upload a document to begin")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display document chunks if available
    if "document_chunks" in st.session_state and st.session_state.document_chunks:
        st.markdown("### 📝 Document Chunks")
        for i, chunk in enumerate(st.session_state.document_chunks):
            with st.expander(f"Chunk {i+1}"):
                st.text(chunk)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("ℹ️ How to Use This App")
    
    st.markdown("""
    **Follow these steps to get started:**
    
    1. **Upload a Document**: Go to the Documents tab and upload your TXT, PDF, DOCX, or JSON file.
    
    2. **Ask Questions**: Switch to the Chat tab and ask questions about your document.
    
    3. **Get AI-Powered Answers**: The assistant will analyze your document and provide relevant answers.
    
    4. **Enable Internet Search**: If your document doesn't contain all the information, enable the Internet Search option for additional context.
    """)
    
    st.markdown("### ✨ Features")
    
    st.markdown('<div class="features-grid">', unsafe_allow_html=True)
    
    # Feature 1
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">📄</div>', unsafe_allow_html=True)
    st.markdown("**Document Processing**")
    st.markdown("Upload and process various document formats")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature 2
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">🤖</div>', unsafe_allow_html=True)
    st.markdown("**AI-Powered Answers**")
    st.markdown("Get intelligent responses based on document content")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature 3
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">🌐</div>', unsafe_allow_html=True)
    st.markdown("**Internet Search**")
    st.markdown("Optionally expand answers with online information")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature 4
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">🔍</div>', unsafe_allow_html=True)
    st.markdown("**Context Analysis**")
    st.markdown("Smart document chunking and retrieval")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature 5
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">🗣️</div>', unsafe_allow_html=True)
    st.markdown("**Multilingual Support**")
    st.markdown("Switch between Vietnamese and English")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature 6
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">💾</div>', unsafe_allow_html=True)
    st.markdown("**Session Management**")
    st.markdown("Persistent chat history and document processing")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="watermark">
        <p>📚 Document Q&A VKU Assistant | Built with Streamlit • Powered by Nam-Giang</p>
        <p>© 2025 - Vietnam-Korea University of Information and Communication Technology</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)