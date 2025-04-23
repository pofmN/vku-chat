import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to the application"""
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

def create_header():
    """Create the application header"""
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("📚 Document Q&A VKU Assistant")
    st.markdown("Upload your document and get AI-powered answers to your questions")
    st.markdown('</div>', unsafe_allow_html=True)

def create_sidebar():
    """Create the application sidebar"""
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

def create_tabs():
    """Create application tabs"""
    return st.tabs(["💬 Chat", "📑 Documents", "ℹ️ Help"])

def create_footer():
    """Create the application footer"""
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