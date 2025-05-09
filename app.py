import streamlit as st
import logging
from ui import apply_custom_styles, create_header, create_sidebar, create_tabs, create_footer
from models import load_embedding_models, initialize_gemini
from document_processor import compute_file_hash, process_uploaded_file
from response_generator import generate_response
from storage import get_relevant_chunks
from utils import clean_text
from intent_routing import detect_intent, dispatch_intent
from get_context_online import get_online_context

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Apply custom styling
    apply_custom_styles()
    
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
    
    # Load models
    try:
        st.session_state.embedding_model = load_embedding_models()
        model = initialize_gemini()
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
    
    # Create main application layout
    create_header()
    create_sidebar()
    tab1, tab2, tab3 = create_tabs()
    
    # Chat tab
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

            # Routing promt from user
            
            
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
                    #context = ""
                    # Try to get context from database
                    # try:
                    #     relevant_chunks = get_relevant_chunks(prompt)
                    #     if relevant_chunks:
                    #         cleaned_chunks = [clean_text(chunk) for chunk in relevant_chunks]
                    #         context = " ".join(cleaned_chunks)
                    # except Exception as e:
                    #     logging.error(f"Database search error: {str(e)}")
                    
                    # # If no context and internet is enabled, try online search
                    # if not context and st.session_state.use_internet:
                    #     try:
                    #         context = get_online_context(prompt)
                    #     except Exception as e:
                    #         logging.error(f"Online search error: {str(e)}")
                    
                    # if not context:
                    #     context = "No specific context available. Providing a general response."
                    
                    # # Generate response
                    # print("Context from database is: " + context)
                    # Detect intent and dispatch accordingly the response
                    intent = detect_intent(prompt)
                    assistant_response = dispatch_intent(intent, prompt)
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
    
    # Documents tab
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
            process_uploaded_file(uploaded_file, upload_col2)
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
    
    # Help tab
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
    
    # Add footer
    create_footer()

if __name__ == "__main__":
    main()