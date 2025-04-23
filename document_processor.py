import streamlit as st
import hashlib
import logging
from utils import extract_text_from_file, clean_text, split_document
from storage import store_document_chunks

def compute_file_hash(file_content):
    """Compute SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()

def process_uploaded_file(uploaded_file, status_column):
    """Process uploaded file and store chunks in database"""
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
                    
                    with status_column:
                        st.markdown('<div class="status-success">', unsafe_allow_html=True)
                        st.success(f"✨ Document processed successfully! Ready for questions using {st.session_state.language} models")
                        st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            with status_column:
                st.markdown('<div class="status-error">', unsafe_allow_html=True)
                st.error(f"Error processing document: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
            logging.error(f"Error processing document: {str(e)}")
    else:
        with status_column:
            st.markdown('<div class="status-success">', unsafe_allow_html=True)
            st.success("✨ Document already processed and ready for questions!")
            st.markdown('</div>', unsafe_allow_html=True)

    # Display file info
    with status_column:
        st.markdown('<div class="status-info">', unsafe_allow_html=True)
        st.markdown(f"**File Details:**")
        st.markdown(f"📄 **Name:** {uploaded_file.name}")
        st.markdown(f"📏 **Size:** {len(file_content) / 1024:.1f} KB")
        st.markdown(f"🧩 **Chunks:** {len(st.session_state.document_chunks)}")
        st.markdown('</div>', unsafe_allow_html=True)