
import streamlit as st
from storage import get_relevant_chunks
import dotenv
import os
import getpass
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from get_context_online import get_online_context
from get_context_online_2 import get_tavily_response
from intent_routing import detect_intent, dispatch_intent
from response_generator import generate_response

#test_prompt = "Ở vku đào tạo những ngành gì?"
#test_prompt = "Học phí của trường là bao nhiêu?"
#test_prompt = "Điểm chuẩn vku 2023 là bao nhiêu?"  
#test_prompt = "Điểm chuẩn vku 2024 là bao nhiêu?"

load_dotenv()
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

#st.session_state.embedding_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')

query = ["Ông Phạm Văn Nam là ai?"]
# relevant_chunks = get_relevant_chunks(query[0])
online_context = get_online_context(query[0])
online_context_2 = get_tavily_response(query[0])
print(online_context_2)
print(online_context)
#print(relevant_chunks)


