import streamlit as st
import logging
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from models import initialize_gemini

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
        print("Complete prompt is: " + enhanced_prompt)
        
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
            model = initialize_gemini()
            response = model.generate_content(enhanced_prompt, safety_settings=None)
            return response.text
        else:
            logging.error(f"Error generating response: {str(e)}")
            raise