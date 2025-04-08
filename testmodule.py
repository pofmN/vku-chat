
import streamlit as st

from storage import get_relevant_chunks
from sentence_transformers import SentenceTransformer
from get_context_online import get_online_context


#st.session_state.embedding_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')


# text1 = "Làm thế nào Đại học Bách khoa Hà Nội thu hút sinh viên quốc tế?",
# text2 = """Đại học Bách khoa Hà Nội đã phát triển các chương trình đào tạo bằng tiếng Anh để làm cho việc học tại đây dễ dàng hơn cho sinh viên quốc tế,
#     Môi trường học tập đa dạng và sự hỗ trợ đầy đủ cho sinh viên quốc tế tại Đại học Bách khoa Hà Nội giúp họ thích nghi nhanh chóng,
#     Hà Nội có khí hậu mát mẻ vào mùa thu,
#     Các món ăn ở Hà Nội rất ngon và đa dạng."""

# embeding1 = model.encode(text1)
# embeding2 = model.encode(text2)
# similiarity = model.similarity(embeding1, embeding2)
# print(similiarity)

        
query = "Chương trình học bổng ở VKU?"
relevant_chunks = get_relevant_chunks(query)
#online_context = get_online_context(query)
print(relevant_chunks)
# print('context online is: ')
# print(online_context)
