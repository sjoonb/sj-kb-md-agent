import streamlit as st
import openai
from src.rag.llm_retriever_rag_impl import LlmRetrieverRAGImpl

st.header("마이데이터 에이전트 🤖")

# Initialize message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "마이데이터 공식 문서를 기반으로, 정책과 기술 사양 등을 답변할 수 있습니다.",
        }
    ]

rag_impl = LlmRetrieverRAGImpl()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your message?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("AI is thinking..."):
            response = rag_impl.query(prompt)
        
        response_placeholder.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})