import streamlit as st
import openai
from src.rag.llm_retriever_rag_impl import LlmRetrieverRAGImpl

st.header("멀티턴 마이데이터 에이전트 🤖")

# Initialize RAG implementation
if 'rag_impl' not in st.session_state:
    st.session_state.rag_impl = LlmRetrieverRAGImpl()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "마이데이터 공식 문서를 기반으로, 정책과 기술 사양 등을 답변할 수 있습니다.",
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is your message?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Prepare conversation history for the RAG system
    conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]])

    # Get response from RAG system
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("AI is thinking..."):
            response = st.session_state.rag_impl.query(f"{conversation_history}\nuser: {prompt}")
        
        response_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
