# streamlit_app.py
import streamlit as st
import openai
from src.rag import RAGSystem

# Initialize OpenAI API key
openai.api_key = st.secrets.openai_key

st.header("Chat with the Streamlit docs 💬 📚")

# Initialize message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

# Initialize RAG System
@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight!"):
        rag_system = RAGSystem(input_dir="./data")
        rag_system.initialize()
        return rag_system

rag_system = initialize_rag_system()

# Prompt for user input and display message history
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Pass query to RAG system and display response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_system.query(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)