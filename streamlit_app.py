import streamlit as st
import openai
from src.rag.llamaindex_rag_impl import LlamaIndexRAGImpl

# Initialize OpenAI API key
openai.api_key = st.secrets.openai_key

st.header("Chat with the Streamlit docs 💬 📚")

# Initialize message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
    ]

rag_type = st.sidebar.selectbox("Choose RAG Implementation", ["LlamaIndex", "LangChain"])

@st.cache_resource(show_spinner=False)
def initialize_rag():
    if rag_type == "LlamaIndex":
        rag_impl = LlamaIndexRAGImpl(input_dir="./data")
    else:
        # TODO
        pass
    
    rag_impl.initialize()
    return rag_impl

rag_impl = initialize_rag()

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
            response = rag_impl.query(prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)