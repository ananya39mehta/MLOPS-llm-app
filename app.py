import streamlit as st
import os
import dotenv
import uuid

# Check if it's Linux (Streamlit Cloud fix for sqlite3)
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    load_doc_to_db, 
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()

# Only Azure model list
MODELS = ["azure-openai/gpt-35-turbo"]

st.set_page_config(
    page_title="RAG LLM app", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i>Azure RAG LLM App</i> 🤖💬</h2>""")

# --- Session State ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# --- Sidebar ---
with st.sidebar:
    st.divider()

    # Show model selector (only Azure models)
    st.selectbox(
        "🤖 Select a Model", 
        options=MODELS,
        key="model",
    )

    cols0 = st.columns(2)
    with cols0[0]:
        is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)
        st.toggle(
            "Use RAG", 
            value=is_vector_db_loaded, 
            key="use_rag", 
            disabled=not is_vector_db_loaded,
        )

    with cols0[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

# --- Main Chat ---
llm_stream = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
    openai_api_version="2024-02-15-preview",
    model_name=st.session_state.model.split("/")[-1],
    openai_api_key=os.getenv("AZ_OPENAI_API_KEY"),
    openai_api_type="azure",
    temperature=0.3,
    streaming=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, [HumanMessage(content=prompt)]))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, [HumanMessage(content=prompt)]))

with st.sidebar:
    st.divider()
    st.video("https://youtu.be/abMwFViFFhI")
    st.write("📋[Medium Blog](https://medium.com/@enricdomingo/program-a-rag-llm-chat-app-with-langchain-streamlit-o1-gtp-4o-and-claude-3-5-529f0f164a5e)")
    st.write("📋[GitHub Repo](https://github.com/enricd/rag_llm_app)")
