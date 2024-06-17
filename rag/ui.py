from dataclasses import dataclass
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders import Blob
from loguru import logger as log

from rag.generator import MODELS, get_generator
from rag.generator.prompt import Prompt
from rag.retriever.rerank import get_reranker
from rag.retriever.retriever import Retriever
from rag.retriever.vector import Document


def set_chat_users():
    log.debug("Setting user and bot value")
    ss = st.session_state
    ss.user = "user"
    ss.bot = "assistant"


def load_generator(model: str):
    log.debug("Loading rag")
    st.session_state.rag = get_generator(model)


def load_reranker(model: str):
    log.debug("Loading reranker model")
    st.session_state.reranker = get_reranker(model)


@st.cache_data(show_spinner=False)
def upload(files):
    retriever = st.session_state.retriever
    with st.spinner("Uploading documents..."):
        for file in files:
            source = file.name
            blob = Blob.from_data(file.read())
            retriever.add_pdf(blob=blob, source=source)


def display_context(documents: List[Document]):
    with st.popover("See Context"):
        for i, doc in enumerate(documents):
            st.markdown(f"### Document {i}")
            st.markdown(f"**Title: {doc.title}**")
            st.markdown(doc.text)
            st.markdown("---")


def display_chat():
    ss = st.session_state
    for msg in ss.chat:
        if isinstance(msg, list):
            display_context(msg)
        else:
            st.chat_message(msg.role).write(msg.message)


def generate_chat(query: str):
    ss = st.session_state

    with st.chat_message(ss.user):
        st.write(query)

    rag = ss.rag
    documents = rag.retrieve(query)
    Prompt(query, documents, self.client)
    with st.chat_message(ss.bot):
        response = st.write_stream(rag.generate(query))

    display_context(prompt.documents)
    store_chat(query, response, prompt.documents)


def store_chat(query: str, response: str, documents: List[Document]):
    log.debug("Storing chat")
    ss = st.session_state
    query = Message(role=ss.user, message=query)
    response = Message(role=ss.bot, message=response)
    ss.chat.append(query)
    ss.chat.append(response)
    ss.chat.append(documents)


def sidebar():
    with st.sidebar:
        st.header("Grounding")
        st.markdown(
            (
                "These files will be uploaded to the knowledge base and used "
                "as groudning if they are relevant to the question."
            )
        )

        files = st.file_uploader(
            "Choose pdfs to add to the knowledge base",
            type="pdf",
            accept_multiple_files=True,
        )

        upload(files)

        st.header("Model")
        st.markdown(
            "Select the model that will be used for reranking and generating the answer."
        )
        st.selectbox("Model", key="model", options=MODELS)
        load_generator(st.session_state.model)
        load_reranker(st.session_state.model)


def page():
    ss = st.session_state
    if "chat" not in st.session_state:
        ss.chat = []

    display_chat()

    query = st.chat_input("Enter query here")
    if query:
        generate_chat(query)


if __name__ == "__main__":
    load_dotenv()
    st.title("Retrieval Augmented Generation")
    set_chat_users()
    load_retriever()
    sidebar()
    page()
