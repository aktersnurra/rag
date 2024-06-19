from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders import Blob
from loguru import logger as log

from rag.generator import MODELS
from rag.generator.prompt import Prompt
from rag.message import Message
from rag.model import Rag
from rag.retriever.vector import Document


def set_chat_users():
    log.debug("Setting user and bot value")
    ss = st.session_state
    ss.user = "user"
    ss.bot = "assistant"

@st.cache_resource
def load_rag():
    log.debug("Loading Rag...")
    st.session_state.rag = Rag()


@st.cache_resource
def set_client(client: str):
    log.debug("Setting client...")
    rag = st.session_state.rag
    rag.set_client(client)


@st.cache_data(show_spinner=False)
def upload(files):
    rag = st.session_state.rag
    with st.spinner("Uploading documents..."):
        for file in files:
            source = file.name
            blob = Blob.from_data(file.read())
            rag.retriever.add_pdf(blob=blob, source=source)


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
            st.chat_message(msg.role).write(msg.content)


def generate_chat(query: str):
    ss = st.session_state

    with st.chat_message(ss.user):
        st.write(query)

    rag = ss.rag
    documents = rag.retrieve(query)
    prompt = Prompt(query, documents, ss.model)
    with st.chat_message(ss.bot):
        response = st.write_stream(rag.generate(prompt))

    rag.add_message(rag.bot, response)

    display_context(prompt.documents)
    store_chat(prompt, response)


def store_chat(prompt: Prompt, response: str):
    log.debug("Storing chat")
    ss = st.session_state
    query = Message(ss.user, prompt.query, ss.model)
    response = Message(ss.bot, response, ss.model)
    ss.chat.append(query)
    ss.chat.append(response)
    ss.chat.append(prompt.documents)


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
        set_client(st.session_state.model)


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
    load_rag()
    sidebar()
    page()
