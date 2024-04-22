from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders import Blob
from loguru import logger as log

from rag.generator import MODELS, get_generator
from rag.generator.prompt import Prompt
from rag.retriever.retriever import Retriever
from rag.retriever.vector import Document


class Cohere(Enum):
    USER = "USER"
    BOT = "CHATBOT"


class Ollama(Enum):
    USER = "user"
    BOT = "assistant"


@dataclass
class Message:
    role: str
    message: str

    def as_dict(self, client: str) -> Dict[str, str]:
        if client == "cohere":
            return {"role": self.role, "message": self.message}
        else:
            return {"role": self.role, "content": self.message}


def set_chat_users():
    log.debug("Setting user and bot value")
    ss = st.session_state
    if ss.generator == "cohere":
        ss.user = Cohere.USER.value
        ss.bot = Cohere.BOT.value
    else:
        ss.user = Ollama.USER.value
        ss.bot = Ollama.BOT.value


@st.cache_resource
def load_retriever():
    log.debug("Loading retriever model")
    st.session_state.retriever = Retriever()


@st.cache_resource
def load_generator(client: str):
    log.debug("Loading generator model")
    st.session_state.generator = get_generator(client)
    set_chat_users()


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

    retriever = ss.retriever
    generator = ss.generator

    documents = retriever.retrieve(query, limit=15)
    prompt = Prompt(query, documents)

    with st.chat_message(ss.bot):
        response = st.write_stream(generator.generate(prompt))

    display_context(documents)
    store_chat(query, response, documents)


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
        st.header("Grouding")
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

        st.header("Generative Model")
        st.markdown("Select the model that will be used for generating the answer.")
        st.selectbox("Generative Model", key="client", options=MODELS)
        load_generator(st.session_state.client)


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
    load_retriever()
    sidebar()
    page()
