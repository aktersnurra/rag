from typing import Type

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders import Blob

from rag.generator import MODELS, get_generator
from rag.generator.abstract import AbstractGenerator
from rag.generator.prompt import Prompt
from rag.retriever.retriever import Retriever


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def load_generator(model: str) -> Type[AbstractGenerator]:
    return get_generator(model)


@st.cache_data(show_spinner=False)
def upload(files):
    with st.spinner("Indexing documents..."):
        for file in files:
            source = file.name
            blob = Blob.from_data(file.read())
            retriever.add_pdf(blob=blob, source=source)


if __name__ == "__main__":
    load_dotenv()
    retriever = load_retriever()

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
        model = st.selectbox("Generative Model", options=MODELS)
        generator = load_generator(model)

    st.title("Retrieval Augmented Generation")

    with st.form(key="query"):
        query = st.text_area(
            "query",
            key="query",
            height=100,
            placeholder="Enter query here",
            help="",
            label_visibility="collapsed",
            disabled=False,
        )
        submit = st.form_submit_button("Generate")

    (result_column, context_column) = st.columns(2)

    if submit and query:
        with st.spinner("Searching for documents..."):
            documents = retriever.retrieve(query)

        prompt = Prompt(query, documents)

        with context_column:
            st.markdown("### Context")
            for i, doc in enumerate(documents):
                st.markdown(f"### Document {i}")
                st.markdown(f"**Title: {doc.title}**")
                st.markdown(doc.text)
                st.markdown("---")

        with result_column:
            st.markdown("### Answer")
            st.write_stream(generator.generate(prompt))
