import streamlit as st

from langchain_community.document_loaders.blob_loaders import Blob
from .rag import RAG

from .generator import get_generator
from .generator.prompt import Prompt

rag = RAG()

MODELS = ["ollama", "cohere"]


def upload_pdfs():
    files = st.file_uploader(
        "Choose pdfs to add to the knowledge base",
        type="pdf",
        accept_multiple_files=True,
    )

    if not files:
        return

    with st.spinner("Indexing documents..."):
        for file in files:
            source = file.name
            blob = Blob.from_data(file.read())
            rag.add_pdf(blob, source)


if __name__ == "__main__":
    ss = st.session_state
    st.header("RAG-UI")

    model = st.selectbox("Model", options=MODELS)

    upload_pdfs()

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

    (b,) = st.columns(1)
    (result_column, context_column) = st.columns(2)

    if submit:
        if not query:
            st.stop()

        query = ss.get("query", "")
        with st.spinner("Searching for documents..."):
            documents = rag.retrieve(query)

        prompt = Prompt(query, documents)

        with context_column:
            st.markdown("### Context")
            for i, doc in enumerate(documents):
                st.markdown(f"### Document {i}")
                st.markdown(f"**Title: {doc.title}**")
                st.markdown(doc.text)
                st.markdown("---")

        with result_column:
            generator = get_generator(model)
            st.markdown("### Answer")
            st.write_stream(rag.generate(generator, prompt))
