import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders.blob_loaders import Blob

from rag.generator import MODELS, get_generator
from rag.generator.prompt import Prompt
from rag.retriever.retriever import Retriever

if __name__ == "__main__":
    load_dotenv()
    retriever = Retriever()
    ss = st.session_state
    st.header("Retrieval Augmented Generation")

    model = st.selectbox("Model", options=MODELS)

    files = st.file_uploader(
        "Choose pdfs to add to the knowledge base",
        type="pdf",
        accept_multiple_files=True,
    )

    if files:
        with st.spinner("Indexing documents..."):
            for file in files:
                source = file.name
                blob = Blob.from_data(file.read())
                retriever.add_pdf(blob=blob, source=source)

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

    if submit and model:
        if not query:
            st.stop()

        query = ss.get("query", "")
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
            generator = get_generator(model)
            st.markdown("### Answer")
            st.write_stream(generator.generate(prompt))
