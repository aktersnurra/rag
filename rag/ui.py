import streamlit as st

from langchain_community.document_loaders.blob_loaders import Blob

try:
    from rag.rag import RAG
    from rag.llm.ollama_generator import Prompt
except ModuleNotFoundError:
    from rag import RAG
    from llm.ollama_generator import Prompt

rag = RAG()


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
            rag.add_pdf_from_blob(blob, source)


if __name__ == "__main__":
    ss = st.session_state
    st.header("RAG-UI")

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
            documents = rag.search(query)

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
            st.write_stream(rag.retrieve(prompt))

