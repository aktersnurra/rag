import streamlit as st

from langchain_community.document_loaders.blob_loaders import Blob

try:
    from rag.rag import RAG
except ModuleNotFoundError:
    from rag import RAG

rag = RAG()


def upload_pdfs():
    files = st.file_uploader(
        "Choose pdfs to add to the knowledge base",
        type="pdf",
        accept_multiple_files=True,
    )
    for file in files:
        blob = Blob.from_data(file.read())
        rag.add_pdf_from_blob(blob)


if __name__ == "__main__":
    st.header("RAG-UI")

    upload_pdfs()
    query = st.text_area(
        "query",
        key="query",
        height=100,
        placeholder="Enter query here",
        help="",
        label_visibility="collapsed",
        disabled=False,
    )

    (result_column, context_column) = st.columns(2)

    if query:
        response = rag.retrive(query)

        with result_column:
            st.markdown("### Answer")
            st.markdown(response.answer)

        with context_column:
            st.markdown("### Context")
            for c in response.context:
                st.markdown(c)
                st.markdown("---")
