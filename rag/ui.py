import streamlit as st


# from loguru import logger as log
# from rag.rag import RAG

def upload_pdfs():
    files = st.file_uploader(
        "Choose pdfs to add to the knowledge base",
        type="pdf",
        accept_multiple_files=True,
    )
    for file in files:
        bytes = file.read()
        st.write(bytes)


if __name__ == "__main__":
    st.header("RAG-UI")
    upload_pdfs()
