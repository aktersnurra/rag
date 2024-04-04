import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def parser(filepath: Path):
    content = PyPDFLoader(filepath).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=os.environ["CHUNK_SIZE"], chunk_overlap=os.environ["CHUNK_OVERLAP"]
    )
    chunks = splitter.split_documents(content)
    return chunks


# TODO: add parser for bytearray
