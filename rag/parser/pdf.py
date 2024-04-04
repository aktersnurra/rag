from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256


def parser(filepath: Path):
    content = PyPDFLoader(filepath).load()
    print(content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(content)
    return chunks


# TODO: add parser for bytearray
