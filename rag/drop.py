import click
from dotenv import load_dotenv
from loguru import logger as log

from rag.retriever.retriever import Retriever


def drop():
    log.debug("Dropping documents")
    retriever = Retriever()
    doc_db = retriever.doc_db
    doc_db.delete_all()
    vec_db = retriever.vec_db
    vec_db.delete_collection()


@click.confirmation_option(prompt="Are you sure you want to drop the db?")
def main():
    drop()


if __name__ == "__main__":
    load_dotenv()
    main()
