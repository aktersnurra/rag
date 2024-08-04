from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger as log
from tqdm import tqdm

from rag.generator.prompt import Prompt
from rag.model import Rag
from rag.retriever.retriever import Retriever


def configure_logging(verbose: int):
    match verbose:
        case 1:
            level = "INFO"
        case 2:
            level = "DEBUG"
        case 3:
            level = "TRACE"
        case _:
            level = "ERROR"
    log.remove()
    log.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=level)


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "-d",
    "--directory",
    help="The full path to the root directory containing pdfs to upload",
    type=click.Path(exists=True),
    default=None,
)
@click.option("-v", "--verbose", count=True)
def upload(directory: str, verbose: int):
    configure_logging(verbose)
    log.info(f"Uploading pfs found in directory {directory}...")
    retriever = Retriever()
    pdfs = Path(directory).glob("**/*.pdf")
    for path in tqdm(list(pdfs)):
        retriever.add_pdf(path=path)


@click.command()
@click.option(
    "-c",
    "--client",
    type=click.Choice(["local", "cohere"], case_sensitive=False),
    default="local",
    help="Generator and rerank model",
)
@click.option("-v", "--verbose", count=True)
def rag(client: str, verbose: int):
    configure_logging(verbose)
    rag = Rag(client)
    while True:
        query = input("Query: ")
        documents = rag.retrieve(query)
        prompt = Prompt(query, documents, client)
        print("Answer: ")
        response = ""
        for chunk in rag.generate(prompt):
            print(chunk, end="", flush=True)
            response += chunk

        rag.add_message(rag.bot, response)

        show_context = input("Display context? [y/n] ").lower() == "y"
        print("\n\n")
        if show_context:
            for i, doc in enumerate(prompt.documents):
                print(f"### Document {i}")
                print(f"**Title: {doc.title}**")
                print(doc.text)
                print("---")


@click.command()
@click.confirmation_option(prompt="Are you sure you want to drop the db?")
def drop():
    log.debug("Deleting all data...")
    retriever = Retriever()
    doc_db = retriever.doc_db
    doc_db.delete_all()
    vec_db = retriever.vec_db
    vec_db.delete_collection()


cli.add_command(rag)
cli.add_command(upload)
cli.add_command(drop)

if __name__ == "__main__":
    load_dotenv()
    cli()
