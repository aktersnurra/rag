from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from loguru import logger as log
from tqdm import tqdm

from rag.generator import get_generator
from rag.generator.prompt import Prompt
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


def upload(directory: str):
    log.info(f"Uploading pfs found in directory {directory}...")
    retriever = Retriever()
    pdfs = Path(directory).glob("**/*.pdf")
    for path in tqdm(list(pdfs)):
        retriever.add_pdf(path=path)


def rag(generator: str, query: str, limit):
    retriever = Retriever()
    generator = get_generator(generator)
    documents = retriever.retrieve(query, limit=limit)
    prompt = Prompt(query, documents)
    print("Answer: ")
    for chunk in generator.generate(prompt):
        print(chunk, end="", flush=True)

    print("\n\n")
    for i, doc in enumerate(documents):
        print(f"### Document {i}")
        print(f"**Title: {doc.title}**")
        print(doc.text)
        print("---")


@click.option(
    "-q",
    "--query",
    prompt_required=False,
    help="The query for rag",
    prompt="Enter your query",
)
@click.option(
    "-g",
    "--generator",
    type=click.Choice(["ollama", "cohere"], case_sensitive=False),
    default="ollama",
    help="Generator client",
)
@click.option(
    "-l",
    "--limit",
    type=click.IntRange(1, 20, clamp=True),
    default=5,
    help="Max number of documents used in grouding",
)
@click.command()
@click.option(
    "-d",
    "--directory",
    help="The full path to the root directory containing pdfs to upload",
    type=click.Path(exists=True),
    default=None,
)
@click.option("-v", "--verbose", count=True)
def main(
    query: Optional[str], generator: str, limit: int, directory: Optional[str], verbose
):
    configure_logging(verbose)
    if query:
        rag(generator, query, limit)
    elif directory:
        upload(directory)
    # TODO: truncate databases
    # TODO: maybe add override for models


if __name__ == "__main__":
    load_dotenv()
    main()
