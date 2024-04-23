from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv
from loguru import logger as log
from tqdm import tqdm

from rag.generator import get_generator
from rag.generator.prompt import Prompt
from rag.retriever.rerank import get_reranker
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


def rag(model: str, query: str):
    retriever = Retriever()
    generator = get_generator(model)
    reranker = get_reranker(model)
    documents = retriever.retrieve(query)
    prompt = reranker.rerank(Prompt(query, documents))
    print("Answer: ")
    for chunk in generator.generate(prompt):
        print(chunk, end="", flush=True)

    print("\n\n")
    for i, doc in enumerate(prompt.documents):
        print(f"### Document {i}")
        print(f"**Title: {doc.title}**")
        print(doc.text)
        print("---")


@click.command()
@click.option(
    "-q",
    "--query",
    prompt_required=False,
    help="The query for rag",
    prompt="Enter your query",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["local", "cohere"], case_sensitive=False),
    default="local",
    help="Generator and rerank model",
)
@click.option(
    "-d",
    "--directory",
    help="The full path to the root directory containing pdfs to upload",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "-q",
    "--query",
    prompt_required=False,
    help="The query for rag",
    prompt="Enter your query",
)
@click.option("-v", "--verbose", count=True)
def main(
    query: Optional[str],
    generator: str,
    directory: Optional[str],
    verbose: int,
):
    configure_logging(verbose)
    if directory:
        upload(directory)
    if query:
        rag(generator, query)
    # TODO: maybe add override for models


if __name__ == "__main__":
    load_dotenv()
    main()
