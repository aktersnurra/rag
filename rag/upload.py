from pathlib import Path

import click
from dotenv import load_dotenv
from loguru import logger as log
from tqdm import tqdm

from rag.retriever.retriever import Retriever

log.remove()
log.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


@click.command()
@click.option(
    "-d",
    "--directory",
    help="The full path to the root directory containing pdfs to upload",
    type=click.Path(exists=True),
)
def main(directory: str):
    log.info(f"Uploading pfs found in directory {directory}...")
    retriever = Retriever()
    pdfs = Path(directory).glob("**/*.pdf")
    for path in tqdm(pdfs):
        retriever.add_pdf(path=path)


if __name__ == "__main__":
    load_dotenv()
    main()
