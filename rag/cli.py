from pathlib import Path

from dotenv import load_dotenv

from rag.generator import MODELS, get_generator
from rag.generator.prompt import Prompt
from rag.retriever.retriever import Retriever

if __name__ == "__main__":
    load_dotenv()
    retriever = Retriever()

    print("\n\nRetrieval Augmented Generation\n")
    model = input(f"Enter model ({MODELS}):")

    while True:
        choice = input("1. Add pdf from path\n2. Enter a query\n")
        match choice:
            case "1":
                path = input("Enter the path to the pdf: ")
                path = Path(path)
                retriever.add_pdf(path=path)
            case "2":
                query = input("Enter your query: ")
                if query:
                    generator = get_generator(model)
                    documents = retriever.retrieve(query)
                    prompt = Prompt(query, documents)
                    print("Answer: \n")
                    for chunk in generator.generate(prompt):
                        print(chunk, end="", flush=True)
            case _:
                print("Invalid option!")
