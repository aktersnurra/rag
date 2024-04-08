from pathlib import Path


try:
    from rag.rag import RAG
except ModuleNotFoundError:
    from rag import RAG

if __name__ == "__main__":
    rag = RAG()

    while True:
        print("Retrieval Augmented Generation")
        choice = input("1. add pdf from path\n2. Enter a query\n")
        match choice:
            case "1":
                path = input("Enter the path to the pdf: ")
                path = Path(path)
                rag.add_pdf_from_path(path)
            case "2":
                query = input("Enter your query: ")
                if query:
                    result = rag.retrive(query)
                    print("Answer: \n")
                    print(result.answer + "\n")
            case _:
                print("Invalid option!")

