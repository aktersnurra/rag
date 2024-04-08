# Retrieval Augmented Generation

RAG with ollama and qdrant.

## Usage

### Setup

#### Environment Variables

Create a .env file or set the following parameters:

```.env
CHUNK_SIZE = <CHUNK_SIZE>
CHUNK_OVERLAP = <CHUNK_OVERLAP>

ENCODER_MODEL = <ENCODER_MODEL>
EMBEDDING_DIM = <EMBEDDING_DIM>

GENERATOR_MODEL = <GENERATOR_MODEL>

DOCUMENT_DB_NAME = <DOCUMENT_DB_NAME>
DOCUMENT_DB_USER = <DOCUMENT_DB_USER>

QDRANT_URL = <QDRANT_URL>
QDRANT_COLLECTION_NAME = <QDRANT_COLLECTION_NAME>

COHERE_API_KEY = <COHERE_API_KEY> # OPTIONAL
```

### Ollama

Make sure ollama is running:

```sh
ollama serve
```

Download the encoder and generator models with ollama:

```sh
ollama pull $GENERATOR_MODEL
ollama pull $ENCODER_MODEL
```

### Qdrant

Qdrant will is used to store the embeddings of the chunks from the documents.

Download and run qdrant.

### Postgres

Postgres is used to save hashes of the document chunks to prevent document chunks from
being added to the vector db more than ones.

Download and run qdrant.

### Cohere

Get an API from their website.

### Running

#### Prerequisites

##### Python Environment

Activate the poetry shell:

```sh
poetry shell
```

#### CLI

```sh
python rag/cli.py
```

#### UI

Run the web app with streamlit:

```sh
streamlit run rag/ui.py
```

### Notes

Yes, it is inefficient/dumb to use ollama when you can just load the models with python
in the same process.

### Inspiration

I took some inspiration from these tutorials.

[rag-openai-qdrant](https://colab.research.google.com/github/qdrant/examples/blob/master/rag-openai-qdrant/rag-openai-qdrant.ipynb)

[building-rag-application-using-langchain-openai-faiss](https://medium.com/@solidokishore/building-rag-application-using-langchain-openai-faiss-3b2af23d98ba)

[knowledge_gpt](https://github.com/mmz-001/knowledge_gpt)
