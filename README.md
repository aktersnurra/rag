# Retrieval Augmented Generation

RAG with ollama (and optionally cohere) and qdrant. This is basically a glorified 
ctrl+f.

## Usage

### Setup

#### 1. Environment Variables

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

### 2. Ollama

Make sure ollama is running:

```sh
ollama serve
```

Download the encoder and generator models with ollama:

```sh
ollama pull $GENERATOR_MODEL
ollama pull $ENCODER_MODEL
```

### 3. Qdrant

Qdrant is used to store the embeddings of the chunks from the documents.

Download and run qdrant.

### 4. Postgres

Postgres is used to save hashes of the document to prevent documents from
being added to the vector db more than ones.

Download and run qdrant.

### 5. Cohere

Get an API from their website.

### 6. Running

#### 6.1 Prerequisites

##### 6.2 Python Environment

Activate the poetry shell:

```sh
poetry shell
```

#### 6.3 CLI

Run the cli with:

```sh
python rag/cli.py
```

#### 6.4 UI

Run the web app with streamlit:

```sh
streamlit run rag/ui.py
```

#### 6.5 Upload Multiple Documents

tbc

### Notes

Yes, it is inefficient/dumb to use ollama when you can just load the models with python
in the same process.

### Inspiration

I took some inspiration from these tutorials:

[rag-openai-qdrant](https://colab.research.google.com/github/qdrant/examples/blob/master/rag-openai-qdrant/rag-openai-qdrant.ipynb)

[building-rag-application-using-langchain-openai-faiss](https://medium.com/@solidokishore/building-rag-application-using-langchain-openai-faiss-3b2af23d98ba)

[knowledge_gpt](https://github.com/mmz-001/knowledge_gpt)
