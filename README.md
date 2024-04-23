# Retrieval Augmented Generation

RAG with ollama (and optionally cohere) and qdrant. This is basically a glorified
(bloated) `ctrl+f`.

## Usage

### Setup

#### 1. Environment Variables

Create a .env file or set the following parameters:

```.env
CHUNK_SIZE = 4096
CHUNK_OVERLAP = 256

ENCODER_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
RETRIEVER_TOP_K = 15
RETRIEVER_SCORE_THRESHOLD = 0.5

RERANK_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"
RERANK_TOP_K = 5

GENERATOR_MODEL = "dolphin-llama3"

DOCUMENT_DB_NAME = "rag"
DOCUMENT_DB_USER = "aktersnurra"

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_NAME = "knowledge-base"

COHERE_API_KEY = <COHERE_API_KEY> # OPTIONAL
COHERE_RERANK_MODEL = "rerank-english-v3.0"
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

Both databases needs to be running as well as ollama.

##### 6.1.1 Python Environment

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

### Notes

Yes, it is inefficient/dumb to use ollama when you can just load the models with python
in the same process.

### Inspiration

I took some inspiration from these tutorials:

[rag-openai-qdrant](https://colab.research.google.com/github/qdrant/examples/blob/master/rag-openai-qdrant/rag-openai-qdrant.ipynb)

[building-rag-application-using-langchain-openai-faiss](https://medium.com/@solidokishore/building-rag-application-using-langchain-openai-faiss-3b2af23d98ba)

[knowledge_gpt](https://github.com/mmz-001/knowledge_gpt)
