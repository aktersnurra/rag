# Retrieval Augmented Generation

RAG with ollama (and optionally cohere) and qdrant. This is basically a glorified
(bloated) `grep`.

## Usage

### Setup

#### 1. Environment Variables

Create a .env file or set the following parameters:

```.env
CHUNK_SIZE=4096
CHUNK_OVERLAP=256

ENCODER_MODEL=nomic-embed-text
EMBEDDING_DIM=768
RETRIEVER_TOP_K=15
RETRIEVER_SCORE_THRESHOLD=0.5

RERANK_MODEL=mixedbread-ai/mxbai-rerank-large-v1
RERANK_TOP_K=5

GENERATOR_MODEL=llama3

DOCUMENT_DB_NAME=rag
DOCUMENT_DB_USER=aktersnurra

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=knowledge-base

COHERE_API_KEY = <COHERE_API_KEY> # OPTIONAL
COHERE_RERANK_MODEL = "rerank-english-v3.0"
```

#### 2. Install Python Dependencies

```
poetry install
```

#### 3. Ollama

Make sure ollama is running:

```sh
ollama serve
```

Download the encoder and generator models with ollama:

```sh
ollama pull $GENERATOR_MODEL
ollama pull $ENCODER_MODEL
```

#### 4. Qdrant

Qdrant is used to store the embeddings of the chunks from the documents.

Download and run qdrant.

#### 5. Postgres

Postgres is used to save hashes of the document to prevent documents from
being added to the vector db more than ones.

Download and run qdrant.

#### 6. Cohere

Get an API from their website, but is optional.

### Running

Activate the poetry shell:

```sh
poetry shell
```

Use the cli:

```sh
python rag/cli.py
```

or the ui using a browser:

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
