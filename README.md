# Retrieval Augmented Generation

## Plan

- [ ] Architecture
  - [ ] Vector store
    - [ ] which one? FAISS?
    - [ ] Build index of the document
  - [ ] Embedding model (mxbai-embed-large)
  - [ ] LLM (Dolphin)
- [ ] Gather some documents
- [ ] Create a prompt for the query


### Pre-Processing of Document
1. Use langchain document loader and splitter
   ```python
   from langchain_community.document_loaders import PyPDFLoader
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   ```

2. Generate embeddings with mxbai, example:
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. load model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# For retrieval you need to pass this prompt.
query = 'Represent this sentence for searching relevant passages: A man is eating a piece of bread'

docs = [
    query,
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

# 2. Encode
embeddings = model.encode(docs)

# 3. Calculate cosine similarity
similarities = cos_sim(embeddings[0], embeddings[1:])
```
But we will use ollama...

(otherwise install `sentence-transformers`)

3. Create vector store 
```python
import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

I need to figure out the vector dim of the mxbai model. -> 1024

4. Use Postgres as a persisted kv-store

Save index of chunk as key and value as paragraph.

5. Create user input pipeline

5.1 Create search prompt for document retrieval

5.2 Fetch nearest neighbors as context

5.3 Retrieve the values from the document db

5.4 Add paragraphs as context to the query

5.5 Send query to LLM

5.6 Return output

5.7 ....

5.8 Profit

### Frontend (Low priority)

[streamlit](https://github.com/streamlit/streamlit)


### Tutorial

[link](https://medium.com/@solidokishore/building-rag-application-using-langchain-openai-faiss-3b2af23d98ba)
