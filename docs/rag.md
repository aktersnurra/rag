# Retrieval Augmented Generation

<center>
  <img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/how-rag-works.jpg" alt="image" width="1024" height="auto">
</center>

{pause up}

# The Retriever

<center>
  <img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/how-indexing-works.jpg" alt="image" width="1024" height="auto">
</center>

{pause up}

<center>
  <img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/how-retrieval-works.jpg" alt="image" width="1024" height="auto">
</center>

{pause up}

<center>
  <img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/how-generation-works.png" alt="image" width="1024" height="auto">
</center>

{pause up}

{#rag}
<center>
  <img src="https://qdrant.tech/articles_data/what-is-rag-in-ai/rag-system.jpg" alt="image" width="1024" height="auto">
</center>

{pause up}

<center>
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXM0ZGtsZzdldjh5cW54bnN1MTA1dDl3cjV2c2p2NmRiMHpkYmZyYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QYLQRR7IF48njkq5an/giphy.gif" alt="image" width="1024" height="auto">

some of the snippets retrieved from the vector database are not always relevant or
of good quality, e.g. the contents pages of a book

{pause .block #solution}
> **Solution?**
>
> {pause focus-at-unpause=solution}
> Add another **LLM** of course!

</center>

{pause up}

# Reranker

<center>
<img src="https://cdn.sanity.io/images/vr8gru94/production/906c3c0f8fe637840f134dbf966839ef89ac7242-3443x1641.png" alt="image" width="1024" height="auto">
</center>

{pause up}

A common reranking model is the cross encoder:

<center>
<img src="https://cdn.sanity.io/images/vr8gru94/production/9f0d2f75571bb58eecf2520a23d300a5fc5b1e2c-2440x1100.png" alt="image" width="1024" height="auto">
</center>

We plug this reranking model into the rag pipeline...
{pause up-at-unpause=rag}
