# Hybrid-RAG-with-Semantic-and-Syntactic-Search

This repository contains the implementation of a Hybrid Retrieval-Augmented Generation (RAG) model that combines semantic (dense vector) and syntactic (sparse vector) search to leverage the strengths of both approaches. The implementation uses the PineconeHybridSearchRetriever, HuggingFaceEmbeddings for dense vectors, and BM25Encoder for sparse vectors.

## Overview

In information retrieval, semantic search (vector similarity) and syntactic search (keyword search) each have their own strengths and weaknesses:

- **Vector Similarity Search**: Handles typos and overall intent well but may miss precise keyword matches.
- **Keyword Search**: Good at precise matching on keywords, abbreviations, and names but may fail to capture the overall intent if keywords are not exact.

### Hybrid Search Approach

In a hybrid search, we combine both vector search and keyword search to take advantage of their strengths while mitigating their limitations. This approach uses a keyword-sensitive semantic search algorithm.

#### Combination Formula

To balance the scores from vector search and keyword search, we use the following formula:

\[ H = (1-\alpha) K + \alpha V \]

where:
- \( H \) is the hybrid search score.
- \( \alpha \) is the weighted parameter.
- \( K \) is the keyword search score.
- \( V \) is the vector search score.

- When \( \alpha \) is 1, the hybrid score is purely based on vector search.
- When \( \alpha \) is 0, the hybrid score is purely based on keyword search.

### Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion (RRF) is used to combine dense and sparse search scores. RRF ranks passages according to their positions in keyword and vector outcome lists and merges these rankings to generate a unified result list. The RRF score is calculated by summing the inverse rankings from each list.

## Implementation

### Dependencies

Install the required libraries:

```bash
pip install -r requirements.txt

# Initialize Pinecone
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")

# Create an index
index_name = "climate-change"
pinecone.create_index(index_name, dimension=768)
index = pinecone.Index(index_name)

# Dense embeddings using HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sparse embeddings using BM25Encoder
bm25encoder = BM25Encoder().default()
Adding Texts to the Index

# Dense embeddings
dense_embeddings = embeddings.encode(texts)


# Display the top result
print(result[0].page_content)
```
Expected Output
The retriever should return the document most related to the query at the top

# We can observe the output based on ranking for different questions we ask
Example:
```python
result =retriever.invoke("How are agricultural zones shifting due to climate change")
result[0]
```
Document(page_content='Shifts in Agricultural Zones As global temperatures rise, agricultural zones are shifting. Crops that were traditionally grown in temperate regions are now being cultivated in areas previously unsuitable for agriculture due to cooler climates. For instance, vineyards are now being established in regions of Northern Europe that were once considered too cold for grape production. However, this shift also means that regions currently suitable for specific crops may become less viable, forcing farmers to adapt by changing crops or altering farming practices.')

```python
result =retriever.invoke("What are the socio-economic implications of climate change on agriculture?")
result[0]
```
Document(page_content='Socio-Economic Implications The socio-economic implications of climate change on agriculture are profound. Smallholder farmers, particularly in developing countries, are among the most vulnerable. They often lack the resources and technology needed to adapt to changing conditions. This vulnerability can lead to reduced income, food insecurity, and increased poverty. Additionally, changes in agricultural productivity can affect global food prices, influencing both the availability and affordability of food worldwide.')


# Conclusion
This repository demonstrates how to implement a hybrid search approach using vector and keyword search, combining their strengths to improve information retrieval performance. The approach balances vector similarity and keyword matching using a combination formula and Reciprocal Rank Fusion.

Feel free to explore and modify the code to suit your specific requirements.

