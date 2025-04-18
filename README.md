# 🔍 Enhanced Document Retriever

A powerful document retrieval system using vector search with hybrid ranking for optimal results.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.7%2B-brightgreen.svg)
![Qdrant](https://img.shields.io/badge/database-Qdrant-orange.svg)

## ✨ Features

- **Hybrid Search** - Combine dense vector similarity and BM25 lexical search for superior results
- **Intelligent Chunk Selection** - Smart document chunk ranking with duplicate removal
- **Document Diversity** - Ensures results come from varied sources instead of the same document
- **Configurable Parameters** - Easily adjust search weights, result count, and more

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-document-retriever.git
cd enhanced-document-retriever

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

```python
from document_retriever import DocumentRetriever

# Initialize the retriever
retriever = DocumentRetriever(
    qdrant_url="http://localhost:6333",
    api_key="your_api_key"
)

# Retrieve relevant chunks
results = retriever.retrieve_relevant_chunks(
    query="What are the effects of climate change?",
    top_k=5,
    use_hybrid_search=True,
    dense_weight=0.6,
    bm25_weight=0.4
)

# Process the results
for i, chunk in enumerate(results):
    print(f"Result {i+1}:\n{chunk}\n")
```

## ⚙️ Configuration

Key parameters for `retrieve_relevant_chunks`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `query` | The user's search query | - |
| `collection_name` | Qdrant collection to search | "vku_document_mul_point_json_v2" |
| `top_k` | Number of top documents to retrieve | 5 |
| `use_hybrid_search` | Whether to use hybrid search | True |
| `dense_weight` | Weight for dense similarity score | 0.5 |
| `bm25_weight` | Weight for BM25 score | 0.5 |
| `top_chunks_per_doc` | Max chunks to return per document | 4 |

## 🧪 How It Works

1. **Query Encoding** - Transforms the query into a vector embedding
2. **Document Retrieval** - Fetches candidate documents from Qdrant
3. **Chunk Extraction** - Extracts text chunks from retrieved documents
4. **Hybrid Scoring** - Scores chunks using both vector similarity and BM25
5. **Diversity Ranking** - Ensures diverse results across documents
6. **Deduplication** - Removes duplicate chunks before returning results

## 🛠️ Advanced Usage

### Custom Embedding Models

```python
from sentence_transformers import SentenceTransformer

retriever = DocumentRetriever(
    qdrant_url="http://localhost:6333",
    api_key="your_api_key",
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
)
```

### Custom Scoring Functions

```python
def custom_score_normalizer(scores):
    """Custom score normalization function"""
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

retriever._normalize_scores = custom_score_normalizer
```

## 📊 Performance Tips

- Increase `top_k` to get more diverse documents but expect longer processing time
- For precision-focused search, increase `dense_weight`
- For recall-focused search, increase `bm25_weight`
- For faster results with less diversity, set `use_hybrid_search=False`

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Qdrant](https://qdrant.tech/) for the vector database
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) implementation for lexical search
- [Sentence Transformers](https://www.sbert.net/) for document embeddings
