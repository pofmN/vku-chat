from rank_bm25 import BM25Okapi
import numpy as np
import uuid
import streamlit as st
from typing import List, Dict
from qdrant_client import QdrantClient, models

# Ensure embedding_model is loaded before running these functions
embedding_model = None  # Replace with your actual embedding model
client = QdrantClient("localhost")  # Replace with your Qdrant server connection details

def get_collection(client: QdrantClient, collection_name: str):
    """
    Ensure the collection exists in Qdrant.
    """
    try:
        collections = client.get_collections()
        if collection_name not in [c.name for c in collections.collections]:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
            )
    except Exception as e:
        st.error(f"Error ensuring collection exists: {str(e)}")
        raise


def store_document_as_single_point(
    client: QdrantClient,
    embedding_model,
    doc_id: str,
    chunks: List[str], 
    collection_name: str
) -> None:
    """
    Store all chunks of a document in a single Qdrant point.
    """
    try:
        if not chunks:
            raise ValueError("No chunks provided for embedding")

        get_collection(client, collection_name)

        with st.spinner(f"Processing document with {len(chunks)} chunks..."):
            # Generate embeddings for all chunks
            chunk_embeddings = []
            for chunk in chunks:
                embedding = embedding_model.encode(chunk).tolist()
                chunk_embeddings.append({
                    "text": chunk,
                    "embedding": embedding
                })

            # Create a representative embedding for the document
            doc_embedding = chunk_embeddings[0]["embedding"]

            # Store document as a single point
            point = models.PointStruct(
                id=int(uuid.uuid4().int % (2**63 - 1)),  # Convert UUID to positive integer
                vector=doc_embedding,
                payload={
                    "doc_id": doc_id,
                    "chunks": chunk_embeddings,
                    "chunk_count": len(chunks)
                }
            )

            # Insert the point
            client.upsert(
                collection_name=collection_name,
                points=[point]
            )

        st.success(f"Successfully stored document '{doc_id}' as a single point with {len(chunks)} chunks")

    except Exception as e:
        st.error(f"Error storing document as single point: {str(e)}")
        raise


def retrieve_relevant_chunks(
    client: QdrantClient,
    embedding_model,
    query: str, 
    collection_name: str = "vku_document_single_point_json", 
    top_k: int = 3,
    use_hybrid_search: bool = True,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    top_chunks_per_doc: int = 3
) -> List[str]:
    """
    Retrieve relevant document chunks using hybrid search (Dense + BM25).
    """
    try:
        # Encode the query for dense search
        query_embedding = embedding_model.encode(query).tolist()

        # Search for relevant points/documents in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )

        relevant_chunks = []

        for point in search_results:
            # Check if this is a single-point document with multiple chunks
            if "chunks" in point.payload:
                chunks = point.payload["chunks"]

                if use_hybrid_search:
                    # Hybrid re-ranking: combine dense cosine similarity with BM25 score
                    ranked_chunks = []
                    tokenized_query = query.split()  # Tokenize query for BM25

                    for chunk in chunks:
                        # --- Dense Score ---
                        chunk_embedding = np.array(chunk["embedding"])
                        query_emb = np.array(query_embedding)
                        cosine_sim = np.dot(query_emb, chunk_embedding) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_embedding) + 1e-8)

                        # --- BM25 Score ---
                        tokenized_chunk = chunk["text"].split()
                        bm25 = BM25Okapi([tokenized_chunk])
                        bm25_score = bm25.get_scores(tokenized_query)[0]

                        # Combine scores (adjust weights)
                        combined_score = dense_weight * cosine_sim + bm25_weight * bm25_score
                        ranked_chunks.append((chunk, combined_score))

                    # Sort chunks by combined score (highest first)
                    ranked_chunks = sorted(ranked_chunks, key=lambda x: x[1], reverse=True)

                    # Select top-ranked chunks from this document
                    top_chunks = ranked_chunks[:top_chunks_per_doc]
                    relevant_chunks.extend([chunk["text"] for chunk, _ in top_chunks])
                else:
                    # If not using hybrid, simply take the first few chunks
                    relevant_chunks.extend([chunk["text"] for chunk in chunks[:top_chunks_per_doc]])
            else:
                # For regular stored (non-grouped) chunks
                relevant_chunks.append(point.payload["text"])

        return relevant_chunks

    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {str(e)}")
        raise
