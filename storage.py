import os
import uuid
from typing import List, Dict, Any, Optional
import qdrant_client

from rank_bm25 import BM25Okapi
from qdrant_client.http import models
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from get_context_online import get_online_context
from config.secretKey import QDRANT_KEY, QDRANT_URL

# Initialize embedding model with cached resource to avoid reloading
@st.cache_resource
def _load_embedding_model():
    return SentenceTransformer('keepitreal/vietnamese-sbert')

embedding_model = _load_embedding_model()

class DocumentStore:
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        """
        Initialize the document store with Qdrant.
        
        Args:
            qdrant_url (str): URL for the Qdrant service
            qdrant_api_key (str): API key for Qdrant authentication
        """
        self.qdrant_url = qdrant_url or QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or QDRANT_KEY
        
        # Initialize Qdrant client with connection timeout
        self.client = qdrant_client.QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=30
        )
        
    def get_collection(self, collection_name: str = "vku_document_single_point_v3") -> None:
        """
        Get or create a Qdrant collection with optimized configuration.
        
        Args:
            collection_name (str): Name of the collection
        """
        try:
            collections = self.client.get_collections().collections
            if collection_name not in [c.name for c in collections]:
                vector_size = embedding_model.get_sentence_embedding_dimension()
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                        hnsw_config=models.HnswConfigDiff(
                            m=16,
                            ef_construct=100
                        )
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000
                    )
                )
                st.success(f"Created new collection: {collection_name}")
        except Exception as e:
            st.error(f"Failed to access/create collection: {str(e)}")
            raise
    
    def store_document_chunks(
        self, 
        doc_id: str,
        chunks: List[str], 
        collection_name: str = "vku_document_single_point_v3"
    ) -> None:
        """
        Store document chunks as individual points in Qdrant with batch processing.
        
        Args:
            doc_id (str): Unique document identifier
            chunks (List[str]): List of text chunks to store
            collection_name (str): Qdrant collection name
        """
        try:
            if not chunks:
                raise ValueError("No chunks provided for embedding")

            self.get_collection(collection_name)
            
            batch_size = 100
            with st.spinner(f"Storing {len(chunks)} chunks..."):
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_points = [
                        models.PointStruct(
                            id=i + j,
                            vector=embedding_model.encode(chunk).tolist(),
                            payload={
                                "doc_id": doc_id,
                                "text": chunk,
                                "chunk_index": i + j
                            }
                        )
                        for j, chunk in enumerate(batch_chunks)
                    ]
                    
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch_points,
                        wait=True
                    )
            
            st.success(f"Stored {len(chunks)} chunks for document '{doc_id}'")
        
        except Exception as e:
            st.error(f"Failed to store document chunks: {str(e)}")
            raise

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0,1] range using min-max normalization.
        
        Args:
            scores (np.ndarray): Input scores to normalize
            
        Returns:
            np.ndarray: Normalized scores
        """
        if scores.size == 0:
            return scores
        if scores.size == 1:
            return np.array([1.0])
            
        score_min = scores.min()
        score_max = scores.max()
        
        if score_max == score_min:
            return np.ones_like(scores) if score_max != 0 else np.zeros_like(scores)
            
        return (scores - score_min) / (score_max - score_min)

    def retrieve_relevant_chunks(
        self,
        query: str, 
        collection_name: str = "vku_document_single_point_v3", 
        top_k: int = 5,
        use_hybrid_search: bool = True,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_chunks_per_doc: int = 4
    ) -> List[str]:
        """
        Retrieve relevant document chunks using hybrid search (Dense + BM25).
        """
        try:
            # Encode the query
            query_embedding = embedding_model.encode(query).tolist()
            
            # Search for more documents to ensure diversity
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k * 2,  # Retrieve more documents to ensure diversity
                with_payload=True
            )

            # Create a single list of all chunks with their scores
            all_ranked_chunks = []
            seen_chunk_texts = set()
            tokenized_query = query.split()

            for point in search_results:
                if "chunks" in point.payload:
                    chunks = point.payload["chunks"]
                    if use_hybrid_search:
                        # Create BM25 scores for this document's chunks
                        bm25 = BM25Okapi([c["text"].split() for c in chunks])
                        bm25_scores = bm25.get_scores(tokenized_query)
                        
                        # Normalize BM25 scores
                        bm25_scores = self._normalize_scores(np.array(bm25_scores))
                        
                        for i, chunk in enumerate(chunks):
                            chunk_text = chunk["text"]
                            
                            # Skip if we've seen this exact text before
                            if chunk_text in seen_chunk_texts:
                                continue
                                
                            seen_chunk_texts.add(chunk_text)
                            
                            # Dense score
                            chunk_embedding = np.array(chunk["embedding"])
                            query_emb = np.array(query_embedding)
                            cosine_sim = np.dot(query_emb, chunk_embedding) / (
                                np.linalg.norm(query_emb) * np.linalg.norm(chunk_embedding) + 1e-8
                            )
                            
                            # Combine scores
                            combined_score = dense_weight * cosine_sim + bm25_weight * bm25_scores[i]
                            
                            # Add document identifier to help with diversity
                            doc_id = point.id
                            all_ranked_chunks.append((chunk_text, combined_score, doc_id))
                    else:
                        # For non-hybrid search, just add chunks with a placeholder score
                        for chunk in chunks:
                            chunk_text = chunk["text"]
                            if chunk_text not in seen_chunk_texts:
                                seen_chunk_texts.add(chunk_text)
                                doc_id = point.id
                                all_ranked_chunks.append((chunk_text, 1.0, doc_id))
                else:
                    # Handle single text payload
                    chunk_text = point.payload.get("text", "")
                    if chunk_text and chunk_text not in seen_chunk_texts:
                        seen_chunk_texts.add(chunk_text)
                        doc_id = point.id
                        all_ranked_chunks.append((chunk_text, 1.0, doc_id))

            # Sort all chunks by score
            all_ranked_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Implement diversity by selecting chunks from different documents
            selected_chunks = []
            selected_docs = set()
            
            # First, select top chunk from each document
            for chunk_text, score, doc_id in all_ranked_chunks:
                if doc_id not in selected_docs and len(selected_chunks) < top_k:
                    selected_chunks.append(chunk_text)
                    selected_docs.add(doc_id)
            
            # Then, add remaining chunks by score until we reach the desired count
            for chunk_text, score, doc_id in all_ranked_chunks:
                if chunk_text not in selected_chunks and len(selected_chunks) < top_k * top_chunks_per_doc:
                    selected_chunks.append(chunk_text)
                    
                    # Stop when we've reached the target number
                    if len(selected_chunks) >= top_k:
                        break

            return selected_chunks

        except Exception as e:
            st.error(f"Failed to retrieve relevant chunks: {str(e)}")
            raise
    def store_document_as_single_point(
        self, 
        doc_id: str,
        chunks: List[str], 
        collection_name: str = "vku_document_single_point_v3"
    ) -> None:
        """
        Store all chunks of a document in a single Qdrant point.
        
        Args:
            doc_id (str): Unique document identifier
            chunks (List[str]): List of text chunks to store
            collection_name (str): Qdrant collection name
        """
        try:
            if not chunks:
                raise ValueError("No chunks provided for embedding")

            self.get_collection(collection_name)
            
            with st.spinner(f"Processing document with {len(chunks)} chunks..."):
                chunk_embeddings = [
                    {"text": chunk, "embedding": embedding_model.encode(chunk).tolist()}
                    for chunk in chunks
                ]
                
                # Use centroid of chunk embeddings for document representation
                doc_embedding = np.mean(
                    [np.array(chunk["embedding"]) for chunk in chunk_embeddings], axis=0
                ).tolist()
                
                point = models.PointStruct(
                    id=int(uuid.uuid4().int & (2**63 - 1)),
                    vector=doc_embedding,
                    payload={
                        "doc_id": doc_id,
                        "chunks": chunk_embeddings,
                        "chunk_count": len(chunks)
                    }
                )
                
                self.client.upsert(
                    collection_name=collection_name,
                    points=[point],
                    wait=True
                )
            
            st.success(f"Stored document '{doc_id}' with {len(chunks)} chunks")
        
        except Exception as e:
            st.error(f"Failed to store document as single point: {str(e)}")
            raise
    
    def clear_collection(self, collection_name: str = "vku_document_single_point_v3") -> None:
        """
        Clear all documents from a collection and recreate it.
        
        Args:
            collection_name (str): Name of the collection to clear
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            st.success(f"Collection '{collection_name}' cleared")
            self.get_collection(collection_name)
        except Exception as e:
            st.error(f"Failed to clear collection: {str(e)}")
            raise

@st.cache_resource
def initialize_document_store() -> DocumentStore:
    """
    Initialize and cache the document store instance.
    
    Returns:
        DocumentStore: Initialized document store
    """
    if not QDRANT_URL or not QDRANT_KEY:
        st.error("Qdrant credentials missing. Please configure QDRANT_URL and QDRANT_KEY.")
        raise ValueError("Missing Qdrant credentials")
    
    return DocumentStore(qdrant_url=QDRANT_URL, qdrant_api_key=QDRANT_KEY)

def store_document_chunks(chunks: List[str], use_single_point: bool = True) -> None:
    """
    Store document chunks using the document store.
    
    Args:
        chunks (List[str]): List of text chunks to store
        use_single_point (bool): Store as single point (True) or multiple points (False)
    """
    doc_store = initialize_document_store()
    doc_id = str(uuid.uuid4())
    if use_single_point:
        doc_store.store_document_as_single_point(doc_id, chunks)
    else:
        doc_store.store_document_chunks(doc_id, chunks)

def get_relevant_chunks(query: str, top_k: int = 3, use_hybrid_search: bool = True) -> List[str]:
    """
    Get relevant chunks for a query.
    
    Args:
        query (str): User query
        top_k (int): Number of results to retrieve
        use_hybrid_search (bool): Whether to use hybrid search
        
    Returns:
        List[str]: List of relevant text chunks
    """
    doc_store = initialize_document_store()
    return doc_store.retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        use_hybrid_search=use_hybrid_search
    )