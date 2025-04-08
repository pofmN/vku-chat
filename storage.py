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

embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
#embedding_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

class DocumentStore:
    
    def __init__(self, qdrant_url: str = None, qdrant_api_key: str = None):
        """
        Initialize the document store with Qdrant.
        
        Args:
            qdrant_url (str): URL for the Qdrant service
            qdrant_api_key (str): API key for Qdrant authentication
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_KEY
        )
        
    def get_collection(self, collection_name: str = "vku_document_mul_point_json") -> None:
        """
        Get or create a Qdrant collection.
        
        Args:
            collection_name (str): Name of the collection
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                # Create a new collection with the embedding dimension from the model
                vector_size = len(embedding_model.encode("test"))
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                st.success(f"Created new collection: {collection_name}")
            
        except Exception as e:
            st.error(f"Error accessing Qdrant collection: {str(e)}")
            raise
    
    def store_document_chunks(
        self, 
        doc_id: str,
        chunks: List[str], 
        collection_name: str = "vku_document_mul_point_json"
    ) -> None:
        """
        Store document chunks as individual points in Qdrant.
        
        Args:
            doc_id (str): Unique document identifier
            chunks (List[str]): List of text chunks to store
            collection_name (str): Qdrant collection name
        """
        try:
            if not chunks:
                raise ValueError("No chunks provided for embedding")

            self.get_collection(collection_name)
            
            # Store each chunk as a separate point
            with st.spinner(f"Storing {len(chunks)} chunks..."):
                batch_size = 100  # Process in batches to avoid memory issues
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_points = []
                    
                    for j, chunk in enumerate(batch_chunks):
                        # Generate a unique ID for each point
                        point_id = i + j
                        
                        # Create point with chunk text and metadata
                        point = models.PointStruct(
                            id=point_id,
                            vector=embedding_model.encode(chunk).tolist(),
                            payload={
                                "doc_id": doc_id,
                                "text": chunk,
                                "chunk_index": i + j
                            }
                        )
                        batch_points.append(point)
                    
                    # Insert batch of points
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch_points
                    )
            
            st.success(f"Successfully stored {len(chunks)} chunks for document '{doc_id}'")
        
        except Exception as e:
            st.error(f"Error storing document chunks: {str(e)}")
            raise

    def retrieve_relevant_chunks(
        self,
        query: str, 
        collection_name: str = "vku_document_mul_point_json", 
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
            search_results = self.client.search(
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
    
    def store_document_as_single_point(

            self, 
            doc_id: str,
            chunks: List[str], 
            collection_name: str = "vku_document_mul_point_json"
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
                # Generate embeddings for all chunks
                chunk_embeddings = []
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk).tolist()
                    chunk_embeddings.append({
                        "text": chunk,
                        "embedding": embedding
                    })
                
                # Create a representative embedding for the document
                # Option 1: Use the first chunk's embedding
                doc_embedding = chunk_embeddings[0]["embedding"]
                
                # Option 2 (alternative): Average all chunk embeddings
                # doc_embedding = np.mean([np.array(chunk["embedding"]) for chunk in chunk_embeddings], axis=0).tolist()
                
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
                self.client.upsert(
                    collection_name=collection_name,
                    points=[point]
                )
            
            st.success(f"Successfully stored document '{doc_id}' as a single point with {len(chunks)} chunks")
        
        except Exception as e:
            st.error(f"Error storing document as single point: {str(e)}")
            raise
    
    def clear_collection(self, collection_name: str = "vku_document_mul_point_json") -> None:
        """
        Clear all documents from a collection.
        
        Args:
            collection_name (str): Name of the collection to clear
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            st.success(f"Collection '{collection_name}' deleted successfully")
            # Recreate empty collection
            self.get_collection(collection_name)
        except Exception as e:
            st.error(f"Error clearing collection: {str(e)}")
            raise

@st.cache_resource
def initialize_document_store() -> DocumentStore:
    """
    Initialize and cache the document store instance.
    
    Returns:
        DocumentStore: Initialized document store
    """
    qdrant_url = QDRANT_URL
    qdrant_api_key = QDRANT_KEY
    
    if not qdrant_url or not qdrant_api_key:
        st.warning("Qdrant credentials not found. Please configure them in the settings.")
    
    return DocumentStore(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)

def store_document_chunks(chunks: List[str], use_single_point: bool = True) -> None:
    """
    Store document chunks using the document store.
    
    Args:
        doc_id (str): Unique document identifier
        chunks (List[str]): List of text chunks to store
        use_single_point (bool): Whether to store as a single point (True) or multiple points (False)
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
        use_hybrid_search (bool): Whether to use hybrid search for single-point documents
        
    Returns:
        List[str]: List of relevant text chunks
    """
    doc_store = initialize_document_store()
    return doc_store.retrieve_relevant_chunks(
        query=query,
        top_k=top_k,
        use_hybrid_search=use_hybrid_search
    )