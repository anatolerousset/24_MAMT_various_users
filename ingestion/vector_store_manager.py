"""Vector store management for document embeddings."""

import logging
import uuid
from typing import List
from tqdm import tqdm

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
from qdrant_client import models, QdrantClient
from fastembed import SparseTextEmbedding

from config.config import Config

_log = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for document embeddings."""
    
    def __init__(self, qdrant_url: str = None):
        """
        Initialize the vector store manager.
        
        Args:
            qdrant_url: URL for Qdrant instance
        """
        self.qdrant_url = qdrant_url or Config.QDRANT_URL
        self.client = QdrantClient(url=self.qdrant_url)
        
        # Initialize embedding models
        self.dense_embedding_model = HuggingFaceEmbeddings(
            model_name=Config.FINETUNED_DENSE_EMBED_MODEL_ID
        )
        self.sparse_embedding_model = SparseTextEmbedding("Qdrant/bm25")
        
        # Get dimensions from sample embeddings
        sample_text = "Sample text for dimension detection"
        dense_sample = list(self.dense_embedding_model.embed_documents([sample_text]))[0]
        self.dense_dim = len(dense_sample)
        
        _log.info(f"Dense embedding dimension: {self.dense_dim}")
    
    def create_collection(self, collection_name: str, recreate: bool = False):
        """
        Create a Qdrant collection with hybrid vector support.
        
        Args:
            collection_name: Name of the collection to create
            recreate: Whether to recreate if collection exists
        """
        # Check if collection exists
        if self.client.collection_exists(collection_name):
            if recreate:
                _log.info(f"Collection {collection_name} exists, recreating...")
                self.client.delete_collection(collection_name)
            else:
                _log.info(f"Collection {collection_name} already exists, skipping creation")
                return
        
        # Create collection with multiple vector types
        _log.info(f"Creating collection {collection_name} with hybrid vectors...")
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # Dense vector (BGE-M3)
                "finetuned_dense": models.VectorParams(
                    size=self.dense_dim, 
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                # Sparse vector (BM25)
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )
        _log.info(f"Collection {collection_name} created successfully")
    
    def add_documents_to_collection(self, 
                                  collection_name: str, 
                                  chunks: List[LCDocument],
                                  batch_size: int = None):
        """
        Add documents to a collection with hybrid embeddings.
        
        Args:
            collection_name: Name of the collection
            chunks: List of document chunks to add
            batch_size: Batch size for processing
        """
        batch_size = batch_size or Config.DEFAULT_BATCH_SIZE
        
        _log.info(f"Adding {len(chunks)} documents to collection {collection_name}...")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
            batch = chunks[i:i+batch_size]
            batch_texts = [doc.page_content for doc in batch]
            batch_metadatas = [doc.metadata for doc in batch]
            
            _log.debug(f"Processing batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}...")
            
            # Generate embeddings for this batch
            dense_embeddings = list(self.dense_embedding_model.embed_documents(batch_texts))
            sparse_embeddings = list(self.sparse_embedding_model.passage_embed(batch_texts))
            
            # Create points
            points = []
            for text, metadata, dense_emb, sparse_emb in zip(
                batch_texts, batch_metadatas, dense_embeddings, sparse_embeddings
            ):
                point_id = str(uuid.uuid4())
                
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "finetuned_dense": dense_emb,
                            "bm25": sparse_emb.as_object(),
                        },
                        payload={
                            "page_content": text,
                            "metadata": metadata
                        }
                    )
                )
            
            # Add batch to the collection
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            
            _log.debug(f"Batch {i//batch_size + 1} added successfully")
        
        _log.info(f"✓ Successfully added {len(chunks)} documents to {collection_name}")
        
        # Get collection info
        collection_info = self.client.get_collection(collection_name)
        _log.info(f"Collection info: {collection_info}")
    
    def update_collection(self, 
                         collection_name: str, 
                         chunks: List[LCDocument],
                         batch_size: int = None):
        """
        Update an existing collection with new documents.
        This appends new documents without removing existing ones.
        
        Args:
            collection_name: Name of the collection
            chunks: List of document chunks to add
            batch_size: Batch size for processing
        """
        if not self.client.collection_exists(collection_name):
            _log.info(f"Collection {collection_name} doesn't exist, creating it...")
            self.create_collection(collection_name, recreate=False)
        
        # Get current collection info before update
        try:
            collection_info = self.client.get_collection(collection_name)
            current_count = collection_info.points_count
            _log.info(f"Collection {collection_name} currently has {current_count} documents")
        except Exception as e:
            _log.warning(f"Could not get collection info: {e}")
            current_count = 0
        
        # Add new documents
        self.add_documents_to_collection(collection_name, chunks, batch_size)
        
        # Log the update
        try:
            updated_info = self.client.get_collection(collection_name)
            new_count = updated_info.points_count
            added_count = new_count - current_count
            _log.info(f"✓ Updated collection {collection_name}: added {added_count} documents (total: {new_count})")
        except Exception as e:
            _log.warning(f"Could not get updated collection info: {e}")
    
    def recreate_collection(self, 
                          collection_name: str, 
                          chunks: List[LCDocument],
                          batch_size: int = None):
        """
        Recreate a collection from scratch with new documents.
        
        Args:
            collection_name: Name of the collection
            chunks: List of document chunks to add
            batch_size: Batch size for processing
        """
        _log.info(f"Recreating collection {collection_name} from scratch...")
        
        # Create collection (will delete if exists)
        self.create_collection(collection_name, recreate=True)
        
        # Add documents
        self.add_documents_to_collection(collection_name, chunks, batch_size)
        
        _log.info(f"✓ Collection {collection_name} recreated with {len(chunks)} documents")
    
    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        try:
            if not self.client.collection_exists(collection_name):
                return {"exists": False, "error": "Collection does not exist"}
            
            info = self.client.get_collection(collection_name)
            return {
                "exists": True,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "config": info.config
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the Qdrant instance.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            _log.error(f"Error listing collections: {e}")
            return []