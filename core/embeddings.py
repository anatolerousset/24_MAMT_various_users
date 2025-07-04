"""
Embedding models management for the RAG Chatbot application.
"""
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from config.config import Config


class EmbeddingManager:
    """Manages embedding models for the RAG system."""
    
    def __init__(self):
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all embedding models."""
        # Initialize dense embedding model
        self.models["finetuned_dense"] = HuggingFaceEmbeddings(
            model_name= Config.FINETUNED_DENSE_EMBED_MODEL_ID, 
            model_kwargs={"device": "cpu"}
        )
        
        # Initialize sparse embedding model
        self.models["sparse"] = SparseTextEmbedding("Qdrant/bm25")
        
        # Store reranker ID (will be loaded on demand to save memory)
        self.models["bge_reranker_id"] = Config.FINETUNED_BGE_RERANKER_MODEL_ID
    
    def get_models(self) -> dict:
        """Get all embedding models."""
        return self.models
    
    def get_dense_model(self):
        """Get the dense embedding model."""
        return self.models.get("finetuned_dense")
    
    def get_sparse_model(self):
        """Get the sparse embedding model."""
        return self.models.get("sparse")
    
    def get_reranker_id(self) -> str:
        """Get the reranker model ID."""
        return self.models.get("bge_reranker_id")


def get_embedding_models() -> dict:
    """
    Factory function to get embedding models.
    Maintains backward compatibility with the original code.
    """
    manager = EmbeddingManager()
    return manager.get_models()