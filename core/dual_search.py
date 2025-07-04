"""
Dual search service that handles searching across multiple collections.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document as LCDocument
from qdrant_client import QdrantClient

from core.search import HybridSearchEngine
from utils.download_manager_image import process_chunks_with_images
from config.config import Config

_log = logging.getLogger(__name__)


class DualSearchService:
    """Service for searching across technical and DCE collections simultaneously."""
    
    def __init__(self, qdrant_client: QdrantClient, models_dict: dict, image_manager=None):
        """
        Initialize the dual search service.
        
        Args:
            qdrant_client: Qdrant client instance
            models_dict: Dictionary of embedding models
            image_manager: Optional image manager for processing
        """
        self.search_engine = HybridSearchEngine(qdrant_client, models_dict)
        self.image_manager = image_manager
    
    async def dual_search(
        self,
        query: str,
        technical_collection: str = None,
        dce_collection: str = None,
        threshold: float = 0.1,
        max_results: int = 15,
        use_reranker: bool = True,
        process_images: bool = True
    ) -> Dict[str, Any]:
        """
        Perform dual search across technical and DCE collections.
        
        Args:
            query: Search query
            technical_collection: Technical collection name
            dce_collection: DCE collection name
            threshold: Similarity threshold
            max_results: Maximum results per collection
            use_reranker: Whether to use reranking
            process_images: Whether to process images in results
            
        Returns:
            Dictionary containing search results from both collections
        """
        # Use default collection names if not provided
        technical_collection = technical_collection or Config.COLLECTION_NAME
        dce_collection = dce_collection or Config.DCE_COLLECTION
        
        _log.info(f"Starting dual search for query: {query[:50]}...")
        _log.info(f"Collections: {technical_collection}, {dce_collection}")
        
        # Search results containers
        results = {
            'technical_results': None,
            'dce_results': None,
            'total_documents': 0,
            'query': query,
            'collections_searched': []
        }
        
        # Search in technical collection
        try:
            _log.debug(f"Searching in technical collection: {technical_collection}")
            tech_search_results = await self.search_engine.search_with_threshold(
                query=query,
                collection_name=technical_collection,
                threshold=threshold,
                max_results=max_results,
                use_reranker=use_reranker
            )
            
            if tech_search_results:
                results['technical_results'] = self._format_search_results(
                    tech_search_results, technical_collection, use_reranker
                )
                results['collections_searched'].append(technical_collection)
                _log.info(f"✓ Technical search: {results['technical_results']['total_results']} results")
            else:
                results['technical_results'] = self._empty_search_results(technical_collection)
                _log.info("Technical search: 0 results")
                
        except Exception as e:
            _log.error(f"Error in technical collection search: {e}")
            results['technical_results'] = self._empty_search_results(technical_collection)
        
        # Search in DCE collection
        try:
            _log.debug(f"Searching in DCE collection: {dce_collection}")
            dce_search_results = await self.search_engine.search_with_threshold(
                query=query,
                collection_name=dce_collection,
                threshold=threshold,
                max_results=max_results,
                use_reranker=use_reranker
            )
            
            if dce_search_results:
                results['dce_results'] = self._format_search_results(
                    dce_search_results, dce_collection, use_reranker
                )
                results['collections_searched'].append(dce_collection)
                _log.info(f"✓ DCE search: {results['dce_results']['total_results']} results")
            else:
                results['dce_results'] = self._empty_search_results(dce_collection)
                _log.info("DCE search: 0 results")
                
        except Exception as e:
            _log.error(f"Error in DCE collection search: {e}")
            results['dce_results'] = self._empty_search_results(dce_collection)
        
        # Calculate total documents
        tech_count = results['technical_results']['total_results'] if results['technical_results'] else 0
        dce_count = results['dce_results']['total_results'] if results['dce_results'] else 0
        results['total_documents'] = tech_count + dce_count
        
        # Process images if requested and available
        if process_images and self.image_manager and results['total_documents'] > 0:
            try:
                results = await self._process_images_in_results(results)
                _log.info("✓ Image processing completed")
            except Exception as e:
                _log.error(f"Image processing error: {e}")
        
        _log.info(f"Dual search completed: {results['total_documents']} total documents")
        return results
    
    def _format_search_results(
        self, 
        search_results: tuple, 
        collection_name: str, 
        use_reranker: bool
    ) -> Dict[str, Any]:
        """Format search results into a standardized dictionary."""
        
        # Unpack results based on whether reranking was used
        if use_reranker and len(search_results) == 6:
            docs, scores, origins, original_ranks, original_scores, reranked = search_results
        else:
            docs, scores, origins = search_results[:3]
            original_ranks = None
            original_scores = None
            reranked = False
        
        # Convert documents to serializable format
        serialized_docs = []
        for doc in docs:
            serialized_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            'documents': serialized_docs,
            'scores': self._convert_numpy_types(scores),
            'origins': self._convert_numpy_types(origins),
            'original_ranks': self._convert_numpy_types(original_ranks) if original_ranks else None,
            'original_scores': self._convert_numpy_types(original_scores) if original_scores else None,
            'reranked': reranked,
            'total_results': len(docs),
            'collection_name': collection_name
        }
    
    def _empty_search_results(self, collection_name: str) -> Dict[str, Any]:
        """Create empty search results structure."""
        return {
            'documents': [],
            'scores': [],
            'origins': [],
            'original_ranks': None,
            'original_scores': None,
            'reranked': False,
            'total_results': 0,
            'collection_name': collection_name
        }
    
    async def _process_images_in_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process images in search results."""
        if not self.image_manager:
            return results
        
        # Collect all documents for image processing
        all_docs_for_processing = []
        
        # Add technical documents
        if results['technical_results'] and results['technical_results']['documents']:
            tech_docs = [
                LCDocument(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in results['technical_results']['documents']
            ]
            all_docs_for_processing.extend(tech_docs)
        
        # Add DCE documents
        if results['dce_results'] and results['dce_results']['documents']:
            dce_docs = [
                LCDocument(page_content=doc["page_content"], metadata=doc["metadata"])
                for doc in results['dce_results']['documents']
            ]
            all_docs_for_processing.extend(dce_docs)
        
        if not all_docs_for_processing:
            return results
        
        # Process images
        processed_chunks = process_chunks_with_images(all_docs_for_processing)
        
        # Update results with processed chunks
        num_technical = len(results['technical_results']['documents']) if results['technical_results'] else 0
        
        if num_technical > 0 and results['technical_results']:
            processed_tech_docs = processed_chunks[:num_technical]
            results['technical_results']['documents'] = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in processed_tech_docs
            ]
        
        if len(processed_chunks) > num_technical and results['dce_results']:
            processed_dce_docs = processed_chunks[num_technical:]
            results['dce_results']['documents'] = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in processed_dce_docs
            ]
        
        return results
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types."""
        import numpy as np
        
        try:
            if obj is None:
                return None
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            elif hasattr(obj, 'item'):  # For numpy scalars
                return obj.item()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (set, frozenset)):
                return [self._convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return [self._convert_numpy_types(item) for item in obj]
            elif isinstance(obj, list):
                return [self._convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(key): self._convert_numpy_types(value) for key, value in obj.items()}
            else:
                try:
                    return str(obj)
                except Exception:
                    return None
        except Exception as e:
            _log.warning(f"Error converting {type(obj)}: {e}")
            return str(obj) if obj is not None else None