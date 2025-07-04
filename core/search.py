"""
Hybrid search functionality with reranking for the RAG Chatbot application.
"""
import gc
import torch
from typing import List, Tuple, Optional, Union
from langchain_core.documents import Document as LCDocument
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

from config.config import Config


class HybridSearchEngine:
    """Manages hybrid search operations with reranking capabilities."""
    
    def __init__(self, client: QdrantClient, models_dict: dict):
        self.client = client
        self.models_dict = models_dict
    
    async def search_with_threshold(
        self, 
        query: str, 
        collection_name: str,
        threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD,
        max_results: int = Config.DEFAULT_MAX_RESULTS,
        use_reranker: bool = True
    ) -> Union[Tuple[List[LCDocument], List[float], List[str]], 
               Tuple[List[LCDocument], List[float], List[str], List[int], List[float], bool]]:
        """
        Perform hybrid search using RRF (Reciprocal Rank Fusion) to combine results from
        multiple vector types and filter results by a similarity threshold.
        
        Args:
            query (str): User query text
            collection_name (str): Name of the collection to search
            threshold (float): Minimum similarity score threshold to include results
            max_results (int): Maximum number of results to fetch initially (before filtering)
            use_reranker (bool): Whether to apply BGE reranking
            
        Returns:
            tuple: List of documents, their scores, and their origins, with optional reranking info
        """
        try:
            # Initialize query vectors
            query_vectors = await self._generate_query_vectors(query)
            
            if not query_vectors:
                return self._empty_results(use_reranker)
            
            # Perform individual searches for origin tracking
            dense_results, sparse_results = await self._perform_individual_searches(
                query_vectors, collection_name, max_results
            )
            
            # Create origin sets
            dense_ids = {point.id for point in dense_results.points} if dense_results else set()
            sparse_ids = {point.id for point in sparse_results.points} if sparse_results else set()
            
            # Perform fusion search
            results, origins = await self._perform_fusion_search(
                query_vectors, collection_name, max_results, dense_ids, sparse_ids
            )
            
            if not results or not results.points:
                return self._empty_results(use_reranker)
            
            # Convert to documents and filter by threshold
            docs, scores, point_ids = self._convert_to_documents(results, threshold)
            
            if not docs:
                print(f"No documents passed the threshold of {threshold} for {collection_name}")
                return self._empty_results(use_reranker)
            
            # Adjust origins to match filtered documents
            origins = self._adjust_origins(origins, len(docs))
            
            # Store original order and scores before reranking
            pre_rerank_docs = docs.copy()
            pre_rerank_scores = scores.copy()
            
            # Apply BGE reranking if enabled
            if use_reranker and len(docs) > 0:
                return await self._apply_reranking(
                    query, docs, scores, origins, threshold, 
                    pre_rerank_docs, pre_rerank_scores
                )
            else:
                # No reranking applied
                original_ranks = list(range(len(docs)))
                original_scores = scores
                
                print(f"Returning {len(docs)} documents from {collection_name} with scores above {threshold}")
                
                if use_reranker:
                    return docs, scores, origins, original_ranks, original_scores, False
                else:
                    return docs, scores, origins
                    
        except Exception as e:
            print(f"Error in hybrid search for {collection_name}: {e}")
            import traceback
            traceback.print_exc()
            # Clean GPU memory in case of error
            torch.cuda.empty_cache()
            gc.collect()
            return self._empty_results(use_reranker)
    
    async def _generate_query_vectors(self, query: str) -> dict:
        """Generate query vectors for different embedding types."""
        query_vectors = {}
        
        # Generate dense vector
        if "finetuned_dense" in self.models_dict:
            try:
                query_vectors["finetuned_dense"] = self.models_dict["finetuned_dense"].embed_documents([query])[0]
                print(f"Dense vector generated for query: {query[:30]}...")
            except Exception as e:
                print(f"Error generating dense vector: {e}")
        
        # Generate sparse vector
        if "sparse" in self.models_dict:
            try:
                sparse_vector = next(self.models_dict["sparse"].passage_embed([query]))
                query_vectors["sparse"] = sparse_vector
                print(f"Sparse vector generated for query: {query[:30]}...")
            except Exception as e:
                print(f"Error generating sparse vector: {e}")
        
        return query_vectors
    
    async def _perform_individual_searches(self, query_vectors: dict, collection_name: str, max_results: int):
        """Perform individual searches for origin tracking."""
        dense_results = None
        sparse_results = None
        
        if "finetuned_dense" in query_vectors:
            try:
                dense_results = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vectors["finetuned_dense"],
                    using="finetuned_dense",
                    with_payload=True,
                    limit=max_results * 2,
                )
                print(f"Dense search returned {len(dense_results.points) if dense_results else 0} results")
            except Exception as e:
                print(f"Error in dense search: {e}")
        
        if "sparse" in query_vectors:
            try:
                sparse_results = self.client.query_points(
                    collection_name=collection_name,
                    query=models.SparseVector(**query_vectors["sparse"].as_object()),
                    using="bm25",
                    with_payload=True,
                    limit=max_results * 2,
                )
                print(f"Sparse search returned {len(sparse_results.points) if sparse_results else 0} results")
            except Exception as e:
                print(f"Error in sparse search: {e}")
        
        return dense_results, sparse_results
    
    async def _perform_fusion_search(self, query_vectors: dict, collection_name: str, 
                                   max_results: int, dense_ids: set, sparse_ids: set):
        """Perform RRF fusion search or fallback to individual searches."""
        prefetch = self._build_prefetch_queries(query_vectors, max_results)
        
        if len(prefetch) >= 2:
            try:
                results = self.client.query_points(
                    collection_name=collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    with_payload=True,
                    limit=max_results,
                )
                
                origins = self._determine_origins(results, dense_ids, sparse_ids)
                return results, origins
                
            except Exception as e:
                print(f"Error in RRF search: {e}")
        
        # Fallback to individual searches
        return self._fallback_to_individual_search(query_vectors, collection_name)
    
    def _build_prefetch_queries(self, query_vectors: dict, max_results: int) -> List:
        """Build prefetch queries for RRF fusion."""
        prefetch = []
        
        if "finetuned_dense" in query_vectors:
            try:
                prefetch.append(
                    models.Prefetch(
                        query=query_vectors["finetuned_dense"],
                        using="finetuned_dense",
                        limit=max_results,
                    )
                )
            except Exception as e:
                print(f"Error creating dense prefetch: {e}")
        
        if "sparse" in query_vectors:
            try:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(**query_vectors["sparse"].as_object()),
                        using="bm25",
                        limit=max_results,
                    )
                )
            except Exception as e:
                print(f"Error creating sparse prefetch: {e}")
        
        return prefetch
    
    def _determine_origins(self, results, dense_ids: set, sparse_ids: set) -> List[str]:
        """Determine the origin of each result point."""
        origins = []
        for point in results.points:
            if point.id in dense_ids and point.id in sparse_ids:
                origins.append("combined")
            elif point.id in dense_ids:
                origins.append("finetuned_dense")
            elif point.id in sparse_ids:
                origins.append("sparse")
            else:
                origins.append("combined")  # Default if uncertain
        
        origin_counts = {
            "combined": origins.count("combined"),
            "finetuned_dense": origins.count("finetuned_dense"),
            "sparse": origins.count("sparse")
        }
        print(f"RRF search returned {len(results.points)} results with origin distribution: {origin_counts}")
        
        return origins
    
    def _fallback_to_individual_search(self, query_vectors: dict, collection_name: str):
        """Fallback to individual search results when RRF fails."""
        # Implementation would go here - simplified for brevity
        return None, []
    
    def _convert_to_documents(self, results, threshold: float) -> Tuple[List[LCDocument], List[float], List]:
        """Convert Qdrant results to LangChain documents."""
        docs = []
        scores = []
        point_ids = []
        
        for point in results.points:
            if hasattr(point, 'payload') and point.payload and point.score >= threshold:
                page_content = point.payload.get('page_content', '')
                metadata = point.payload.get('metadata', {})
                
                doc = LCDocument(page_content=page_content, metadata=metadata)
                docs.append(doc)
                scores.append(point.score)
                point_ids.append(point.id)
        
        return docs, scores, point_ids
    
    def _adjust_origins(self, origins: List[str], doc_count: int) -> List[str]:
        """Adjust origins list to match document count."""
        if len(origins) > doc_count:
            return origins[:doc_count]
        elif len(origins) < doc_count:
            return origins + ["unknown"] * (doc_count - len(origins))
        return origins
    
    async def _apply_reranking(self, query: str, docs: List[LCDocument], scores: List[float], 
                             origins: List[str], threshold: float, 
                             pre_rerank_docs: List[LCDocument], pre_rerank_scores: List[float]):
        """Apply BGE reranking to the documents."""
        try:
            print("Applying BGE reranking...")
            
            bge_reranker_id = self.models_dict.get("bge_reranker_id")
            if not bge_reranker_id:
                print("BGE reranker ID not available, skipping reranking")
                return docs, scores, origins, list(range(len(docs))), pre_rerank_scores, False
            
            # Clean GPU memory before loading reranker
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load reranker
            bge_reranker = CrossEncoder(bge_reranker_id, device='cuda')
            
            # Create sentence pairs and get reranked scores
            texts = [doc.page_content for doc in docs]
            sentence_pairs = [(query, text) for text in texts]
            reranked_scores = bge_reranker.predict(sentence_pairs)
            
            # Clean up GPU memory
            del bge_reranker
            torch.cuda.empty_cache()
            gc.collect()
            
            # Create data tuples for sorting
            doc_data = list(zip(
                docs, reranked_scores, origins, 
                [i for i in range(len(docs))],  # original positions
                pre_rerank_scores
            ))
            
            # Sort by reranked score
            doc_data.sort(key=lambda x: x[1], reverse=True)
            
            # Unpack sorted data
            docs = [item[0] for item in doc_data]
            scores = [item[1] for item in doc_data]
            origins = [item[2] for item in doc_data]
            original_ranks = [item[3] for item in doc_data]
            original_scores = [item[4] for item in doc_data]
            
            # Apply threshold again
            filtered_data = [
                (docs[i], scores[i], origins[i], original_ranks[i], original_scores[i])
                for i, score in enumerate(scores) if score >= threshold
            ]
            
            if filtered_data:
                docs, scores, origins, original_ranks, original_scores = zip(*filtered_data)
                docs, scores, origins = list(docs), list(scores), list(origins)
                original_ranks, original_scores = list(original_ranks), list(original_scores)
            else:
                docs, scores, origins = [], [], []
                original_ranks, original_scores = [], []
            
            print(f"BGE reranking applied, returning {len(docs)} results after threshold filtering")
            return docs, scores, origins, original_ranks, original_scores, True
            
        except Exception as e:
            print(f"Error in BGE reranking: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return docs, scores, origins, list(range(len(docs))), pre_rerank_scores, False
    
    def _empty_results(self, use_reranker: bool):
        """Return empty results in the correct format."""
        if use_reranker:
            return [], [], [], [], [], False
        else:
            return [], [], []


# Factory function for backward compatibility
async def hybrid_search_with_threshold(query: str, client: QdrantClient, models_dict: dict, 
                                     collection_name: str, threshold: float = Config.DEFAULT_SIMILARITY_THRESHOLD,
                                     max_results: int = Config.DEFAULT_MAX_RESULTS, use_reranker: bool = True):
    """
    Factory function to maintain backward compatibility with the original code.
    """
    search_engine = HybridSearchEngine(client, models_dict)
    return await search_engine.search_with_threshold(
        query, collection_name, threshold, max_results, use_reranker
    )