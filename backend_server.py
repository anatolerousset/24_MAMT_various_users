"""
Backend FastAPI server.
"""
import os
import asyncio
import uvicorn
import json
from typing import List, Optional, Dict, Any, Set, Tuple, Union, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import numpy as np

# Import your core modules
from config.config import Config
from core.embeddings import get_embedding_models
from core.llm import initialize_llm
from core.dual_search import DualSearchService
from utils.document_processing import format_docs
from utils.export_docx import get_backend_docx_exporter
from utils.blob_utils import BlobStorageManager
from utils.download_manager_image import (
    get_image_download_manager, 
    cleanup_previous_images, 
    process_chunks_with_images
)
from qdrant_client import QdrantClient

# Backend-compatible response generation
from core.response_generation import (
    generate_llm_response,
    generate_streaming_llm_response,
    BackendResponseGenerator,
    validate_response_quality
)

from main_ingestion import run_ingestion_async

import logging
_log = logging.getLogger(__name__)

# Global variables
qdrant_client = None
embedding_models = None
llm = None
blob_manager = None
image_manager = None
backend_docx_exporter = None
dual_search_service = None

def convert_numpy_types(obj):
    """Convert numpy types to native Python types recursively with better error handling."""
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
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): convert_numpy_types(value) for key, value in obj.items()}
        else:
            # For any other type, try to convert to string as fallback
            try:
                return str(obj)
            except Exception:
                return None
    except Exception as e:
        print(f"WARNING: Error converting {type(obj)}: {e}")
        return str(obj) if obj is not None else None

def make_json_safe(obj):
    """Make any object JSON-serializable by converting problematic types."""
    try:
        # First try normal JSON encoding
        import json
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # If that fails, convert types
        return convert_numpy_types(obj)

def initialize_models():
    """Initialize ML models and connect to external services"""
    global embedding_models, llm, qdrant_client, blob_manager, image_manager, backend_docx_exporter, dual_search_service
    try:
        print("Connecting to external services and initializing models...")
        
        # Connect to external Qdrant client
        qdrant_client = QdrantClient(url=Config.QDRANT_URL)
        
        # Test connection
        try:
            collections = qdrant_client.get_collections()
            print(f"✓ Connected to Qdrant. Found {len(collections.collections)} collections")
        except Exception as e:
            print(f"⚠️ Warning: Could not connect to Qdrant at {Config.QDRANT_URL}: {e}")
        
        # Initialize embedding models
        embedding_models = get_embedding_models()
        
        # Initialize LLM
        llm = initialize_llm()
        
        # Initialize blob storage manager
        blob_manager = BlobStorageManager()
        
        # Initialize image download manager
        image_manager = get_image_download_manager()
        
        # Initialize backend DOCX exporter (no Chainlit dependency)
        backend_docx_exporter = get_backend_docx_exporter("exports")
        
        # Initialize dual search service
        dual_search_service = DualSearchService(qdrant_client, embedding_models, image_manager)
        
        # Cleanup previous images on startup
        try:
            cleanup_previous_images()
            print("✓ Previous images cleaned up")
        except Exception as e:
            print(f"Warning: Could not cleanup previous images: {e}")
        
        print("✓ All models and services initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    # Startup
    print("Starting backend services...")
    initialize_models()
    print("✓ Backend services started")
    
    yield
    
    # Shutdown
    print("Shutting down backend services...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Simplified RAG Backend Server", 
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
app.mount("/images", StaticFiles(directory="public"), name="images")

# Pydantic models for API (keeping your existing models)
class SearchRequest(BaseModel):
    query: str
    collection_name: str
    threshold: float = 0.1
    max_results: int = 15
    use_reranker: bool = True

class SearchResponse(BaseModel):
    documents: List[Dict[str, Any]]
    scores: List[float]
    origins: List[str]
    original_ranks: Optional[List[int]] = None
    original_scores: Optional[List[float]] = None
    reranked: bool = False
    total_results: int
    collection_name: str
    
    class Config:
        arbitrary_types_allowed = False
        json_encoders = {
            set: list,
            tuple: list,
        }

class DualSearchRequest(BaseModel):
    query: str
    technical_collection: str = None
    dce_collection: str = None
    threshold: float = 0.1
    max_results: int = 15
    use_reranker: bool = True

class DualSearchResponse(BaseModel):
    technical_results: SearchResponse
    dce_results: SearchResponse
    total_documents: int
    
    class Config:
        arbitrary_types_allowed = False

class ChatRequest(BaseModel):
    query: str
    technical_docs: List[Dict[str, Any]] = []
    dce_docs: List[Dict[str, Any]] = []
    user_specifications: Optional[str] = None
    use_both: bool = True
    use_technical: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: List[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    generation_info: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = False
        json_encoders = {
            set: list,
            tuple: list,
        }

class IngestionRequest(BaseModel):
    data_type: str
    region_name: Optional[str] = None
    office_name: Optional[str] = None  # NEW: Added office_name field
    collection_name: Optional[str] = None
    recreate_collection: bool = False
    remove_duplicates: bool = False
    file_patterns: Optional[List[str]] = None
    archive_processed_files: Optional[bool] = None

class ExportRequest(BaseModel):
    content: str
    query: str
    sources: List[Tuple[str, Union[str, int]]]
    technical_docs: Optional[List[Dict[str, Any]]] = []
    technical_scores: Optional[List[float]] = []
    technical_origins: Optional[List[str]] = []
    dce_docs: Optional[List[Dict[str, Any]]] = []
    dce_scores: Optional[List[float]] = []
    dce_origins: Optional[List[str]] = []
    include_retrieved_docs: bool = False
    include_images_catalog: bool = False
    filename: Optional[str] = None
    
    @field_validator('sources', mode='before')
    @classmethod
    def normalize_sources(cls, v):
        """Normaliser les sources pour gérer différents formats de page"""
        if not v:
            return []
        
        normalized_sources = []
        for source_item in v:
            if isinstance(source_item, (list, tuple)) and len(source_item) >= 2:
                source_name = str(source_item[0])
                page_value = source_item[1]
                
                # Si c'est une chaîne 'N/A', la garder telle quelle
                # Sinon essayer de convertir en entier
                if isinstance(page_value, str):
                    if page_value.upper() in ['N/A', 'NA', 'UNKNOWN', '']:
                        final_page = 'N/A'
                    else:
                        try:
                            final_page = int(page_value)
                        except ValueError:
                            final_page = 'N/A'
                else:
                    try:
                        final_page = int(page_value) if page_value is not None else 'N/A'
                    except (ValueError, TypeError):
                        final_page = 'N/A'
                
                normalized_sources.append((source_name, final_page))
            else:
                # Source malformée, créer une entrée par défaut
                normalized_sources.append((str(source_item), 'N/A'))
        
        return normalized_sources

class BlobExportRequest(BaseModel):
    content: str
    query: str
    sources: List[Tuple[str, Union[str, int]]]
    metadata: Optional[Dict[str, Any]] = {}
    
    @field_validator('sources', mode='before')
    @classmethod
    def normalize_sources(cls, v):
        """Normaliser les sources pour gérer différents formats de page"""
        if not v:
            return []
        
        normalized_sources = []
        for source_item in v:
            if isinstance(source_item, (list, tuple)) and len(source_item) >= 2:
                source_name = str(source_item[0])
                page_value = source_item[1]
                
                if isinstance(page_value, str):
                    if page_value.upper() in ['N/A', 'NA', 'UNKNOWN', '']:
                        final_page = 'N/A'
                    else:
                        try:
                            final_page = int(page_value)
                        except ValueError:
                            final_page = 'N/A'
                else:
                    try:
                        final_page = int(page_value) if page_value is not None else 'N/A'
                    except (ValueError, TypeError):
                        final_page = 'N/A'
                
                normalized_sources.append((source_name, final_page))
            else:
                normalized_sources.append((str(source_item), 'N/A'))
        
        return normalized_sources

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Backend Server with Image Serving",
        "version": "2.1.0",
        "qdrant_url": Config.QDRANT_URL,
        "features": [
            "Dual collection search",
            "Image processing and serving",
            "Blob storage integration", 
            "DOCX export",
            "LLM response generation",
            "HTTP image serving at /images/",
            "Ingestion without progress tracking"
        ],
        "endpoints": {
            "health": "/health",
            "dual_search": "/api/dual-search",
            "search": "/api/search",
            "chat": "/api/chat",
            "ingestion": "/api/ingestion",
            "collections": "/api/collections",
            "export": "/api/export",
            "blob": "/api/blob",
            "images": "/images/{filename}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    services_status = {}
    
    # Check Qdrant
    try:
        if qdrant_client:
            collections = qdrant_client.get_collections()
            services_status["qdrant"] = "healthy"
            services_status["qdrant_collections"] = len(collections.collections)
        else:
            services_status["qdrant"] = "not_connected"
    except Exception as e:
        services_status["qdrant"] = f"error: {str(e)}"
    
    # Check models
    services_status["embedding_models"] = "healthy" if embedding_models else "not_loaded"
    services_status["llm"] = "healthy" if llm else "not_loaded"
    services_status["blob_manager"] = "healthy" if blob_manager else "not_loaded"
    services_status["image_manager"] = "healthy" if image_manager else "not_loaded"
    services_status["backend_docx_exporter"] = "healthy" if backend_docx_exporter else "not_loaded"
    
    # Check blob storage connection
    if blob_manager:
        try:
            containers = blob_manager.blob_service_client.list_containers()
            list(containers)  # Force evaluation
            services_status["blob_storage"] = "healthy"
        except Exception as e:
            services_status["blob_storage"] = f"error: {str(e)}"
    
    # Check image serving
    try:
        from pathlib import Path
        public_path = Path("public")
        if public_path.exists():
            image_count = len(list(public_path.glob("*")))
            services_status["image_serving"] = "healthy"
            services_status["available_images"] = image_count
        else:
            services_status["image_serving"] = "public_folder_missing"
            services_status["available_images"] = 0
    except Exception as e:
        services_status["image_serving"] = f"error: {str(e)}"
        services_status["available_images"] = 0
    
    return {
        "status": "healthy" if services_status.get("qdrant") == "healthy" else "degraded",
        "services": services_status,
        "qdrant_url": Config.QDRANT_URL,
        "config": {
            "technical_collection": Config.COLLECTION_NAME,
            "dce_collection": Config.DCE_COLLECTION,
            "llm_provider": Config.get_active_llm_provider(),
            "default_threshold": Config.DEFAULT_SIMILARITY_THRESHOLD,
            "default_max_results": Config.DEFAULT_MAX_RESULTS,
            "image_serving_enabled": True,
            "public_folder": str(Path("public").absolute())
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/ingestion")
async def start_ingestion(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    UPDATED: Basic ingestion endpoint using main_ingestion.py with office support
    """
    try:
        # Run ingestion in background using main_ingestion.py
        background_tasks.add_task(
            run_ingestion_async,
            data_type=request.data_type,
            region_name=request.region_name,
            office_name=request.office_name,  # NEW: Pass office_name to ingestion
            collection_name=request.collection_name,
            recreate_collection=request.recreate_collection,
            remove_duplicates=request.remove_duplicates,
            file_patterns=request.file_patterns,
            archive_processed_files=request.archive_processed_files
        )
        
        # Build response message with office information
        message_parts = [f"Ingestion démarrée en arrière-plan pour {request.data_type}"]
        
        if request.region_name:
            message_parts.append(f"région: {request.region_name}")
        
        if request.office_name:
            message_parts.append(f"bureau: {request.office_name}")  # NEW: Include office in response
        
        response_message = " - ".join(message_parts)
        
        return {
            "success": True,
            "message": response_message,
            "data_type": request.data_type,
            "region_name": request.region_name,
            "office_name": request.office_name,  # NEW: Include office_name in response
            "collection_name": request.collection_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dual-search", response_model=DualSearchResponse)
async def dual_search_documents(request: DualSearchRequest):
    """Search documents in both technical and DCE collections with image processing"""
    if not dual_search_service:
        raise HTTPException(status_code=503, detail="Dual search service not ready")
    
    try:
        # Perform dual search using the service
        search_results = await dual_search_service.dual_search(
            query=request.query,
            technical_collection=request.technical_collection,
            dce_collection=request.dce_collection,
            threshold=request.threshold,
            max_results=request.max_results,
            use_reranker=request.use_reranker,
            process_images=True  # Always process images
        )
        
        # Convert service results to API response format
        technical_results = SearchResponse(**search_results['technical_results'])
        dce_results = SearchResponse(**search_results['dce_results'])
        
        return DualSearchResponse(
            technical_results=technical_results,
            dce_results=dce_results,
            total_documents=search_results['total_documents']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def stream_chat_response(request: ChatRequest):
    """Generate streaming chat response using Server-Sent Events"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not ready")
    
    async def generate_stream():
        try:
            # Convert document dictionaries back to LangChain Document objects
            from langchain_core.documents import Document as LCDocument
            
            technical_docs = []
            for doc_data in request.technical_docs:
                doc = LCDocument(
                    page_content=doc_data.get("page_content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                technical_docs.append(doc)
            
            dce_docs = []
            for doc_data in request.dce_docs:
                doc = LCDocument(
                    page_content=doc_data.get("page_content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                dce_docs.append(doc)
            
            # Determine collections to use
            collections_used = []
            if request.use_both and technical_docs and dce_docs:
                collections_used = [Config.COLLECTION_NAME, Config.DCE_COLLECTION]
            elif request.use_technical and technical_docs:
                collections_used = [Config.COLLECTION_NAME]
                dce_docs = []
            elif not request.use_technical and dce_docs:
                collections_used = [Config.DCE_COLLECTION]
                technical_docs = []
            else:
                yield {
                    "data": json.dumps({
                        "type": "error",
                        "content": "No valid documents available for selected collections"
                    })
                }
                return
            
            # Send initial metadata
            yield {
                "data": json.dumps({
                    "type": "start",
                    "metadata": {
                        "collections_used": collections_used,
                        "technical_docs_count": len(technical_docs),
                        "dce_docs_count": len(dce_docs),
                        "query": request.query
                    }
                })
            }
            
            # Generate streaming response using existing function
            async for chunk in generate_streaming_llm_response(
                query=request.query,
                technical_docs=technical_docs,
                dce_docs=dce_docs,
                user_specifications=request.user_specifications,
                use_both=request.use_both,
                use_technical=request.use_technical,
                llm=llm
            ):
                yield {"data": json.dumps(chunk)}
            
        except Exception as e:
            yield {
                "data": json.dumps({
                    "type": "error",
                    "content": f"Streaming error: {str(e)}"
                })
            }
    
    return EventSourceResponse(generate_stream())

# Also add this alternative endpoint for regular streaming without SSE
@app.post("/api/chat/stream-text")
async def stream_chat_text(request: ChatRequest):
    """Generate streaming chat response as plain text stream"""
    if not llm:
        raise HTTPException(status_code=503, detail="LLM not ready")
    
    async def generate_text_stream():
        try:
            # Same document preparation logic as above
            from langchain_core.documents import Document as LCDocument
            
            technical_docs = []
            for doc_data in request.technical_docs:
                doc = LCDocument(
                    page_content=doc_data.get("page_content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                technical_docs.append(doc)
            
            dce_docs = []
            for doc_data in request.dce_docs:
                doc = LCDocument(
                    page_content=doc_data.get("page_content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                dce_docs.append(doc)
            
            # Filter docs based on selection
            if not request.use_both:
                if request.use_technical:
                    dce_docs = []
                else:
                    technical_docs = []
            
            # Generate streaming response
            async for chunk in generate_streaming_llm_response(
                query=request.query,
                technical_docs=technical_docs,
                dce_docs=dce_docs,
                user_specifications=request.user_specifications,
                use_both=request.use_both,
                use_technical=request.use_technical,
                llm=llm
            ):
                if chunk.get("type") == "chunk":
                    yield chunk.get("content", "")
                elif chunk.get("type") == "complete":
                    # Send final newline to indicate completion
                    yield "\n\n[STREAM_COMPLETE]"
                elif chunk.get("type") == "error":
                    yield f"\n\n[ERROR: {chunk.get('content', 'Unknown error')}]"
            
        except Exception as e:
            yield f"\n\n[ERROR: {str(e)}]"
    
    return StreamingResponse(
        generate_text_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/api/export/docx/direct")
async def export_docx_direct_download(request: ExportRequest):
    """Export and return DOCX file directly for download with auto-archive"""
    if not backend_docx_exporter:
        raise HTTPException(status_code=503, detail="DOCX exporter not ready")
    
    try:
        _log.info(f"DEBUG: request technical : {request.technical_docs}")
        # Même logique de conversion des documents
        technical_docs = []
        if request.technical_docs:
            from langchain_core.documents import Document as LCDocument
            for doc_data in request.technical_docs:
                doc = LCDocument(
                    page_content=doc_data.get("page_content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                technical_docs.append(doc)
        
        dce_docs = []
        if request.dce_docs:
            from langchain_core.documents import Document as LCDocument
            for doc_data in request.dce_docs:
                doc = LCDocument(
                    page_content=doc_data.get("page_content", ""),
                    metadata=doc_data.get("metadata", {})
                )
                dce_docs.append(doc)
        
        # Generate filename
        if not request.filename:
            if not request.include_retrieved_docs:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                sanitized_query = _sanitize_filename(request.query)
                filename = f"Paragraphe_{sanitized_query}_{timestamp}.docx"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                sanitized_query = _sanitize_filename(request.query)
                filename = f"ParagrapheEtSources_{sanitized_query}_{timestamp}.docx"
        else:
            filename = request.filename
            if not filename.endswith('.docx'):
                filename += '.docx'
        
        _log.info(f"Paragraphes récupérés: {technical_docs}")

        # Export to bytes
        export_result = backend_docx_exporter.export_to_bytes(
            markdown_content=request.content,
            query=request.query,
            sources=set(request.sources),
            technical_docs=technical_docs, #if request.include_retrieved_docs else None,
            technical_scores=request.technical_scores, #if request.include_retrieved_docs else None,
            technical_origins=request.technical_origins, #if request.include_retrieved_docs else None,
            dce_docs=dce_docs, #if request.include_retrieved_docs else None,
            dce_scores=request.dce_scores, #if request.include_retrieved_docs else None,
            dce_origins=request.dce_origins, #if request.include_retrieved_docs else None,
            collection_name=Config.COLLECTION_NAME,
            dce_collection=Config.DCE_COLLECTION,
            include_retrieved_docs=request.include_retrieved_docs,
            include_images_catalog=request.include_images_catalog,
            filename=filename
        )
        
        if export_result.get("success"):
            file_bytes = export_result.get("file_bytes")
            
            # AUTO-ARCHIVE: Upload to blob storage
            try:
                if blob_manager:
                    # Create blob path with timestamp folder structure
                    timestamp_folder = datetime.now().strftime("%Y/%m")
                    blob_name = f"docx_responses/{timestamp_folder}/{filename}"
                    
                    # Upload to exports container
                    blob_manager.upload_data(
                        data=file_bytes,
                        container_name=Config.EXPORTS_CONTAINER,
                        blob_name=blob_name,
                        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        metadata={
                            "query": request.query[:100],  # Truncated for metadata limits
                            "export_timestamp": datetime.now().isoformat(),
                            "include_retrieved_docs": str(request.include_retrieved_docs),
                            "include_images_catalog": str(request.include_images_catalog),
                            "source": "chainlit_export",
                            "file_size": str(len(file_bytes))
                        }
                    )
                    _log.info(f"✓ Auto-archived DOCX to blob: {Config.EXPORTS_CONTAINER}/{blob_name}")
                else:
                    _log.warning("Blob manager not available for auto-archiving")
            except Exception as e:
                _log.error(f"Failed to auto-archive DOCX to blob: {e}")
                # Don't fail the export if archiving fails
            
            # Return file for download
            def generate_file():
                yield file_bytes
            
            return StreamingResponse(
                generate_file(),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Length": str(len(file_bytes))
                }
            )
        else:
            raise HTTPException(status_code=500, detail=export_result.get("error", "Export failed"))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@app.post("/api/blob/export")
async def export_to_blob(request: BlobExportRequest):
    """Export data to Azure Blob Storage"""
    if not blob_manager:
        raise HTTPException(status_code=503, detail="Blob storage not ready")
    
    try:
        # Create export data
        export_data = {
            "content": request.content,
            "query": request.query,
            "sources": request.sources,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "export_type": "api_export",
                "user_session": "backend_api",
                "export_date": datetime.now().isoformat(),
                **request.metadata
            }
        }
        
        # Generate blob name
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_query = _sanitize_filename(request.query)
        blob_name = f"api_responses/{timestamp_str}_{sanitized_query}.json"
        
        # Upload to blob storage
        blob_manager.upload_data(
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            container_name=Config.EXPORTS_CONTAINER,
            blob_name=blob_name,
            content_type="application/json",
            metadata={
                "query": request.query[:100],  # Truncate for metadata
                "export_timestamp": datetime.now().isoformat(),
                "source": "backend_api"
            }
        )
        
        return {
            "success": True,
            "message": "Data exported to blob storage successfully",
            "blob_name": blob_name,
            "container": Config.EXPORTS_CONTAINER,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blob export error: {str(e)}")

@app.get("/api/blob/list")
async def list_blob_exports():
    """List available exports in blob storage"""
    if not blob_manager:
        raise HTTPException(status_code=503, detail="Blob storage not ready")
    
    try:
        blobs = blob_manager.list_blobs(Config.EXPORTS_CONTAINER)
        
        exports_list = []
        for blob in sorted(blobs, key=lambda x: x.get('last_modified', datetime.min), reverse=True)[:50]:
            exports_list.append({
                "name": blob['name'],
                "size": blob.get('size', 0),
                "last_modified": blob.get('last_modified').isoformat() if blob.get('last_modified') else None,
                "metadata": blob.get('metadata', {})
            })
        
        return {
            "success": True,
            "exports": exports_list,
            "total_count": len(exports_list),
            "container": Config.EXPORTS_CONTAINER
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List blobs error: {str(e)}")

@app.get("/api/blob/download/{blob_name:path}")
async def download_from_blob(blob_name: str):
    """Download a specific export from blob storage"""
    if not blob_manager:
        raise HTTPException(status_code=503, detail="Blob storage not ready")
    
    try:
        # Download blob data
        blob_data = blob_manager.download_data(Config.EXPORTS_CONTAINER, blob_name)
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(blob_data.encode('utf-8') if isinstance(blob_data, str) else blob_data)
            temp_file_path = temp_file.name
        
        # Return file response
        filename = blob_name.split('/')[-1]  # Get just the filename
        return FileResponse(
            path=temp_file_path,
            filename=filename,
            media_type="application/json"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

@app.get("/api/collections")
async def list_collections():
    """List available collections"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not ready")
    
    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [collection.name for collection in collections.collections]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/collection/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Get information about a specific collection"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not ready")
    
    try:
        if not qdrant_client.collection_exists(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found")
        
        info = qdrant_client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "config": {
                "distance": info.config.params.vectors.distance.name if info.config.params.vectors else None
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "collections": {
            "technical": Config.COLLECTION_NAME,
            "dce": Config.DCE_COLLECTION
        },
        "search": {
            "default_threshold": Config.DEFAULT_SIMILARITY_THRESHOLD,
            "default_max_results": Config.DEFAULT_MAX_RESULTS
        },
        "llm": {
            "provider": Config.get_active_llm_provider()
        },
        "services": {
            "qdrant_url": Config.QDRANT_URL,
            "blob_storage": Config.EXPORTS_CONTAINER if hasattr(Config, 'EXPORTS_CONTAINER') else None
        }
    }


@app.post("/api/cleanup-images")
async def cleanup_images():
    """Cleanup previous images"""
    try:
        if image_manager:
            cleanup_previous_images()
            return {
                "success": True,
                "message": "Previous images cleaned up successfully"
            }
        else:
            raise HTTPException(status_code=503, detail="Image manager not ready")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

# Helper functions
def _sanitize_filename(text: str, max_length: int = 50) -> str:
    """Create a safe filename from text."""
    import re
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = sanitized.strip('._')
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('._')
    
    return sanitized or "document"

from fastapi.encoders import jsonable_encoder

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle any unhandled exceptions and return proper JSON response"""
    print(f"GLOBAL EXCEPTION: {type(exc).__name__}: {str(exc)}")
    import traceback
    traceback.print_exc()
    
    # Check if it's a JSON serialization error
    if "not JSON serializable" in str(exc):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Response serialization error",
                "error_type": "json_serialization",
                "message": "The server response contains data that cannot be converted to JSON"
            }
        )
    
    # General error response
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__,
            "message": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "backend_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        access_log=True
    )