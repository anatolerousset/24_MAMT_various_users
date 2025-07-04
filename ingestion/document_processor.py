"""
Document processor with better individual file tracking
"""

import logging
import sys
import threading
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple, Callable
from tqdm import tqdm
from time import sleep

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, WordFormatOption, ExcelFormatOption, PdfFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.settings import settings
from docling_core.types.doc import ImageRefMode
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from langchain_core.documents import Document as LCDocument

from parser_and_chunker.msword_backend_modif_metadataplusv250408 import (
    MsWordDocumentBackendModMetadataplus, 
    DocumentOriginMod
)
from parser_and_chunker.hybrid_chunker_extendedv250408 import HybridChunkerWithImages
from utils.file_utils import convert_xls_to_xlsx, format_headings, fix_image_paths
from utils.blob_utils import BlobStorageManager
from config.config import Config

_log = logging.getLogger(__name__)

class DocumentProcessor:
    """Document processor with detailed progress tracking and user support."""
    
    def __init__(self, tokenizer_model: str = None, max_tokens: int = None, 
                 progress_callback: Callable = None, user_session_id: str = None):
        """
        Initialize the document processor with user support.
        
        Args:
            tokenizer_model: Model name for tokenization
            max_tokens: Maximum tokens per chunk
            progress_callback: Callback function for progress updates
            user_session_id: User session ID for user-specific processing
        """
        self.tokenizer_model = tokenizer_model or Config.FINETUNED_DENSE_EMBED_MODEL_ID
        self.max_tokens = max_tokens or Config.DEFAULT_MAX_TOKENS
        self.blob_manager = BlobStorageManager()
        self.progress_callback = progress_callback or self._default_progress_callback
        self.user_session_id = user_session_id
        
        # Track processing results per file
        self.processing_results = {}
        self.successful_files = []
        self.failed_files = []
        
        # Initialize converters
        self._init_converters()
        
        # Initialize chunker
        self.chunker = None
    
    async def process_files_from_blob_with_progress_user(self, 
                                                        blob_names: List[str], 
                                                        data_type: str,
                                                        region_name: Optional[str] = None,
                                                        input_container: str = None,
                                                        user_session_id: str = None) -> Tuple[List[LCDocument], List[str]]:
        """
        Process files from blob storage with user-specific paths and progress tracking.
        
        Args:
            blob_names: List of blob names to process
            data_type: Type of data being processed ('dce' or 'region')
            region_name: Name of the region (if processing regional data)
            input_container: Container containing input files
            user_session_id: User session ID for user-specific processing
            
        Returns:
            Tuple of (all_chunks, all_texts)
        """
        input_container = input_container or Config.INPUT_CONTAINER
        user_session_id = user_session_id or self.user_session_id
        
        # Reset tracking variables
        self.processing_results = {}
        self.successful_files = []
        self.failed_files = []
        
        self.progress_callback(
            current_step="Starting document processing",
            step_progress=0,
            log_message=f"Processing {len(blob_names)} documents from blob storage for {data_type} (User: {user_session_id})..."
        )
        
        # Get user-specific temporary paths for processing
        if user_session_id:
            temp_paths = Config.get_user_temp_paths(user_session_id, data_type, region_name)
        else:
            temp_paths = Config.get_temp_paths(data_type, region_name)
        
        # Initialize chunker with user-specific temporary image directory
        self._init_chunker(temp_paths["images"])
        
        # Choose appropriate converter
        converter = self.dce_converter if data_type == 'dce' else self.region_converter
        
        # Process documents
        all_chunks = []
        all_texts = []
        
        try:
            for i, blob_name in enumerate(blob_names):
                file_success = False
                file_start_time = time.time()
                
                try:
                    # Update progress for current file
                    self.progress_callback(
                        current_file=blob_name,
                        files_processed=i,
                        overall_progress=15 + (i / len(blob_names)) * 50,  # 15-65% for processing
                        log_message=f"Processing file {i+1}/{len(blob_names)}: {blob_name} (User: {user_session_id})"
                    )
                    
                    # Process single file with user-specific paths
                    chunks, texts = await self._process_single_blob_with_progress_user(
                        blob_name, input_container, converter, temp_paths, data_type, 
                        i+1, len(blob_names), user_session_id
                    )
                    
                    if chunks:  # Only consider successful if chunks were generated
                        all_chunks.extend(chunks)
                        all_texts.extend(texts)
                        file_success = True
                        self.successful_files.append(blob_name)
                        
                        file_duration = time.time() - file_start_time
                        self.processing_results[blob_name] = {
                            "status": "success",
                            "chunks_count": len(chunks),
                            "duration_seconds": file_duration,
                            "error": None,
                            "user_session_id": user_session_id
                        }
                        
                        self.progress_callback(
                            files_processed=i+1,
                            log_message=f"✅ Completed {blob_name}: {len(chunks)} chunks generated in {file_duration:.1f}s (User: {user_session_id})"
                        )
                    else:
                        # No chunks generated - consider this a failure
                        self.failed_files.append(blob_name)
                        self.processing_results[blob_name] = {
                            "status": "failed",
                            "chunks_count": 0,
                            "duration_seconds": time.time() - file_start_time,
                            "error": "No chunks generated",
                            "user_session_id": user_session_id
                        }
                        
                        self.progress_callback(
                            files_processed=i+1,
                            log_message=f"❌ Error processing {blob_name}: No chunks generated (User: {user_session_id})"
                        )
                
                except Exception as e:
                    # File processing failed
                    file_duration = time.time() - file_start_time
                    self.failed_files.append(blob_name)
                    self.processing_results[blob_name] = {
                        "status": "failed",
                        "chunks_count": 0,
                        "duration_seconds": file_duration,
                        "error": str(e),
                        "user_session_id": user_session_id
                    }
                    
                    self.progress_callback(
                        files_processed=i+1,
                        log_message=f"❌ Error processing {blob_name}: {str(e)} (User: {user_session_id})"
                    )
                    
                    _log.error(f"Error processing {blob_name} for user {user_session_id}: {e}")
                    # Continue with next file instead of stopping entire process
                    continue
            
            # Upload all outputs to blob storage with user prefix
            self.progress_callback(
                current_step="Uploading processed outputs",
                overall_progress=65,
                step_progress=90,
                log_message=f"Uploading processed files to blob storage (User: {user_session_id})..."
            )
            
            self._upload_outputs_to_blob_user(temp_paths, data_type, region_name, user_session_id)
            
        finally:
            # Clean up temporary files
            self.progress_callback(
                current_step="Cleaning up temporary files",
                step_progress=95,
                log_message=f"Cleaning up temporary files (User: {user_session_id})..."
            )
            self.blob_manager.cleanup_temp_files(temp_paths)
        
        # Final summary
        success_count = len(self.successful_files)
        failure_count = len(self.failed_files)
        total_chunks = len(all_chunks)
        
        session_info = f" (User: {user_session_id})" if user_session_id else ""
        
        if success_count == 0:
            self.progress_callback(
                log_message=f"❌ No documents were processed successfully{session_info}."
            )
            return [], []
        
        self.progress_callback(
            current_step="Document processing completed",
            step_progress=100,
            log_message=f"✅ Processing summary{session_info}: {success_count} successful, {failure_count} failed, {total_chunks} total chunks"
        )
        
        return all_chunks, all_texts
    
    def _upload_outputs_to_blob_user(self, 
                                    temp_paths: Dict[str, Path], 
                                    data_type: str, 
                                    region_name: Optional[str] = None,
                                    user_session_id: str = None):
        """Upload all temporary outputs to blob storage with user-specific prefixes."""
        
        # Create prefix for organizing files with user support
        if user_session_id and data_type == 'dce':
            prefix = f"user_{user_session_id}/dce/"
        elif data_type == 'region' and region_name:
            prefix = f"region_{region_name}/"
        else:
            prefix = f"{data_type}/"
        
        # Upload images with user-specific flat structure
        if temp_paths["images"].exists():
            self.progress_callback(
                log_message=f"Uploading images to blob storage with prefix: {prefix}..."
            )
            uploaded_images = self.blob_manager.upload_images_flat(temp_paths["images"])
            self.progress_callback(
                log_message=f"✅ Uploaded {len(uploaded_images)} images to blob storage"
            )
        
        # Upload other outputs with user prefix
        for output_type, container in [
            ("markdown", Config.MARKDOWN_CONTAINER),
            ("toc", Config.TOC_CONTAINER), 
            ("chunks", Config.CHUNKS_CONTAINER)
        ]:
            if temp_paths[output_type].exists():
                self.progress_callback(
                    log_message=f"Uploading {output_type} files with prefix: {prefix}..."
                )
                self.blob_manager.upload_directory_contents(
                    temp_paths[output_type], 
                    container, 
                    prefix
                )
        
        self.progress_callback(
            log_message=f"✅ All outputs uploaded to blob storage with prefix: {prefix}"
        )