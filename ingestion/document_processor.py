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
    """Document processor with detailed progress tracking and individual file success tracking."""
    
    def __init__(self, tokenizer_model: str = None, max_tokens: int = None, progress_callback: Callable = None):
        """
        Initialize the document processor.
        
        Args:
            tokenizer_model: Model name for tokenization
            max_tokens: Maximum tokens per chunk
            progress_callback: Callback function for progress updates
        """
        self.tokenizer_model = tokenizer_model or Config.FINETUNED_DENSE_EMBED_MODEL_ID
        self.max_tokens = max_tokens or Config.DEFAULT_MAX_TOKENS
        self.blob_manager = BlobStorageManager()
        self.progress_callback = progress_callback or self._default_progress_callback
        
        # ADDED: Track processing results per file
        self.processing_results = {}
        self.successful_files = []
        self.failed_files = []
        
        # Enable profiling to measure time spent
        settings.debug.profile_pipeline_timings = True
        
        # Initialize converters
        self._init_converters()
        
        # Initialize chunker
        self.chunker = None
    
    def _default_progress_callback(self, **kwargs):
        """Default progress callback that does nothing"""
        pass
    
    def _init_converters(self):
        """Initialize document converters for different formats."""
        # Pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False,
            ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["fr"]),
            table_structure_options=dict(
                do_cell_matching=False,
                mode=TableFormerMode.ACCURATE
            ),
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=Config.IMAGE_RESOLUTION_SCALE,
        )
        
        # Document converter for DCE (PDF + Excel + DOCX)
        self.dce_converter = DocumentConverter(
            allowed_formats=[InputFormat.DOCX, InputFormat.PDF, InputFormat.XLSX],
            format_options={
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline,
                ),
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline, 
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_options=pipeline_options
                ),
                InputFormat.XLSX: ExcelFormatOption(
                    pipeline_cls=SimplePipeline,
                ),
            },
        )
        
        # Document converter for regions (DOCX only)
        self.region_converter = DocumentConverter(
            allowed_formats=[InputFormat.DOCX],
            format_options={
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline,
                    backend=MsWordDocumentBackendModMetadataplus,
                ),
            },
        )
    
    def _init_chunker(self, image_output_dir: Path):
        """Initialize the chunker with the specified image output directory."""
        if not self.chunker:
            self.chunker = HybridChunkerWithImages(
                tokenizer=self.tokenizer_model,
                image_mode=ImageRefMode.REFERENCED,
                image_output_dir=image_output_dir,
                max_tokens=self.max_tokens
            )
    
    async def process_files_from_blob_with_progress(self, 
                                                   blob_names: List[str], 
                                                   data_type: str,
                                                   region_name: Optional[str] = None,
                                                   input_container: str = None) -> Tuple[List[LCDocument], List[str]]:
        """
        FIXED: Process files from blob storage with detailed progress tracking and individual file success tracking.
        
        Args:
            blob_names: List of blob names to process
            data_type: Type of data being processed ('dce' or 'region')
            region_name: Name of the region (if processing regional data)
            input_container: Container containing input files
            
        Returns:
            Tuple of (all_chunks, all_texts)
        """
        input_container = input_container or Config.INPUT_CONTAINER
        
        # Reset tracking variables
        self.processing_results = {}
        self.successful_files = []
        self.failed_files = []
        
        self.progress_callback(
            current_step="Starting document processing",
            step_progress=0,
            log_message=f"Processing {len(blob_names)} documents from blob storage for {data_type}..."
        )
        
        # Get temporary paths for processing
        temp_paths = Config.get_temp_paths(data_type, region_name)
        
        # Initialize chunker with temporary image directory
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
                        log_message=f"Processing file {i+1}/{len(blob_names)}: {blob_name}"
                    )
                    
                    # Process single file
                    chunks, texts = await self._process_single_blob_with_progress(
                        blob_name, input_container, converter, temp_paths, data_type, i+1, len(blob_names)
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
                            "error": None
                        }
                        
                        self.progress_callback(
                            files_processed=i+1,
                            log_message=f"✅ Completed {blob_name}: {len(chunks)} chunks generated in {file_duration:.1f}s"
                        )
                    else:
                        # No chunks generated - consider this a failure
                        self.failed_files.append(blob_name)
                        self.processing_results[blob_name] = {
                            "status": "failed",
                            "chunks_count": 0,
                            "duration_seconds": time.time() - file_start_time,
                            "error": "No chunks generated"
                        }
                        
                        self.progress_callback(
                            files_processed=i+1,
                            log_message=f"❌ Error processing {blob_name}: No chunks generated"
                        )
                    
                except Exception as e:
                    # File processing failed
                    file_duration = time.time() - file_start_time
                    self.failed_files.append(blob_name)
                    self.processing_results[blob_name] = {
                        "status": "failed",
                        "chunks_count": 0,
                        "duration_seconds": file_duration,
                        "error": str(e)
                    }
                    
                    self.progress_callback(
                        files_processed=i+1,
                        log_message=f"❌ Error processing {blob_name}: {str(e)}"
                    )
                    
                    _log.error(f"Error processing {blob_name}: {e}")
                    # Continue with next file instead of stopping entire process
                    continue
            
            # Upload all outputs to blob storage
            self.progress_callback(
                current_step="Uploading processed outputs",
                overall_progress=65,
                step_progress=90,
                log_message="Uploading processed files to blob storage..."
            )
            
            self._upload_outputs_to_blob(temp_paths, data_type, region_name)
            
        finally:
            # Clean up temporary files
            self.progress_callback(
                current_step="Cleaning up temporary files",
                step_progress=95,
                log_message="Cleaning up temporary files..."
            )
            self.blob_manager.cleanup_temp_files(temp_paths)
        
        # Final summary
        success_count = len(self.successful_files)
        failure_count = len(self.failed_files)
        total_chunks = len(all_chunks)
        
        if success_count == 0:
            self.progress_callback(
                log_message="❌ No documents were processed successfully."
            )
            return [], []
        
        self.progress_callback(
            current_step="Document processing completed",
            step_progress=100,
            log_message=f"✅ Processing summary: {success_count} successful, {failure_count} failed, {total_chunks} total chunks"
        )
        
        # Log detailed results
        if self.successful_files:
            self.progress_callback(log_message=f"Successfully processed files:")
            for file in self.successful_files[:5]:  # Show first 5
                result = self.processing_results[file]
                self.progress_callback(
                    log_message=f"  ✅ {file}: {result['chunks_count']} chunks ({result['duration_seconds']:.1f}s)"
                )
            if len(self.successful_files) > 5:
                self.progress_callback(log_message=f"  ... and {len(self.successful_files) - 5} more successful files")
        
        if self.failed_files:
            self.progress_callback(log_message=f"Failed to process files:")
            for file in self.failed_files:
                result = self.processing_results[file]
                self.progress_callback(
                    log_message=f"  ❌ {file}: {result['error']}"
                )
        
        return all_chunks, all_texts
    
    async def _process_single_blob_with_progress(self, 
                                               blob_name: str,
                                               input_container: str,
                                               converter: DocumentConverter,
                                               temp_paths: Dict[str, Path],
                                               data_type: str,
                                               file_index: int,
                                               total_files: int) -> Tuple[List[LCDocument], List[str]]:
        """FIXED: Process a single blob with detailed progress tracking and better error handling."""
        
        # Step 1: Download blob
        self.progress_callback(
            current_step=f"Downloading {blob_name}",
            step_progress=0,
            log_message=f"Downloading {blob_name} from blob storage..."
        )
        
        temp_file = self.blob_manager.download_blob_to_temp(input_container, blob_name)
        
        try:
            # Step 2: Handle XLS files
            if temp_file.suffix.lower() == '.xls':
                self.progress_callback(
                    current_step=f"Converting XLS to XLSX",
                    step_progress=10,
                    log_message=f"Converting {blob_name} from XLS to XLSX..."
                )
                
                temp_file = convert_xls_to_xlsx(temp_file)
                if temp_file.suffix.lower() == '.xls':
                    self.progress_callback(
                        log_message=f"⚠️ Could not convert {blob_name} from XLS, skipping"
                    )
                    return [], []
            
            # Step 3: Document conversion with Docling
            self.progress_callback(
                current_step=f"Processing with Docling",
                step_progress=20,
                log_message=f"Starting Docling conversion for {blob_name}..."
            )
            
            conversion_start = time.time()
            conversion_result = converter.convert(source=str(temp_file))
            document = conversion_result.document
            conversion_time = time.time() - conversion_start
            
            self.progress_callback(
                current_step=f"Docling conversion completed",
                step_progress=50,
                log_message=f"✅ Docling conversion completed in {conversion_time:.2f}s"
            )
            
            # Step 4: Add metadata for DCE files
            if data_type == 'dce':
                self.progress_callback(
                    current_step=f"Adding metadata",
                    step_progress=60,
                    log_message="Adding DCE metadata..."
                )
                
                new_origin = DocumentOriginMod(
                    filename=document.origin.filename,
                    mimetype=document.origin.mimetype,
                    binary_hash=document.origin.binary_hash,
                    creation_date=None,
                    last_modified_date=None
                )
                document.origin = new_origin
            
            # Step 5: Export outputs
            self.progress_callback(
                current_step=f"Exporting document outputs",
                step_progress=70,
                log_message="Exporting markdown and TOC files..."
            )
            
            doc_filename = Path(blob_name).stem
            self._export_document_outputs(document, doc_filename, temp_paths, data_type)
            
            # Step 6: Chunk document
            self.progress_callback(
                current_step=f"Chunking document",
                step_progress=80,
                log_message="Breaking document into chunks..."
            )
            
            chunks, texts = self._chunk_document_with_progress(document, blob_name, doc_filename, temp_paths)
            
            self.progress_callback(
                current_step=f"File processing completed",
                step_progress=100,
                log_message=f"✅ Generated {len(chunks)} chunks from {blob_name}"
            )
            
            return chunks, texts
            
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def _export_document_outputs(self, 
                                document, 
                                doc_filename: str, 
                                temp_paths: Dict[str, Path],
                                data_type: str):
        """Export document to various output formats in temporary directories."""
        
        # Export markdown
        md_filename = temp_paths["markdown"] / f"{doc_filename}.md"
        document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
        
        # Fix image paths for DOCX files to point to blob storage
        if data_type == 'region':
            with open(md_filename, "r", encoding="utf-8") as f:
                md_content = f.read()
            
            fixed_md_content = fix_image_paths(md_content)
            
            with open(md_filename, "w", encoding="utf-8") as f:
                f.write(fixed_md_content)
        
        _log.info(f"Exported markdown to temp: {md_filename}")
        
        # Export TOC for DOCX files
        if data_type == 'region':
            toc_filename = temp_paths["toc"] / f"{doc_filename}_TOC.md"
            with open(toc_filename, "w", encoding="utf-8") as f:
                f.write(MsWordDocumentBackendModMetadataplus.export_table_of_contents(document))
            _log.info(f"Exported TOC to temp: {toc_filename}")
    
    def _chunk_document_with_progress(self, 
                                    document, 
                                    blob_name: str, 
                                    doc_filename: str, 
                                    temp_paths: Dict[str, Path]) -> Tuple[List[LCDocument], List[str]]:
        """Chunk a document with progress tracking."""
        
        chunking_start = time.time()
        document_chunks = []
        chunk_count = 0
        all_texts = []
        
        self.progress_callback(
            log_message="Starting document chunking process..."
        )
        
        # Process document into chunks
        for chunk in self.chunker.chunk(document):
            chunk_count += 1
            
            # Create metadata
            metadata = {
                "source": blob_name,  # Use blob name as source
                "filename": Path(blob_name).name,
                "file_extension": Path(blob_name).suffix.lower(),
                "dl_meta": chunk.meta.export_json_dict()
            }
            
            # Extract and format headings
            headings = []
            if "headings" in chunk.meta.export_json_dict():
                headings = chunk.meta.export_json_dict()["headings"]
            
            formatted_headings = format_headings(headings)
            enhanced_content = formatted_headings + chunk.text
            
            all_texts.append(enhanced_content)
            
            lc_doc = LCDocument(
                page_content=enhanced_content,
                metadata=metadata
            )
            
            document_chunks.append(lc_doc)
            
            # Update progress every 10 chunks
            if chunk_count % 10 == 0:
                self.progress_callback(
                    log_message=f"Generated {chunk_count} chunks so far..."
                )
        
        chunking_time = time.time() - chunking_start
        
        self.progress_callback(
            log_message=f"✅ Document chunked into {chunk_count} chunks in {chunking_time:.2f} seconds"
        )
        
        # Export chunks to temporary directory
        self._export_chunks(document_chunks, doc_filename, temp_paths)
        
        return document_chunks, all_texts
    
    def _export_chunks(self, 
                      chunks: List[LCDocument], 
                      doc_filename: str, 
                      temp_paths: Dict[str, Path]):
        """Export chunks to markdown and JSON formats in temporary directories."""
        
        # Export chunks as markdown
        chunks_md_filename = temp_paths["chunks"] / f"{doc_filename}_chunks.md"
        with open(chunks_md_filename, "w", encoding="utf-8") as f:
            f.write(f"# Document Chunks: {doc_filename}\n\n")
            
            for i, chunk in enumerate(chunks, 1):
                f.write(f"## Chunk {i}\n\n")
                
                if chunk.metadata:
                    f.write("### Metadata\n")
                    for key, value in chunk.metadata.items():
                        if key != "dl_meta":
                            f.write(f"- **{key}**: {value}\n")
                        else:
                            if "headings" in value:
                                f.write(f"- **headings**: {value['headings']}\n")
                    f.write("\n")
                
                f.write("### Content\n")
                f.write(chunk.page_content)
                f.write("\n\n---\n\n")
        
        _log.info(f"Exported chunk details to temp: {chunks_md_filename}")
        
        # Export chunks to JSON
        chunks_json_filename = temp_paths["chunks"] / f"{doc_filename}_chunks.json"
        with open(chunks_json_filename, "w", encoding="utf-8") as f:
            json.dump({
                "documents": [chunk.page_content for chunk in chunks],
                "metadatas": [chunk.metadata for chunk in chunks]
            }, f, ensure_ascii=False, indent=2)
        
        _log.info(f"Exported chunks to JSON temp: {chunks_json_filename}")
    
    def _upload_outputs_to_blob(self, 
                               temp_paths: Dict[str, Path], 
                               data_type: str, 
                               region_name: Optional[str] = None):
        """Upload all temporary outputs to blob storage with progress tracking."""
        
        # Create prefix for organizing files
        if data_type == 'region' and region_name:
            prefix = f"region_{region_name}/"
        else:
            prefix = f"{data_type}/"
        
        # Upload images with flat structure (no subfolders)
        if temp_paths["images"].exists():
            self.progress_callback(
                log_message="Uploading images to blob storage..."
            )
            uploaded_images = self.blob_manager.upload_images_flat(temp_paths["images"])
            self.progress_callback(
                log_message=f"✅ Uploaded {len(uploaded_images)} images to blob storage"
            )
        
        # Upload markdown files
        if temp_paths["markdown"].exists():
            self.progress_callback(
                log_message="Uploading markdown files..."
            )
            self.blob_manager.upload_directory_contents(
                temp_paths["markdown"], 
                Config.MARKDOWN_CONTAINER, 
                prefix
            )
        
        # Upload TOC files
        if temp_paths["toc"].exists():
            self.progress_callback(
                log_message="Uploading TOC files..."
            )
            self.blob_manager.upload_directory_contents(
                temp_paths["toc"], 
                Config.TOC_CONTAINER, 
                prefix
            )
        
        # Upload chunk files
        if temp_paths["chunks"].exists():
            self.progress_callback(
                log_message="Uploading chunk files..."
            )
            self.blob_manager.upload_directory_contents(
                temp_paths["chunks"], 
                Config.CHUNKS_CONTAINER, 
                prefix
            )
        
        self.progress_callback(
            log_message=f"✅ All outputs uploaded to blob storage with prefix: {prefix}"
        )