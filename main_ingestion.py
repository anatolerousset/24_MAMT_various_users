"""
Ingestion pipeline with file deletion for non-archived files - UPDATED with office support
"""

import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Callable

from ingestion.document_processor import DocumentProcessor
from ingestion.vector_store_manager import VectorStoreManager
from ingestion.duplicate_manager import DuplicateManager
from utils.file_utils import get_input_files_from_blob
from utils.blob_utils import BlobStorageManager
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').disabled = True

async def main_ingestion_pipeline_with_progress(
    data_type: str = 'dce', 
    region_name: Optional[str] = None,
    office_name: Optional[str] = None,  # NEW: Added office_name parameter
    collection_name: str = None,
    recreate_collection: bool = False,
    remove_duplicates: bool = False,
    file_patterns: List[str] = None,
    archive_processed_files: bool = None,
    progress_callback: Callable = None
):
    """
    Main ingestion pipeline with file deletion for non-archived files - UPDATED with office support
    
    Args:
        data_type: Type of data ('dce' or 'region')
        region_name: Region name for regional processing
        office_name: Office name for DCE processing (NEW)
        collection_name: Qdrant collection name
        recreate_collection: Whether to recreate the collection
        remove_duplicates: Whether to remove duplicates after ingestion
        file_patterns: Optional list of file patterns to filter
        archive_processed_files: Whether to archive files after processing
        progress_callback: Callback function for progress updates
    """
    
    # Default progress callback
    def default_progress_callback(**kwargs):
        """Default progress callback that logs to console"""
        step = kwargs.get('current_step', '')
        progress = kwargs.get('overall_progress', 0)
        message = kwargs.get('log_message', '')
        
        if message:
            _log.info(f"[{progress:.1f}%] {step}: {message}")
    
    if progress_callback is None:
        progress_callback = default_progress_callback
    
    # Initialize components
    blob_manager = BlobStorageManager()
    processor = DocumentProcessor(progress_callback=progress_callback)
    vector_manager = VectorStoreManager()
    
    # Set collection name using the new Config method that supports office_name
    if collection_name is None:
        collection_name = Config.get_collection_name_for_data_type(
            data_type, 
            region_name, 
            office_name  # NEW: Pass office_name to collection name generation
        )
    
    # Set archiving behavior based on data type if not explicitly provided
    if archive_processed_files is None:
        archive_processed_files = Config.should_archive_data_type(data_type)
    
    progress_callback(
        status="starting",
        current_step="Initializing pipeline",
        overall_progress=0,
        log_message=f"Starting ingestion pipeline for {data_type}"
    )
    progress_callback(log_message=f"Collection: {collection_name}")
    if region_name:
        progress_callback(log_message=f"Region: {region_name}")
    if office_name:
        progress_callback(log_message=f"Office: {office_name}")  # NEW: Log office name
    progress_callback(log_message=f"Archive after processing: {archive_processed_files}")
    
    # Track processed files
    successfully_processed_files = []
    failed_files = []
    original_input_files = []
    
    try:
        # Step 1: Get input files from blob storage
        progress_callback(
            current_step="Getting input files from blob storage",
            overall_progress=5,
            step_progress=0,
            log_message="Scanning blob storage for input files..."
        )
        
        blob_names = get_input_files_from_blob(
            container_name=Config.INPUT_CONTAINER,
            file_extensions=['.pdf', '.docx', '.xlsx', '.xls']
        )
        
        # Store original list for tracking
        original_input_files = blob_names.copy()
        
        if file_patterns:
            # Filter by patterns if specified
            filtered_blobs = []
            for blob_name in blob_names:
                if any(pattern.lower() in blob_name.lower() for pattern in file_patterns):
                    filtered_blobs.append(blob_name)
            blob_names = filtered_blobs
            progress_callback(log_message=f"Filtered to {len(blob_names)} files matching patterns: {file_patterns}")
        
        if not blob_names:
            progress_callback(
                status="error",
                current_step="No files found",
                log_message="❌ No input files found in blob storage!"
            )
            return False
        
        progress_callback(
            overall_progress=10,
            total_files=len(blob_names),
            files_processed=0,
            log_message=f"✅ Found {len(blob_names)} files to process"
        )
        
        # Verify files exist before processing
        progress_callback(
            current_step="Verifying input files",
            overall_progress=12,
            log_message="Verifying all input files exist in blob storage..."
        )
        
        verified_files = []
        missing_files = []
        
        for blob_name in blob_names:
            if blob_manager.verify_blob_exists(Config.INPUT_CONTAINER, blob_name):
                verified_files.append(blob_name)
            else:
                missing_files.append(blob_name)
                _log.warning(f"File not found in blob storage: {blob_name}")
        
        if missing_files:
            progress_callback(
                log_message=f"⚠️ {len(missing_files)} files are missing from blob storage"
            )
            for missing in missing_files:
                progress_callback(log_message=f"  Missing: {missing}")
        
        if not verified_files:
            progress_callback(
                status="error",
                current_step="No valid files found",
                log_message="❌ No valid input files found in blob storage!"
            )
            return False
        
        blob_names = verified_files
        progress_callback(
            log_message=f"✅ Verified {len(blob_names)} files are accessible"
        )
        
        # Step 2: Process documents with progress tracking
        progress_callback(
            status="running",
            current_step="Processing documents",
            overall_progress=15,
            log_message="Starting document processing with detailed tracking..."
        )
        
        # Process the documents and get results from DocumentProcessor
        # Pass office_name for DCE processing or region_name for region processing
        processing_region = region_name if data_type == 'region' else None
        processing_office = office_name if data_type == 'dce' else None
        
        chunks, texts = await processor.process_files_from_blob_with_progress(
            blob_names=blob_names,
            data_type=data_type,
            region_name=processing_region,  # For regions
            office_name=processing_office,  # NEW: For DCE offices
            input_container=Config.INPUT_CONTAINER
        )
        
        # Get tracking results from the processor
        successfully_processed_files = processor.successful_files.copy()
        failed_files = processor.failed_files.copy()
        
        if not chunks:
            progress_callback(
                status="error",
                current_step="Processing failed",
                log_message="❌ No chunks were generated from the documents!"
            )
            return False
        
        progress_callback(
            overall_progress=65,
            current_step="Document processing completed",
            log_message=f"✅ Generated {len(chunks)} chunks from {len(successfully_processed_files)}/{len(blob_names)} documents"
        )
        
        # Log processing results
        if successfully_processed_files:
            progress_callback(
                log_message=f"Successfully processed files: {len(successfully_processed_files)}"
            )
            for i, file in enumerate(successfully_processed_files[:3]):  # Show first 3
                progress_callback(log_message=f"  ✅ {file}")
            if len(successfully_processed_files) > 3:
                progress_callback(log_message=f"  ... and {len(successfully_processed_files) - 3} more")
        
        if failed_files:
            progress_callback(
                log_message=f"Failed to process files: {len(failed_files)}"
            )
            for file in failed_files:
                progress_callback(log_message=f"  ❌ {file}")
        
        # Step 3: Create/update vector store
        progress_callback(
            current_step="Updating vector store",
            overall_progress=70,
            step_progress=0,
            log_message="Creating embeddings and updating vector store..."
        )
        
        if recreate_collection:
            progress_callback(log_message="Recreating collection from scratch...")
            vector_manager.recreate_collection(collection_name, chunks)
        else:
            progress_callback(log_message="Updating existing collection...")
            vector_manager.update_collection(collection_name, chunks)
        
        progress_callback(
            overall_progress=80,
            step_progress=100,
            log_message="✅ Vector store updated successfully"
        )
        
        # Step 4: Remove duplicates if requested
        if remove_duplicates:
            progress_callback(
                current_step="Removing duplicates",
                overall_progress=85,
                step_progress=0,
                log_message="Starting duplicate detection and removal..."
            )
            
            duplicate_manager = DuplicateManager()
            
            # First run a dry run to see what would be deleted
            progress_callback(log_message="Analyzing documents for duplicates...")
            result = duplicate_manager.remove_duplicates(
                collection_name=collection_name,
                similarity_threshold=0.95,
                dry_run=True,
                save_report=True
            )
            
            if result["success"] and result.get("duplicates_found", False):
                progress_callback(
                    log_message=f"Found {result['duplicate_groups']} duplicate groups"
                )
                progress_callback(
                    log_message=f"Will delete {result['documents_to_delete']} documents ({result['deletion_percentage']:.1f}%)"
                )
                
                # Proceed with deletion
                progress_callback(log_message="Removing duplicate documents...")
                duplicate_manager.remove_duplicates(
                    collection_name=collection_name,
                    similarity_threshold=0.95,
                    dry_run=False,
                    save_report=False  # Already saved from dry run
                )
                progress_callback(log_message="✅ Duplicate removal completed")
            else:
                progress_callback(log_message="✅ No duplicates found")
        
        progress_callback(
            overall_progress=90,
            step_progress=100,
            current_step="Duplicate processing completed"
        )
        
        # STEP 5: Handle file management (archiving OR deletion)
        if successfully_processed_files:
            if archive_processed_files:
                # Archive files as before
                progress_callback(
                    current_step="Archiving processed files",
                    overall_progress=95,
                    step_progress=0,
                    log_message=f"Archiving {len(successfully_processed_files)} successfully processed files..."
                )
                
                progress_callback(
                    log_message=f"Files to archive: {successfully_processed_files}"
                )
                
                try:
                    archived_files = blob_manager.archive_processed_files(
                        processed_blob_names=successfully_processed_files,
                        source_container=Config.INPUT_CONTAINER,
                        archive_container=Config.ARCHIVE_CONTAINER,
                        add_timestamp=Config.ARCHIVE_ADD_TIMESTAMP
                    )
                    
                    if archived_files:
                        progress_callback(
                            log_message=f"✅ Successfully archived {len(archived_files)} files"
                        )
                        
                        # Verify archiving was successful
                        verified_archives = 0
                        for original, archived in archived_files.items():
                            if blob_manager.verify_blob_exists(Config.ARCHIVE_CONTAINER, archived):
                                verified_archives += 1
                            else:
                                _log.warning(f"Archived file not found: {Config.ARCHIVE_CONTAINER}/{archived}")
                        
                        progress_callback(
                            log_message=f"✅ Verified {verified_archives}/{len(archived_files)} archived files"
                        )
                        
                        # Show sample of archived files
                        for original, archived in list(archived_files.items())[:3]:  # Show first 3
                            progress_callback(log_message=f"  {original} -> {archived}")
                        if len(archived_files) > 3:
                            progress_callback(log_message=f"  ... and {len(archived_files) - 3} more files")
                    else:
                        progress_callback(log_message="⚠️ No files were archived (all archive operations may have failed)")
                        
                except Exception as e:
                    progress_callback(
                        log_message=f"❌ Error during archiving: {str(e)}"
                    )
                    _log.error(f"Archiving error: {e}")
                    # Don't fail the entire pipeline due to archiving errors
            
            else:
                # NEW: Delete files from input container when archiving is disabled
                progress_callback(
                    current_step="Deleting processed files from input container",
                    overall_progress=95,
                    step_progress=0,
                    log_message=f"Deleting {len(successfully_processed_files)} successfully processed files from input container..."
                )
                
                progress_callback(
                    log_message=f"Files to delete: {successfully_processed_files}"
                )
                
                deleted_files = []
                deletion_errors = []
                
                try:
                    for i, blob_name in enumerate(successfully_processed_files, 1):
                        try:
                            progress_callback(
                                log_message=f"Deleting file {i}/{len(successfully_processed_files)}: {blob_name}"
                            )
                            
                            # Verify file exists before attempting deletion
                            if blob_manager.verify_blob_exists(Config.INPUT_CONTAINER, blob_name):
                                blob_manager.delete_blob(Config.INPUT_CONTAINER, blob_name)
                                deleted_files.append(blob_name)
                                progress_callback(
                                    log_message=f"✅ [{i}/{len(successfully_processed_files)}] Deleted: {blob_name}"
                                )
                            else:
                                progress_callback(
                                    log_message=f"⚠️ [{i}/{len(successfully_processed_files)}] File not found, skipping: {blob_name}"
                                )
                                deletion_errors.append((blob_name, "File not found"))
                        
                        except Exception as e:
                            error_msg = f"Failed to delete {blob_name}: {str(e)}"
                            deletion_errors.append((blob_name, str(e)))
                            progress_callback(
                                log_message=f"❌ [{i}/{len(successfully_processed_files)}] {error_msg}"
                            )
                            _log.error(error_msg)
                            # Continue with other files even if one fails
                            continue
                    
                    # Report deletion results
                    if deleted_files:
                        progress_callback(
                            log_message=f"✅ Successfully deleted {len(deleted_files)} files from input container"
                        )
                        
                        # Show sample of deleted files
                        for file in deleted_files[:3]:  # Show first 3
                            progress_callback(log_message=f"  🗑️ {file}")
                        if len(deleted_files) > 3:
                            progress_callback(log_message=f"  ... and {len(deleted_files) - 3} more files")
                    
                    if deletion_errors:
                        progress_callback(
                            log_message=f"⚠️ Failed to delete {len(deletion_errors)} files:"
                        )
                        for file, error in deletion_errors[:3]:  # Show first 3 errors
                            progress_callback(log_message=f"  ❌ {file}: {error}")
                        if len(deletion_errors) > 3:
                            progress_callback(log_message=f"  ... and {len(deletion_errors) - 3} more deletion errors")
                    
                    # Log final deletion statistics
                    progress_callback(
                        log_message=f"📊 Deletion summary: {len(deleted_files)} deleted, {len(deletion_errors)} failed"
                    )
                    
                except Exception as e:
                    progress_callback(
                        log_message=f"❌ Error during file deletion: {str(e)}"
                    )
                    _log.error(f"File deletion error: {e}")
                    # Don't fail the entire pipeline due to deletion errors
        
        elif not successfully_processed_files:
            if archive_processed_files:
                progress_callback(
                    log_message="⚠️ No files to archive (no files were processed successfully)"
                )
            else:
                progress_callback(
                    log_message="⚠️ No files to delete (no files were processed successfully)"
                )
        
        # Step 6: Final collection info and completion
        progress_callback(
            current_step="Getting final collection info",
            overall_progress=98,
            log_message="Retrieving final collection statistics..."
        )
        
        collection_info = vector_manager.get_collection_info(collection_name)
        if collection_info.get("exists"):
            progress_callback(
                log_message=f"✅ Final collection stats: {collection_info['points_count']} documents, {collection_info['vectors_count']} vectors"
            )
        
        # Final completion summary
        progress_callback(
            status="completed",
            current_step="Pipeline completed",
            overall_progress=100,
            step_progress=100,
            log_message="🎉 Ingestion pipeline completed successfully!"
        )
        
        # Final summary
        progress_callback(
            log_message=f"📊 FINAL SUMMARY:"
        )
        progress_callback(
            log_message=f"  • Input files found: {len(original_input_files)}"
        )
        progress_callback(
            log_message=f"  • Files processed successfully: {len(successfully_processed_files)}"
        )
        progress_callback(
            log_message=f"  • Files failed: {len(failed_files)}"
        )
        progress_callback(
            log_message=f"  • Chunks generated: {len(chunks)}"
        )
        progress_callback(
            log_message=f"  • Target collection: {collection_name}"
        )
        
        # File management summary
        if archive_processed_files:
            try:
                archived_count = len(blob_manager.list_archived_files(Config.ARCHIVE_CONTAINER))
                progress_callback(
                    log_message=f"  • Files archived: {archived_count} total in archive container"
                )
            except:
                progress_callback(
                    log_message=f"  • Archiving was attempted (could not get final count)"
                )
        else:
            # Count remaining files in input container to verify deletion
            try:
                remaining_files = get_input_files_from_blob(Config.INPUT_CONTAINER)
                progress_callback(
                    log_message=f"  • Files deleted from input container: files processed successfully"
                )
                progress_callback(
                    log_message=f"  • Remaining files in input container: {len(remaining_files)}"
                )
            except:
                progress_callback(
                    log_message=f"  • File deletion was attempted (could not get final count)"
                )
        
        return True
        
    except Exception as e:
        progress_callback(
            status="error",
            current_step="Error occurred",
            log_message=f"❌ Ingestion pipeline failed: {str(e)}"
        )
        
        # If there was an error but some files were processed successfully, 
        # still try to handle them according to the archiving setting
        if successfully_processed_files:
            if archive_processed_files:
                progress_callback(
                    log_message="Attempting to archive successfully processed files despite pipeline error..."
                )
                try:
                    archived_files = blob_manager.archive_processed_files(
                        processed_blob_names=successfully_processed_files,
                        source_container=Config.INPUT_CONTAINER,
                        archive_container=Config.ARCHIVE_CONTAINER,
                        add_timestamp=Config.ARCHIVE_ADD_TIMESTAMP
                    )
                    progress_callback(
                        log_message=f"✅ Archived {len(archived_files)} successfully processed files despite error"
                    )
                except Exception as archive_error:
                    progress_callback(
                        log_message=f"❌ Failed to archive files after pipeline error: {archive_error}"
                    )
            else:
                progress_callback(
                    log_message="Attempting to delete successfully processed files despite pipeline error..."
                )
                try:
                    deleted_count = 0
                    for blob_name in successfully_processed_files:
                        try:
                            if blob_manager.verify_blob_exists(Config.INPUT_CONTAINER, blob_name):
                                blob_manager.delete_blob(Config.INPUT_CONTAINER, blob_name)
                                deleted_count += 1
                        except:
                            continue
                    progress_callback(
                        log_message=f"✅ Deleted {deleted_count} successfully processed files despite error"
                    )
                except Exception as deletion_error:
                    progress_callback(
                        log_message=f"❌ Failed to delete files after pipeline error: {deletion_error}"
                    )
        
        return False


# Wrapper function for backward compatibility with async handling
def main_ingestion_pipeline(data_type: str = 'dce', 
                           region_name: Optional[str] = None,
                           office_name: Optional[str] = None,  # NEW: Added office_name parameter
                           collection_name: str = None,
                           recreate_collection: bool = False,
                           remove_duplicates: bool = False,
                           file_patterns: List[str] = None,
                           archive_processed_files: bool = None):
    """
    Main ingestion pipeline - wrapper for backward compatibility.
    This function handles the async call using asyncio.run()
    """
    return asyncio.run(main_ingestion_pipeline_with_progress(
        data_type=data_type,
        region_name=region_name,
        office_name=office_name,  # NEW: Pass office_name
        collection_name=collection_name,
        recreate_collection=recreate_collection,
        remove_duplicates=remove_duplicates,
        file_patterns=file_patterns,
        archive_processed_files=archive_processed_files,
        progress_callback=None  # Use default console logging
    ))


# Async function for use in async contexts
async def run_ingestion_async(data_type: str = 'dce', 
                             region_name: Optional[str] = None,
                             office_name: Optional[str] = None,  # NEW: Added office_name parameter
                             collection_name: str = None,
                             recreate_collection: bool = False,
                             remove_duplicates: bool = False,
                             file_patterns: List[str] = None,
                             archive_processed_files: bool = None,
                             progress_callback: Callable = None):
    """
    Async version of the ingestion pipeline for use in async contexts.
    """
    return await main_ingestion_pipeline_with_progress(
        data_type=data_type,
        region_name=region_name,
        office_name=office_name,  # NEW: Pass office_name
        collection_name=collection_name,
        recreate_collection=recreate_collection,
        remove_duplicates=remove_duplicates,
        file_patterns=file_patterns,
        archive_processed_files=archive_processed_files,
        progress_callback=progress_callback
    )


# Debug function to check file management status
def debug_file_management_status(data_type: str = 'dce', office_name: str = None):
    """
    Debug function to check the status of file management operations - UPDATED with office support
    """
    _log.info("🔍 DEBUG: Checking file management status...")
    
    blob_manager = BlobStorageManager()
    
    # Check input container
    input_files = blob_manager.list_blobs(Config.INPUT_CONTAINER)
    _log.info(f"Input container ({Config.INPUT_CONTAINER}): {len(input_files)} files")
    for file_info in input_files[:5]:  # Show first 5
        _log.info(f"  - {file_info['name']} ({file_info['size']} bytes)")
    
    # Check archive container
    try:
        archive_files = blob_manager.list_blobs(Config.ARCHIVE_CONTAINER)
        _log.info(f"Archive container ({Config.ARCHIVE_CONTAINER}): {len(archive_files)} files")
        for file_info in archive_files[:5]:  # Show first 5
            _log.info(f"  - {file_info['name']} ({file_info['size']} bytes)")
    except Exception as e:
        _log.error(f"Error accessing archive container: {e}")
    
    # Check archiving behavior for data type
    should_archive = Config.should_archive_data_type(data_type)
    _log.info(f"Data type '{data_type}' archiving behavior: {'ARCHIVE' if should_archive else 'DELETE'}")
    
    # Show what collection would be used
    collection_name = Config.get_collection_name_for_data_type(data_type, office_name=office_name)
    _log.info(f"Target collection would be: {collection_name}")
    
    return {
        "input_files_count": len(input_files),
        "archive_files_count": len(archive_files) if 'archive_files' in locals() else 0,
        "input_files": [f['name'] for f in input_files],
        "archive_files": [f['name'] for f in archive_files] if 'archive_files' in locals() else [],
        "data_type_should_archive": should_archive,
        "target_collection": collection_name
    }


if __name__ == "__main__":
    # Example usage with office support
    
    def example_progress_callback(**kwargs):
        """Example progress callback that demonstrates all available fields"""
        import json
        print(f"PROGRESS UPDATE: {json.dumps(kwargs, indent=2, default=str)}")
    
    # First, run debug to check current state
    print("=" * 80)
    print("DEBUGGING CURRENT FILE MANAGEMENT STATUS")
    print("=" * 80)
    debug_status = debug_file_management_status('dce', 'CSP')  # NEW: Test with office
    print(f"Debug results: {debug_status}")
    
    print("\n" + "=" * 80)
    print("STARTING DCE INGESTION WITH OFFICE SELECTION (FILES WILL BE DELETED, NOT ARCHIVED)")
    print("=" * 80)
    
    # Example: Run DCE ingestion with office selection where files are deleted instead of archived
    success = asyncio.run(main_ingestion_pipeline_with_progress(
        data_type='dce',  # DCE files are not archived by default
        office_name='CSP',  # NEW: Specify office for DCE
        collection_name=None,  # Let it auto-generate: dce_csp_documents
        recreate_collection=False,
        remove_duplicates=True,
        archive_processed_files=False,  # Explicitly disable archiving = files will be deleted
        progress_callback=example_progress_callback
    ))
    
    print(f"\nDCE ingestion pipeline result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    # Check status after ingestion
    print("\n" + "=" * 80)
    print("POST-INGESTION FILE MANAGEMENT STATUS")
    print("=" * 80)
    post_debug_status = debug_file_management_status('dce', 'CSP')
    print(f"Post-ingestion debug results: {post_debug_status}")
    
    # Show the difference
    files_removed = set(debug_status['input_files']) - set(post_debug_status['input_files'])
    
    print(f"\nFiles removed from input container: {len(files_removed)}")
    for file in files_removed:
        print(f"  🗑️ {file}")
    
    if not files_removed:
        print("⚠️ No files were removed from input container")