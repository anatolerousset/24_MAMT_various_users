"""
Azure blob storage manager
"""
import logging
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
import uuid
from datetime import datetime

from config.config import Config

_log = logging.getLogger(__name__)

class BlobStorageManager:
    """Manages Azure Blob Storage operations with archiving support"""
    
    def __init__(self):
        self.blob_service_client = Config.get_blob_client()
        Config.ensure_containers()
    
    def upload_file_to_blob(self, 
                           local_file_path: Path, 
                           container_name: str, 
                           blob_name: str = None,
                           overwrite: bool = True) -> str:
        """
        Upload a local file to blob storage
        
        Args:
            local_file_path: Path to local file
            container_name: Target container name
            blob_name: Name for the blob (if None, uses filename)
            overwrite: Whether to overwrite existing blob
            
        Returns:
            Blob name
        """
        if blob_name is None:
            blob_name = local_file_path.name
            
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)
                
            #_log.info(f"Uploaded {local_file_path.name} to {container_name}/{blob_name}")
            return blob_name
            
        except Exception as e:
            _log.error(f"Error uploading {local_file_path.name}: {e}")
            raise
    
    def upload_data(self, 
                   data: Union[str, bytes], 
                   container_name: str, 
                   blob_name: str,
                   content_type: str = None,
                   metadata: Dict[str, str] = None,
                   overwrite: bool = True) -> str:
        """
        Upload data directly to blob storage without creating a local file
        
        Args:
            data: Data to upload (string or bytes)
            container_name: Target container name
            blob_name: Name for the blob
            content_type: MIME type of the content
            metadata: Optional metadata dictionary
            overwrite: Whether to overwrite existing blob
            
        Returns:
            Blob name
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
                if content_type is None:
                    content_type = 'text/plain; charset=utf-8'
            
            # Set content settings if content_type is provided
            content_settings = None
            if content_type:
                from azure.storage.blob import ContentSettings
                content_settings = ContentSettings(content_type=content_type)
            
            # Upload the data
            blob_client.upload_blob(
                data, 
                overwrite=overwrite,
                content_settings=content_settings,
                metadata=metadata or {}
            )
                
            _log.info(f"Uploaded data to {container_name}/{blob_name}")
            return blob_name
            
        except Exception as e:
            _log.error(f"Error uploading data to {container_name}/{blob_name}: {e}")
            raise
    
    def download_data(self, container_name: str, blob_name: str) -> bytes:
        """
        Download blob data directly as bytes
        
        Args:
            container_name: Source container name
            blob_name: Source blob name
            
        Returns:
            Raw blob data as bytes
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob().readall()
            _log.debug(f"Downloaded data from {container_name}/{blob_name}")
            return blob_data
            
        except Exception as e:
            _log.error(f"Error downloading data from {container_name}/{blob_name}: {e}")
            raise
    
    def download_blob_to_temp(self, 
                             container_name: str, 
                             blob_name: str) -> Path:
        """
        Download a blob to a temporary local file
        
        Args:
            container_name: Source container name
            blob_name: Source blob name
            
        Returns:
            Path to downloaded temporary file
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Create temp file with same extension
            suffix = Path(blob_name).suffix
            temp_file = Path(tempfile.mktemp(suffix=suffix))
            
            with open(temp_file, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
                
            _log.debug(f"Downloaded {container_name}/{blob_name} to {temp_file}")
            return temp_file
            
        except Exception as e:
            _log.error(f"Error downloading {container_name}/{blob_name}: {e}")
            raise
    
    def copy_blob_between_containers(self,
                                   source_container: str,
                                   source_blob: str,
                                   dest_container: str,
                                   dest_blob: str = None,
                                   timeout_seconds: int = 300) -> str:
        """
        FIXED: Copy a blob from one container to another with proper error handling and timeout
        
        Args:
            source_container: Source container name
            source_blob: Source blob name
            dest_container: Destination container name
            dest_blob: Destination blob name (if None, uses source blob name)
            timeout_seconds: Maximum time to wait for copy completion
            
        Returns:
            Destination blob name
        """
        if dest_blob is None:
            dest_blob = source_blob
            
        try:
            # Ensure destination container exists
            self.ensure_container_exists(dest_container)
            
            # Get source blob URL
            source_blob_client = self.blob_service_client.get_blob_client(
                container=source_container,
                blob=source_blob
            )
            
            # Verify source blob exists
            try:
                source_props = source_blob_client.get_blob_properties()
                _log.debug(f"Source blob {source_container}/{source_blob} exists, size: {source_props.size} bytes")
            except ResourceNotFoundError:
                raise ResourceNotFoundError(f"Source blob {source_container}/{source_blob} not found")
            
            source_url = source_blob_client.url
            
            # Copy to destination
            dest_blob_client = self.blob_service_client.get_blob_client(
                container=dest_container,
                blob=dest_blob
            )
            
            _log.info(f"Starting copy from {source_container}/{source_blob} to {dest_container}/{dest_blob}")
            
            # Start the copy operation
            copy_operation = dest_blob_client.start_copy_from_url(source_url)
            
            # Wait for copy to complete with timeout
            start_time = time.time()
            max_wait_time = timeout_seconds
            
            while True:
                try:
                    copy_props = dest_blob_client.get_blob_properties()
                    copy_status = copy_props.copy.status
                    
                    _log.debug(f"Copy status: {copy_status}")
                    
                    if copy_status == 'success':
                        _log.info(f"✓ Copy completed successfully: {source_container}/{source_blob} -> {dest_container}/{dest_blob}")
                        return dest_blob
                    elif copy_status == 'failed':
                        error_msg = f"Copy operation failed: {copy_props.copy.status_description}"
                        _log.error(error_msg)
                        raise Exception(error_msg)
                    elif copy_status == 'aborted':
                        error_msg = f"Copy operation was aborted: {copy_props.copy.status_description}"
                        _log.error(error_msg)
                        raise Exception(error_msg)
                    elif copy_status == 'pending':
                        # Check timeout
                        elapsed_time = time.time() - start_time
                        if elapsed_time > max_wait_time:
                            # Try to abort the copy operation
                            try:
                                dest_blob_client.abort_copy(copy_operation['copy_id'])
                                _log.error(f"Copy operation timed out after {elapsed_time:.1f} seconds, aborted")
                            except:
                                _log.error(f"Copy operation timed out after {elapsed_time:.1f} seconds, could not abort")
                            raise TimeoutError(f"Copy operation timed out after {max_wait_time} seconds")
                        
                        # Wait before checking again
                        time.sleep(2)
                    else:
                        _log.warning(f"Unknown copy status: {copy_status}")
                        time.sleep(2)
                
                except ResourceNotFoundError:
                    # Destination blob doesn't exist yet, continue waiting
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_wait_time:
                        raise TimeoutError(f"Copy operation timed out after {max_wait_time} seconds (blob not created)")
                    time.sleep(2)
                except Exception as e:
                    if "copy" in str(e).lower():
                        # Copy-related error, re-raise
                        raise
                    else:
                        # Other error, log and continue
                        _log.warning(f"Error checking copy status: {e}")
                        time.sleep(2)
                
        except Exception as e:
            _log.error(f"Error copying blob {source_container}/{source_blob} to {dest_container}/{dest_blob}: {e}")
            raise
    
    def move_blob_to_archive(self,
                           source_container: str,
                           blob_name: str,
                           archive_container: str = None,
                           add_timestamp: bool = True,
                           preserve_folder_structure: bool = True) -> str:
        """
        FIXED: Move a blob from source container to archive container with proper error handling
        
        Args:
            source_container: Source container name
            blob_name: Blob name to move
            archive_container: Archive container name (defaults to Config.ARCHIVE_CONTAINER)
            add_timestamp: Whether to add timestamp to archived blob name
            preserve_folder_structure: Whether to preserve folder structure in archive
            
        Returns:
            Archived blob name
        """
        if archive_container is None:
            archive_container = getattr(Config, 'ARCHIVE_CONTAINER', 'archive')
        
        # Ensure both containers exist
        self.ensure_container_exists(source_container)
        self.ensure_container_exists(archive_container)
        
        # Build archive blob name
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_path = Path(blob_name)
            if preserve_folder_structure:
                # Keep folder structure but add timestamp to filename
                archive_blob_name = str(blob_path.parent / f"{blob_path.stem}_{timestamp}{blob_path.suffix}")
            else:
                # Flatten structure and add timestamp
                archive_blob_name = f"{blob_path.stem}_{timestamp}{blob_path.suffix}"
        else:
            archive_blob_name = blob_name
        
        # Ensure forward slashes for blob names
        archive_blob_name = archive_blob_name.replace("\\", "/")
        
        try:
            _log.info(f"Moving blob: {source_container}/{blob_name} -> {archive_container}/{archive_blob_name}")
            
            # Step 1: Copy to archive container (with timeout)
            self.copy_blob_between_containers(
                source_container, 
                blob_name, 
                archive_container, 
                archive_blob_name,
                timeout_seconds=300  # 5 minutes timeout
            )
            
            # Step 2: Verify the copied blob exists and has correct size
            try:
                source_client = self.blob_service_client.get_blob_client(
                    container=source_container, blob=blob_name
                )
                dest_client = self.blob_service_client.get_blob_client(
                    container=archive_container, blob=archive_blob_name
                )
                
                source_props = source_client.get_blob_properties()
                dest_props = dest_client.get_blob_properties()
                
                if source_props.size != dest_props.size:
                    raise Exception(f"Size mismatch: source={source_props.size}, dest={dest_props.size}")
                
                _log.debug(f"✓ Verified copied blob size: {dest_props.size} bytes")
                
            except Exception as e:
                _log.error(f"Error verifying copied blob: {e}")
                raise Exception(f"Copy verification failed: {e}")
            
            # Step 3: Delete from source container only if copy was successful
            try:
                self.delete_blob(source_container, blob_name)
                _log.info(f"✓ Successfully moved {source_container}/{blob_name} to {archive_container}/{archive_blob_name}")
            except Exception as e:
                _log.error(f"Error deleting source blob after copy: {e}")
                # Don't raise here as the copy was successful
                _log.warning(f"File was copied to archive but could not be deleted from source: {e}")
            
            return archive_blob_name
            
        except Exception as e:
            _log.error(f"Error moving blob {source_container}/{blob_name} to archive: {e}")
            # Try to clean up partial copy if it exists
            try:
                dest_client = self.blob_service_client.get_blob_client(
                    container=archive_container, blob=archive_blob_name
                )
                if dest_client.exists():
                    dest_client.delete_blob()
                    _log.info(f"Cleaned up partial copy: {archive_container}/{archive_blob_name}")
            except:
                pass
            raise
    
    def archive_processed_files(self,
                               processed_blob_names: List[str],
                               source_container: str = None,
                               archive_container: str = None,
                               add_timestamp: bool = True) -> Dict[str, str]:
        """
        FIXED: Archive multiple processed files with better error handling and progress tracking
        
        Args:
            processed_blob_names: List of blob names that were processed
            source_container: Source container (defaults to Config.INPUT_CONTAINER)
            archive_container: Archive container (defaults to Config.ARCHIVE_CONTAINER)
            add_timestamp: Whether to add timestamp to archived files
            
        Returns:
            Dict mapping original blob names to archived blob names
        """
        if source_container is None:
            source_container = Config.INPUT_CONTAINER
        if archive_container is None:
            archive_container = getattr(Config, 'ARCHIVE_CONTAINER', 'archive')
        
        archived_files = {}
        successful_archives = 0
        failed_archives = 0
        
        _log.info(f"Starting to archive {len(processed_blob_names)} processed files...")
        _log.info(f"Source container: {source_container}")
        _log.info(f"Archive container: {archive_container}")
        
        # Ensure containers exist
        self.ensure_container_exists(source_container)
        self.ensure_container_exists(archive_container)
        
        for i, blob_name in enumerate(processed_blob_names, 1):
            try:
                _log.info(f"Archiving file {i}/{len(processed_blob_names)}: {blob_name}")
                
                # Check if source file exists before attempting to move
                source_client = self.blob_service_client.get_blob_client(
                    container=source_container, blob=blob_name
                )
                
                if not source_client.exists():
                    _log.warning(f"Source file does not exist, skipping: {source_container}/{blob_name}")
                    failed_archives += 1
                    continue
                
                archived_name = self.move_blob_to_archive(
                    source_container,
                    blob_name,
                    archive_container,
                    add_timestamp
                )
                archived_files[blob_name] = archived_name
                successful_archives += 1
                
                _log.info(f"✓ [{i}/{len(processed_blob_names)}] Archived: {blob_name} -> {archived_name}")
                
            except Exception as e:
                _log.error(f"❌ [{i}/{len(processed_blob_names)}] Failed to archive {blob_name}: {e}")
                failed_archives += 1
                # Continue with other files even if one fails
                continue
        
        _log.info(f"Archive operation completed: {successful_archives} successful, {failed_archives} failed")
        
        if successful_archives > 0:
            _log.info("Successfully archived files:")
            for original, archived in list(archived_files.items())[:5]:  # Show first 5
                _log.info(f"  {original} -> {archived}")
            if len(archived_files) > 5:
                _log.info(f"  ... and {len(archived_files) - 5} more files")
        
        if failed_archives > 0:
            _log.warning(f"Failed to archive {failed_archives} files. Check logs for details.")
        
        return archived_files
    
    def upload_directory_contents(self, 
                                 local_dir: Path, 
                                 container_name: str,
                                 prefix: str = "") -> List[str]:
        """
        Upload all files from a local directory to blob container
        
        Args:
            local_dir: Local directory path
            container_name: Target container name
            prefix: Optional prefix for blob names
            
        Returns:
            List of uploaded blob names
        """
        uploaded_blobs = []
        
        if not local_dir.exists():
            _log.warning(f"Directory {local_dir} does not exist")
            return uploaded_blobs
            
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                # Create blob name with prefix
                relative_path = file_path.relative_to(local_dir)
                blob_name = f"{prefix}{relative_path}".replace("\\", "/") if prefix else str(relative_path).replace("\\", "/")
                
                try:
                    self.upload_file_to_blob(file_path, container_name, blob_name)
                    uploaded_blobs.append(blob_name)
                except Exception as e:
                    _log.error(f"Failed to upload {file_path}: {e}")
                    
        _log.info(f"Uploaded {len(uploaded_blobs)} files to {container_name}")
        return uploaded_blobs
    
    def upload_images_flat(self, local_images_dir: Path) -> List[str]:
        """
        Upload all images to the images container with flat structure (no subfolders)
        
        Args:
            local_images_dir: Local directory containing images
            
        Returns:
            List of uploaded image blob names
        """
        uploaded_images = []
        
        if not local_images_dir.exists():
            _log.warning(f"Images directory {local_images_dir} does not exist")
            return uploaded_images
            
        for image_path in local_images_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg']:
                
                try:
                    self.upload_file_to_blob(image_path, Config.IMAGES_CONTAINER, image_path.name)
                    uploaded_images.append(image_path.name)
                    _log.debug(f"Uploaded image: {image_path.name} -> {image_path.name}")
                except Exception as e:
                    _log.error(f"Failed to upload image {image_path}: {e}")
                    
        _log.info(f"Uploaded {len(uploaded_images)} images to {Config.IMAGES_CONTAINER}")
        return uploaded_images
    
    def ensure_container_exists(self, container_name: str):
        """FIXED: Ensure a container exists, create if it doesn't with better error handling"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # Check if container exists
            if container_client.exists():
                _log.debug(f"Container already exists: {container_name}")
                return
            
            # Create container
            container_client.create_container()
            _log.info(f"Created container: {container_name}")
            
        except ResourceExistsError:
            # Container already exists (race condition)
            _log.debug(f"Container already exists (race condition): {container_name}")
        except Exception as e:
            _log.error(f"Error ensuring container {container_name} exists: {e}")
            # Don't raise here as this might be a permissions issue but container might exist
            try:
                # Try to access the container to see if it exists
                container_client = self.blob_service_client.get_container_client(container_name)
                container_client.get_container_properties()
                _log.info(f"Container exists but couldn't create: {container_name}")
            except:
                # Container really doesn't exist and we can't create it
                _log.error(f"Cannot access or create container: {container_name}")
                raise
    
    def list_blobs(self, container_name: str, prefix: str = None) -> List[Dict[str, Any]]:
        """
        List all blobs in a container with their metadata
        
        Args:
            container_name: Container name
            prefix: Optional prefix filter
            
        Returns:
            List of dictionaries containing blob information
        """
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blob_list = container_client.list_blobs(name_starts_with=prefix)
            
            blobs_info = []
            for blob in blob_list:
                blobs_info.append({
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified,
                    'content_type': blob.content_settings.content_type if blob.content_settings else None,
                    'metadata': blob.metadata or {}
                })
            
            return blobs_info
        except Exception as e:
            _log.error(f"Error listing blobs in {container_name}: {e}")
            return []
    
    def delete_blob(self, container_name: str, blob_name: str):
        """FIXED: Delete a blob from container with better error handling"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Check if blob exists before attempting to delete
            if not blob_client.exists():
                _log.warning(f"Blob does not exist, cannot delete: {container_name}/{blob_name}")
                return
            
            blob_client.delete_blob()
            _log.debug(f"✓ Deleted blob: {container_name}/{blob_name}")
            
        except ResourceNotFoundError:
            _log.warning(f"Blob not found during deletion: {container_name}/{blob_name}")
        except Exception as e:
            _log.error(f"Error deleting blob {container_name}/{blob_name}: {e}")
            raise
    
    def cleanup_temp_files(self, temp_paths: Dict[str, Path]):
        """Clean up temporary directories"""
        for path_type, path in temp_paths.items():
            if path.exists():
                try:
                    shutil.rmtree(path)
                    _log.debug(f"Cleaned up temp {path_type}: {path}")
                except Exception as e:
                    _log.warning(f"Error cleaning up {path}: {e}")

    def get_blob_metadata(self, container_name: str, blob_name: str) -> Dict[str, Any]:
        """Get metadata for a specific blob"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            properties = blob_client.get_blob_properties()
            return {
                'name': blob_name,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'content_type': properties.content_settings.content_type,
                'metadata': properties.metadata
            }
        except Exception as e:
            _log.error(f"Error getting metadata for {container_name}/{blob_name}: {e}")
            return {}

    def list_archived_files(self, archive_container: str = None, prefix: str = None) -> List[Dict[str, Any]]:
        """
        List archived files with their metadata
        
        Args:
            archive_container: Archive container name
            prefix: Optional prefix filter
            
        Returns:
            List of dictionaries containing file information
        """
        if archive_container is None:
            archive_container = getattr(Config, 'ARCHIVE_CONTAINER', 'archive')
        
        archived_files = []
        blobs_info = self.list_blobs(archive_container, prefix)
        
        for blob_info in blobs_info:
            archived_files.append(blob_info)
        
        return archived_files
    
    def verify_blob_exists(self, container_name: str, blob_name: str) -> bool:
        """
        ADDED: Verify if a blob exists in the container
        
        Args:
            container_name: Container name
            blob_name: Blob name
            
        Returns:
            True if blob exists, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            return blob_client.exists()
        except Exception as e:
            _log.error(f"Error checking if blob exists {container_name}/{blob_name}: {e}")
            return False