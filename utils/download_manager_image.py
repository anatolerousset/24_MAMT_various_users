"""
Image download manager for handling blob storage images in retrieved chunks with user support.
"""
import logging
import re
import shutil
from pathlib import Path
from typing import List, Set, Dict, Optional
from azure.core.exceptions import ResourceNotFoundError

from config.config import Config
from utils.blob_utils import BlobStorageManager

_log = logging.getLogger(__name__)

class UserImageDownloadManager:
    """Manages downloading images from blob storage to user-specific public folders for display."""
    
    def __init__(self, user_session_id: str, public_folder: Path = None):
        """
        Initialize the user-specific image download manager.
        
        Args:
            user_session_id: User session ID
            public_folder: Base public folder (defaults to ./public)
        """
        self.user_session_id = user_session_id
        self.base_public_folder = public_folder or Path("./public")
        
        # Create user-specific public folder
        self.user_public_folder = Config.get_user_public_folder(user_session_id)
        
        self.blob_manager = BlobStorageManager()
        
        # Pattern to match markdown images
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        
        _log.info(f"UserImageDownloadManager initialized for user {user_session_id} with folder: {self.user_public_folder.absolute()}")
    
    def cleanup_user_images(self):
        """
        Delete all previously downloaded images from the user's public folder.
        Only deletes image files to avoid removing other static content.
        """
        try:
            if not self.user_public_folder.exists():
                _log.debug(f"User public folder doesn't exist: {self.user_public_folder}")
                return
            
            image_files = list(self.user_public_folder.glob("*"))
            
            if image_files:
                _log.info(f"Cleaning up {len(image_files)} images for user {self.user_session_id}...")
                
                for image_file in image_files:
                    try:
                        if image_file.is_file():
                            image_file.unlink()
                            _log.debug(f"Deleted: {image_file.name}")
                    except Exception as e:
                        _log.warning(f"Failed to delete {image_file.name}: {e}")
                        
                _log.info(f"✓ Cleaned up {len(image_files)} images for user {self.user_session_id}")
            else:
                _log.debug(f"No images to clean up for user {self.user_session_id}")
                
        except Exception as e:
            _log.error(f"Error during image cleanup for user {self.user_session_id}: {e}")
    
    def extract_image_paths_from_chunks(self, chunks: List) -> Set[str]:
        """
        Extract all unique image paths from a list of document chunks.
        
        Args:
            chunks: List of document chunks (LCDocument objects)
            
        Returns:
            Set of unique image paths found in the chunks
        """
        image_paths = set()
        
        for chunk in chunks:
            # Get the page content
            content = getattr(chunk, 'page_content', '')
            if not content:
                continue
                
            # Find all image references in the content
            matches = self.image_pattern.findall(content)
            
            for alt_text, image_path in matches:
                # Clean up the path
                image_path = image_path.strip()
                
                # Skip if it's already a public path or external URL
                if (image_path.startswith(('public/', 'http://', 'https://', 'data:', '/images/'))):
                    continue
                
                # Extract just the filename if it contains folder structure
                if '/' in image_path:
                    # Remove container prefix if present (e.g., "images/filename.png" -> "filename.png")
                    filename = image_path.split('/')[-1]
                else:
                    filename = image_path
                
                if filename and self._is_image_file(filename):
                    image_paths.add(filename)
        
        _log.info(f"Found {len(image_paths)} unique images in chunks for user {self.user_session_id}: {list(image_paths)}")
        return image_paths
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if filename has an image extension."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def download_images_from_blob(self, image_filenames: Set[str]) -> Dict[str, str]:
        """
        Download images from blob storage to user's public folder.
        
        Args:
            image_filenames: Set of image filenames to download
            
        Returns:
            Dict mapping original filename to downloaded filename
        """
        if not image_filenames:
            _log.info(f"No images to download for user {self.user_session_id}")
            return {}
        
        downloaded_mapping = {}
        successful_downloads = 0
        failed_downloads = 0
        
        _log.info(f"Starting download of {len(image_filenames)} images for user {self.user_session_id}...")
        
        for filename in image_filenames:
            try:
                # Use the original filename in user's folder
                local_path = self.user_public_folder / filename
                
                # Check if file already exists and is recent
                if local_path.exists():
                    _log.debug(f"Image already exists for user {self.user_session_id}, skipping: {filename}")
                    downloaded_mapping[filename] = filename
                    successful_downloads += 1
                    continue
                
                # Download from blob storage
                blob_client = self.blob_manager.blob_service_client.get_blob_client(
                    container=Config.IMAGES_CONTAINER,
                    blob=filename
                )
                
                # Download the blob data
                blob_data = blob_client.download_blob().readall()
                
                # Write to local file
                with open(local_path, 'wb') as f:
                    f.write(blob_data)
                
                downloaded_mapping[filename] = filename
                successful_downloads += 1
                _log.debug(f"Downloaded for user {self.user_session_id}: {filename}")
                
            except ResourceNotFoundError:
                _log.warning(f"Image not found in blob storage for user {self.user_session_id}: {filename}")
                failed_downloads += 1
            except Exception as e:
                _log.error(f"Failed to download {filename} for user {self.user_session_id}: {e}")
                failed_downloads += 1
        
        _log.info(f"Image download completed for user {self.user_session_id}: {successful_downloads} successful, {failed_downloads} failed")
        return downloaded_mapping
    
    def update_chunk_image_paths(self, chunks: List, image_mapping: Dict[str, str]) -> List:
        """
        Update image paths in chunks to point to the user's public folder.
        
        Args:
            chunks: List of document chunks
            image_mapping: Dict mapping original filename to downloaded filename
            
        Returns:
            List of updated chunks
        """
        if not image_mapping:
            return chunks
        
        updated_chunks = []
        
        for chunk in chunks:
            # Create a copy of the chunk to avoid modifying the original
            updated_chunk = type(chunk)(
                page_content=chunk.page_content,
                metadata=chunk.metadata.copy() if hasattr(chunk, 'metadata') else {}
            )
            
            # Update image paths in the content
            updated_content = self._replace_image_paths_in_content(
                updated_chunk.page_content, 
                image_mapping
            )
            
            updated_chunk.page_content = updated_content
            updated_chunks.append(updated_chunk)
        
        _log.info(f"Updated image paths in {len(updated_chunks)} chunks for user {self.user_session_id}")
        return updated_chunks
    
    def _replace_image_paths_in_content(self, content: str, image_mapping: Dict[str, str]) -> str:
        """
        Replace image paths in content with user-specific public folder paths.
        
        Args:
            content: Original content with image references
            image_mapping: Mapping of original to downloaded filenames
            
        Returns:
            Updated content with user-specific public/ paths
        """
        def replace_image_path(match):
            alt_text = match.group(1)
            original_path = match.group(2).strip()
            
            # Extract filename from path
            if '/' in original_path:
                filename = original_path.split('/')[-1]
            else:
                filename = original_path
            
            # Check if we have a mapping for this image
            if filename in image_mapping:
                downloaded_filename = image_mapping[filename]
                # Use user-specific public path
                new_path = f"public/user_{self.user_session_id}/{downloaded_filename}"
                return f"![{alt_text}]({new_path})"
            
            # Return original if no mapping found
            return match.group(0)
        
        updated_content = self.image_pattern.sub(replace_image_path, content)
        
        # Debug logging to track changes
        if updated_content != content:
            original_images = set(self.image_pattern.findall(content))
            updated_images = set(self.image_pattern.findall(updated_content))
            _log.debug(f"Image path update for user {self.user_session_id}: {len(original_images)} -> {len(updated_images)} images")
        
        return updated_content
    
    def process_chunks_with_images(self, chunks: List) -> List:
        """
        Complete process: extract images, download them, and update chunk paths.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of updated chunks with corrected image paths
        """
        if not chunks:
            _log.debug(f"No chunks provided for image processing for user {self.user_session_id}")
            return chunks
        
        try:
            _log.info(f"Processing {len(chunks)} chunks for images for user {self.user_session_id}...")
            
            # Step 1: Extract image paths from chunks
            image_paths = self.extract_image_paths_from_chunks(chunks)
            
            if not image_paths:
                _log.info(f"No images found in chunks for user {self.user_session_id}")
                return chunks
            
            # Step 2: Download images from blob storage
            image_mapping = self.download_images_from_blob(image_paths)
            
            if not image_mapping:
                _log.warning(f"No images were downloaded successfully for user {self.user_session_id}")
                return chunks
            
            # Step 3: Update chunk image paths
            updated_chunks = self.update_chunk_image_paths(chunks, image_mapping)
            
            _log.info(f"✓ Successfully processed {len(updated_chunks)} chunks with {len(image_mapping)} images for user {self.user_session_id}")
            return updated_chunks
            
        except Exception as e:
            _log.error(f"Error processing chunks with images for user {self.user_session_id}: {e}")
            import traceback
            traceback.print_exc()
            return chunks  # Return original chunks on error
    
    def get_downloaded_images_info(self) -> List[Dict[str, any]]:
        """
        Get information about currently downloaded images in the user's public folder.
        
        Returns:
            List of dictionaries containing image information
        """
        images_info = []
        
        try:
            if not self.user_public_folder.exists():
                return images_info
            
            image_files = list(self.user_public_folder.glob("*"))
            
            for image_file in image_files:
                try:
                    if image_file.is_file():
                        stat = image_file.stat()
                        images_info.append({
                            'filename': image_file.name,
                            'path': str(image_file),
                            'url': f"/images/user_{self.user_session_id}/{image_file.name}",  # HTTP URL for access
                            'size_bytes': stat.st_size,
                            'size_mb': round(stat.st_size / (1024 * 1024), 2),
                            'modified': stat.st_mtime,
                            'user_session_id': self.user_session_id
                        })
                except Exception as e:
                    _log.warning(f"Error getting info for {image_file.name}: {e}")
            
            images_info.sort(key=lambda x: x['filename'])
            _log.debug(f"Found {len(images_info)} images in user {self.user_session_id} public folder")
            
        except Exception as e:
            _log.error(f"Error getting downloaded images info for user {self.user_session_id}: {e}")
        
        return images_info
    
    def verify_image_access(self, filename: str) -> Dict[str, any]:
        """
        Verify that an image exists and can be accessed for this user.
        
        Args:
            filename: Name of the image file
            
        Returns:
            Dict with verification results
        """
        image_path = self.user_public_folder / filename
        
        result = {
            'filename': filename,
            'exists': False,
            'accessible': False,
            'url': f"/images/user_{self.user_session_id}/{filename}",
            'local_path': str(image_path),
            'user_session_id': self.user_session_id,
            'error': None
        }
        
        try:
            if image_path.exists():
                result['exists'] = True
                
                # Check if readable
                with open(image_path, 'rb') as f:
                    f.read(1)  # Try to read first byte
                result['accessible'] = True
                
                # Get file info
                stat = image_path.stat()
                result['size_bytes'] = stat.st_size
                result['size_mb'] = round(stat.st_size / (1024 * 1024), 2)
                
            else:
                result['error'] = "File does not exist"
                
        except Exception as e:
            result['error'] = str(e)
            
        return result


# User manager instances
_user_image_managers = {}

def get_user_image_download_manager(user_session_id: str) -> UserImageDownloadManager:
    """Get or create a user-specific image download manager instance."""
    global _user_image_managers
    if user_session_id not in _user_image_managers:
        _user_image_managers[user_session_id] = UserImageDownloadManager(user_session_id)
    return _user_image_managers[user_session_id]


def cleanup_user_image_manager(user_session_id: str):
    """Cleanup and remove a user's image manager."""
    global _user_image_managers
    if user_session_id in _user_image_managers:
        manager = _user_image_managers[user_session_id]
        manager.cleanup_user_images()
        del _user_image_managers[user_session_id]
        _log.info(f"Cleaned up image manager for user {user_session_id}")


# Convenience functions with user support
def cleanup_previous_images(user_session_id: str = None):
    """Cleanup previous images from public folder."""
    if user_session_id:
        manager = get_user_image_download_manager(user_session_id)
        manager.cleanup_user_images()
    else:
        # Cleanup all users (for backward compatibility)
        base_public = Path("./public")
        if base_public.exists():
            for user_folder in base_public.glob("user_*"):
                if user_folder.is_dir():
                    try:
                        shutil.rmtree(user_folder)
                        _log.info(f"Cleaned up folder: {user_folder}")
                    except Exception as e:
                        _log.error(f"Error cleaning up {user_folder}: {e}")


def process_chunks_with_images(chunks: List, user_session_id: str) -> List:
    """
    Process chunks to download and update image paths for a specific user.
    
    Args:
        chunks: List of document chunks
        user_session_id: User session ID
        
    Returns:
        List of updated chunks with corrected image paths
    """
    manager = get_user_image_download_manager(user_session_id)
    return manager.process_chunks_with_images(chunks)


def get_downloaded_images_info(user_session_id: str) -> List[Dict[str, any]]:
    """Get information about downloaded images for a specific user."""
    manager = get_user_image_download_manager(user_session_id)
    return manager.get_downloaded_images_info()


def verify_image_access(filename: str, user_session_id: str) -> Dict[str, any]:
    """Verify that an image can be accessed via HTTP for a specific user."""
    manager = get_user_image_download_manager(user_session_id)
    return manager.verify_image_access(filename)


# Legacy functions for backward compatibility
def get_image_download_manager():
    """Legacy function - returns a default manager"""
    return get_user_image_download_manager("default")