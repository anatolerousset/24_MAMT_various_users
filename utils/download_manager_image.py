"""
Image download manager for handling blob storage images in retrieved chunks.
"""
import logging
import re
import shutil
from pathlib import Path
from typing import List, Set, Dict
from azure.core.exceptions import ResourceNotFoundError

from config.config import Config
from utils.blob_utils import BlobStorageManager

_log = logging.getLogger(__name__)

class ImageDownloadManager:
    """Manages downloading images from blob storage to public folder for display."""
    
    def __init__(self, public_folder: Path = None):
        """
        Initialize the image download manager.
        
        Args:
            public_folder: Path to the public folder (defaults to ./public)
        """
        self.public_folder = public_folder or Path("./public")
        self.blob_manager = BlobStorageManager()
        
        # Ensure public folder exists
        self.public_folder.mkdir(exist_ok=True)
        
        # Pattern to match markdown images
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        
        _log.info(f"ImageDownloadManager initialized with public folder: {self.public_folder.absolute()}")
    
    def cleanup_previous_images(self):
        """
        Delete all previously downloaded images from the public folder.
        Only deletes image files to avoid removing other static content.
        """
        try:
            image_files = list(self.public_folder.glob("image_*"))
            
            if image_files:
                _log.info(f"Cleaning up {len(image_files)} previous images from public folder...")
                
                for image_file in image_files:
                    try:
                        image_file.unlink()
                        _log.debug(f"Deleted: {image_file.name}")
                    except Exception as e:
                        _log.warning(f"Failed to delete {image_file.name}: {e}")
                        
                _log.info(f"✓ Cleaned up {len(image_files)} previous images")
            else:
                _log.debug("No previous images to clean up")
                
        except Exception as e:
            _log.error(f"Error during image cleanup: {e}")
    
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
        
        _log.info(f"Found {len(image_paths)} unique images in chunks: {list(image_paths)}")
        return image_paths
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if filename has an image extension."""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'}
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def download_images_from_blob(self, image_filenames: Set[str]) -> Dict[str, str]:
        """
        Download images from blob storage to public folder.
        
        Args:
            image_filenames: Set of image filenames to download
            
        Returns:
            Dict mapping original filename to downloaded filename
        """
        if not image_filenames:
            _log.info("No images to download")
            return {}
        
        downloaded_mapping = {}
        successful_downloads = 0
        failed_downloads = 0
        
        _log.info(f"Starting download of {len(image_filenames)} images...")
        
        for filename in image_filenames:
            try:
                # Use the original filename (no prefix needed)
                local_path = self.public_folder / filename
                
                # Check if file already exists and is recent
                if local_path.exists():
                    _log.debug(f"Image already exists, skipping: {filename}")
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
                _log.debug(f"Downloaded: {filename} -> {filename}")
                
            except ResourceNotFoundError:
                _log.warning(f"Image not found in blob storage: {filename}")
                failed_downloads += 1
            except Exception as e:
                _log.error(f"Failed to download {filename}: {e}")
                failed_downloads += 1
        
        _log.info(f"Image download completed: {successful_downloads} successful, {failed_downloads} failed")
        return downloaded_mapping
    
    def update_chunk_image_paths(self, chunks: List, image_mapping: Dict[str, str]) -> List:
        """
        Update image paths in chunks to point to the public folder.
        Note: This creates public/ references that will be converted to HTTP URLs by the backend.
        
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
        
        _log.info(f"Updated image paths in {len(updated_chunks)} chunks")
        return updated_chunks
    
    def _replace_image_paths_in_content(self, content: str, image_mapping: Dict[str, str]) -> str:
        """
        Replace image paths in content with public folder paths.
        
        Args:
            content: Original content with image references
            image_mapping: Mapping of original to downloaded filenames
            
        Returns:
            Updated content with public/ paths (will be converted to HTTP URLs later)
        """
        def replace_image_path(match):
            alt_text = match.group(1)
            original_path = match.group(2).strip()
            """
            # Skip if already processed or external
            if original_path.startswith(('public/', 'http://', 'https://', 'data:', '/images/')):
                return match.group(0)
            """
            # Extract filename from path
            if '/' in original_path:
                filename = original_path.split('/')[-1]
            else:
                filename = original_path
            
            # Check if we have a mapping for this image
            if filename in image_mapping:
                downloaded_filename = image_mapping[filename]
                # Use public/ prefix - this will be converted to HTTP URL by backend
                new_path = f"public/{downloaded_filename}"
                return f"![{alt_text}]({new_path})"
            
            # Return original if no mapping found
            return match.group(0)
        
        updated_content = self.image_pattern.sub(replace_image_path, content)
        
        # Debug logging to track changes
        if updated_content != content:
            original_images = set(self.image_pattern.findall(content))
            updated_images = set(self.image_pattern.findall(updated_content))
            _log.debug(f"Image path update: {len(original_images)} -> {len(updated_images)} images")
        
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
            _log.debug("No chunks provided for image processing")
            return chunks
        
        try:
            _log.info(f"Processing {len(chunks)} chunks for images...")
            
            # Step 1: Extract image paths from chunks
            image_paths = self.extract_image_paths_from_chunks(chunks)
            
            if not image_paths:
                _log.info("No images found in chunks")
                return chunks
            
            # Step 2: Download images from blob storage
            image_mapping = self.download_images_from_blob(image_paths)
            
            if not image_mapping:
                _log.warning("No images were downloaded successfully")
                return chunks
            
            # Step 3: Update chunk image paths
            updated_chunks = self.update_chunk_image_paths(chunks, image_mapping)
            
            _log.info(f"✓ Successfully processed {len(updated_chunks)} chunks with {len(image_mapping)} images")
            return updated_chunks
            
        except Exception as e:
            _log.error(f"Error processing chunks with images: {e}")
            import traceback
            traceback.print_exc()
            return chunks  # Return original chunks on error
    
    def get_downloaded_images_info(self) -> List[Dict[str, any]]:
        """
        Get information about currently downloaded images in the public folder.
        
        Returns:
            List of dictionaries containing image information
        """
        images_info = []
        
        try:
            image_files = list(self.public_folder.glob("image_*"))
            
            for image_file in image_files:
                try:
                    stat = image_file.stat()
                    images_info.append({
                        'filename': image_file.name,
                        'path': str(image_file),
                        'url': f"/images/{image_file.name}",  # HTTP URL for access
                        'size_bytes': stat.st_size,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'modified': stat.st_mtime
                    })
                except Exception as e:
                    _log.warning(f"Error getting info for {image_file.name}: {e}")
            
            images_info.sort(key=lambda x: x['filename'])
            _log.debug(f"Found {len(images_info)} images in public folder")
            
        except Exception as e:
            _log.error(f"Error getting downloaded images info: {e}")
        
        return images_info
    
    def verify_image_access(self, filename: str) -> Dict[str, any]:
        """
        Verify that an image exists and can be accessed.
        
        Args:
            filename: Name of the image file
            
        Returns:
            Dict with verification results
        """
        image_path = self.public_folder / filename
        
        result = {
            'filename': filename,
            'exists': False,
            'accessible': False,
            'url': f"/images/{filename}",
            'local_path': str(image_path),
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


# Global instance
_image_download_manager = None

def get_image_download_manager() -> ImageDownloadManager:
    """Get or create the global image download manager instance."""
    global _image_download_manager
    if _image_download_manager is None:
        _image_download_manager = ImageDownloadManager()
    return _image_download_manager


# Convenience functions
def cleanup_previous_images():
    """Cleanup previous images from public folder."""
    manager = get_image_download_manager()
    manager.cleanup_previous_images()


def process_chunks_with_images(chunks: List) -> List:
    """
    Process chunks to download and update image paths.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        List of updated chunks with corrected image paths
    """
    manager = get_image_download_manager()
    return manager.process_chunks_with_images(chunks)


def get_downloaded_images_info() -> List[Dict[str, any]]:
    """Get information about downloaded images."""
    manager = get_image_download_manager()
    return manager.get_downloaded_images_info()


def verify_image_access(filename: str) -> Dict[str, any]:
    """Verify that an image can be accessed via HTTP."""
    manager = get_image_download_manager()
    return manager.verify_image_access(filename)