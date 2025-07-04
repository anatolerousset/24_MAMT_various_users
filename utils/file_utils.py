import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import re

from utils.blob_utils import BlobStorageManager
from config.config import Config

_log = logging.getLogger(__name__)

def convert_xls_to_xlsx(xls_file_path: Path) -> Path:
    """
    Convert XLS file to XLSX format using pandas.
    
    Args:
        xls_file_path: Path to the XLS file
        
    Returns:
        Path to the converted XLSX file
    """
    try:
        _log.info(f"Converting {xls_file_path.name} from XLS to XLSX format...")
        
        # Define the output path for the XLSX file
        xlsx_file_path = xls_file_path.with_suffix('.xlsx')
        
        # Read the XLS file using pandas
        # Using engine='xlrd' as it works better with older XLS files
        xls_data = pd.read_excel(xls_file_path, engine='xlrd', sheet_name=None)
        
        # Write to XLSX format
        with pd.ExcelWriter(xlsx_file_path, engine='openpyxl') as writer:
            for sheet_name, df in xls_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        _log.info(f"✓ Successfully converted to {xlsx_file_path.name}")
        return xlsx_file_path
    
    except Exception as e:
        _log.error(f"❌ Error converting {xls_file_path.name}: {str(e)}")
        # If conversion fails, return the original file path
        return xls_file_path

def format_headings(headings):
    """
    Format headings into a hierarchical structure for prepending to content.
    
    Args:
        headings: List of headings from the document metadata
        
    Returns:
        Formatted headings string
    """
    if not headings:
        return ""
    
    heading_text = "HEADINGS:\n"
    for i, heading in enumerate(headings):
        heading_text += f"{i+1}. {heading}\n"
    
    heading_text += "\nCONTENT:\n"
    return heading_text

def fix_image_paths(content):
    """
    Convert Windows-style image paths to proper forward slash Markdown links.
    Handles both regular backslash paths and URL-encoded paths (%20).
    
    Args:
        content: String content with image references
        
    Returns:
        String with fixed image paths
    """
    import re
    
    # Define regex patterns to match Markdown image links with problematic paths
    backslash_pattern = r'!\[Image\]\((.*?\\.*?)\)'
    url_encoded_pattern = r'!\[Image\]\((.*? .*?)\)'
    
    # Fix backslash paths
    def replace_backslashes(match):
        path = match.group(1)
        fixed_path = path.replace('\\', '/')
        return f'![Image]({fixed_path})'
    
    # Fix URL-encoded spaces
    def replace_url_encoded(match):
        path = match.group(1)
        fixed_path = path.replace(' ', '%20')
        return f'![Image]({fixed_path})'
    
    # Apply the fixes
    content = re.sub(backslash_pattern, replace_backslashes, content)
    content = re.sub(url_encoded_pattern, replace_url_encoded, content)
    
    return content

def get_input_files_from_blob(container_name: str = None, file_extensions: List[str] = None) -> List[str]:
    """
    Get list of input files from blob storage
    
    Args:
        container_name: Container to search (defaults to INPUT_CONTAINER)
        file_extensions: List of allowed extensions (e.g., ['.pdf', '.docx', '.xlsx'])
        
    Returns:
        List of blob names
    """
    if container_name is None:
        container_name = Config.INPUT_CONTAINER
        
    if file_extensions is None:
        file_extensions = ['.pdf', '.docx', '.xlsx', '.xls']
    
    blob_manager = BlobStorageManager()
    all_blobs_info = blob_manager.list_blobs(container_name)
    
    # Filter by file extensions
    # The list_blobs method returns a list of dictionaries with blob info
    filtered_blobs = []
    for blob_info in all_blobs_info:
        # Extract the blob name from the dictionary
        blob_name = blob_info['name']
        if any(blob_name.lower().endswith(ext.lower()) for ext in file_extensions):
            filtered_blobs.append(blob_name)
    
    _log.info(f"Found {len(filtered_blobs)} input files in {container_name}")
    return filtered_blobs