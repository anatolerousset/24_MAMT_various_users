"""
Backend DOCX export utilities - removed all Chainlit dependencies
Only handles DOCX generation and file path return
"""
import os
import re
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, List, Set, Union
from pathlib import Path
from langchain_core.documents import Document as LCDocument
from utils.document_processing import process_markdown_images, get_document_date

import logging
_log = logging.getLogger(__name__)


class BackendDocxExporter:
    """DOCX exporter for backend - no Chainlit dependencies."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the DOCX exporter with optional output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("exports")
        self.output_dir.mkdir(exist_ok=True)
        
        if not self._check_pandoc():
            raise RuntimeError("pandoc is not installed or not in PATH. Please install pandoc.")
    
    def export_to_bytes(
        self, 
        markdown_content: str, 
        query: str, 
        sources: set = None,
        technical_docs: List[LCDocument] = None,
        technical_scores: List[float] = None,
        technical_origins: List[str] = None,
        dce_docs: List[LCDocument] = None,
        dce_scores: List[float] = None,
        dce_origins: List[str] = None,
        collection_name: str = "Collection Technique",
        dce_collection: str = "Collection DCE",
        include_retrieved_docs: bool = True,
        include_images_catalog: bool = True,
        filename: Optional[str] = None
    ) -> dict:
        """
        Export to DOCX and return file bytes for direct download.
        
        Returns:
            dict: File information with bytes content
        """
        # Use temporary directory for this export
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if not filename.endswith('.docx'):
                filename += '.docx'
            
            output_path = temp_path / filename
            
            # Build the complete document content (same as export_to_file)
            full_content = ""
            
            # Add document header
            timestamp = datetime.now().strftime("%d/%m/%Y à %H:%M")
            full_content += self._create_document_header(query, timestamp, sources or set())
            
            # Add main response content
            cleaned_content = self._remove_code_cell_markers(markdown_content)
            processed_content = self._process_images_for_pandoc(cleaned_content)
            full_content += processed_content
            
            # Collect all unique images from all content
            all_images = set()
            all_images.update(self._extract_unique_images(processed_content))
            _log.info(f"all images au debut: {all_images}")

            # Add retrieved documents section if requested
            if (technical_docs or dce_docs):
                retrieved_section = self._format_retrieved_documents_section(
                    technical_docs or [], 
                    technical_scores or [],
                    technical_origins or [],
                    dce_docs or [], 
                    dce_scores or [],
                    dce_origins or [],
                    collection_name,
                    dce_collection
                )
                # Add images from retrieved documents
                all_images.update(self._extract_unique_images(retrieved_section))

            if include_retrieved_docs:
                full_content += retrieved_section
            
            _log.info(f"all_images: {all_images}")
            # Add images catalog if requested
            if include_images_catalog and all_images:
                images_section = self._create_images_catalog_section(all_images)
                full_content += images_section

            _log.info(f"FULL CONTENT AVANT DETRE ENVOY2 à PANDOC: {full_content}")
            
            # Create temporary markdown file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_md:
                temp_md.write(full_content)
                temp_md_path = temp_md.name
            
            try:
                # Run pandoc to convert markdown to DOCX
                pandoc_cmd = [
                    'pandoc',
                    temp_md_path,
                    '-o', str(output_path),
                    '--from', 'markdown',
                    '--to', 'docx',
                    '--standalone',
                ]
                
                # Add reference doc if available
                if self._has_reference_doc():
                    pandoc_cmd.extend(['--reference-doc', self._get_reference_doc()])
                
                result = subprocess.run(
                    pandoc_cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Read file bytes
                with open(output_path, 'rb') as f:
                    file_bytes = f.read()
                
                return {
                    "success": True,
                    "filename": filename,
                    "file_bytes": file_bytes,
                    "file_size": len(file_bytes),
                    "metadata": {
                        "include_retrieved_docs": include_retrieved_docs,
                        "include_images_catalog": include_images_catalog,
                        "technical_docs_count": len(technical_docs) if technical_docs else 0,
                        "dce_docs_count": len(dce_docs) if dce_docs else 0,
                        "images_count": len(all_images),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Pandoc conversion failed: {e.stderr}"
                print(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "filename": filename
                }
            finally:
                # Clean up temporary markdown file
                try:
                    os.unlink(temp_md_path)
                except:
                    pass
    
    def _check_pandoc(self) -> bool:
        """Check if pandoc is available in the system."""
        try:
            subprocess.run(
                ["pandoc", "--version"], 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Create a safe filename from text."""
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('._')
        
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('._')
        
        return sanitized or "document"
    
    def _remove_code_cell_markers(self, content: str) -> str:
        """Remove code cell markers from the content."""
        content = re.sub(r'^```[\w]*\s*\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n\n\n+', '\n\n', content)
        return content.strip()
    
    def _process_images_for_pandoc(self, markdown_content: str) -> str:
        """Process image paths in markdown for pandoc compatibility."""
        def replace_image_path(match):
            alt_text = match.group(1)
            image_path = match.group(2)
            if not os.path.exists(image_path):
                return f"Image introuvable."
            if not os.path.isabs(image_path):
                if image_path.startswith('./'):
                    image_path = image_path[2:]
                abs_path = os.path.abspath(os.path.join(os.getcwd(), image_path))
                return f"![{alt_text}]({abs_path})"
            
            return match.group(0)
        
        processed = re.sub(r'!\[(.*?)\]\(([^)]+)\)', replace_image_path, markdown_content)
        return processed
    
    def _create_document_header(self, query: str, timestamp: str, sources: set) -> str:
        """Create a document header with metadata."""
        header = f"""# Réponse Technique

**Sujet:** {query}

**Date de génération:** {timestamp}

**Sources utilisées:**
"""
        
        if sources:
            for source, page in sources:
                if page not in ["","N/A"]:
                    header += f"/ {source} (page {page})\n"
                else:
                    header += f"/ {source}\n"
        else:
            header += "- Aucune source spécifique\n"
        
        header += "\n---\n\n"
        return header

    def _extract_content_after_marker(self, content: str, marker: str = "CONTENT:") -> str:
        """Extract content that comes after a specific marker and clean it."""
        # Find the marker (case insensitive)
        marker_lower = marker.lower()
        content_lower = content.lower()
        
        marker_pos = content_lower.find(marker_lower)
        if marker_pos != -1:
            # Extract everything after the marker
            content = content[marker_pos + len(marker):].strip()
        
        # Remove markdown code block markers
        # Remove opening markers like ```python, ```markdown, etc.
        content = re.sub(r'^```[\w]*\s*\n', '', content, flags=re.MULTILINE)
        # Remove closing markers
        content = re.sub(r'\n```\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*```\s*$', '', content, flags=re.MULTILINE)
        
        # Clean up extra newlines
        content = re.sub(r'\n\n\n+', '\n\n', content)
        
        return content.strip()
        
    def _format_retrieved_documents_section(
        self, 
        technical_docs: List[LCDocument], 
        technical_scores: List[float],
        technical_origins: List[str],
        dce_docs: List[LCDocument], 
        dce_scores: List[float],
        dce_origins: List[str],
        collection_name: str,
        dce_collection: str
    ) -> str:
        """Format retrieved documents section for DOCX export."""
        content = "\n\n# Documents Récupérés\n\n"
        
        if technical_docs:
            content += f"## Collection: {collection_name}\n\n"
        
            for i, doc in enumerate(technical_docs):
                # Extract content after "CONTENT:" marker and clean it
                cleaned_content = self._extract_content_after_marker(doc.page_content, "CONTENT:")
                
                # Process markdown images in the cleaned content
                processed_content = self._process_images_for_pandoc(
                    process_markdown_images(cleaned_content)
                )
            
                # Extract document metadata
                date_str = get_document_date(doc)
                source = doc.metadata.get('source', 'Inconnu')
                author = doc.metadata.get('dl_meta', {}).get('origin', {}).get('author', 'Inconnu')
                headings = doc.metadata.get('dl_meta', {}).get('headings', [])
            
                # Format document section
                content += f"### Document {i+1}\n\n"
                content += f"**Score de similarité:** {technical_scores[i]:.4f}\n\n"
                content += f"**Source:** {source}\n\n"
                content += f"**Auteur:** {author}\n\n"
                content += f"**Origine du vecteur:** {technical_origins[i] if i < len(technical_origins) else 'Non spécifié'}\n\n"
            
                if headings and isinstance(headings, list):
                    content += f"**Titres:** {' > '.join(headings)}\n\n"
            
                content += f"**Date de modification:** {date_str}\n\n"
                content += "**Contenu:**\n\n"
                content += processed_content + "\n\n"
                content += "---\n\n"

        # Format DCE documents
        if dce_docs:
            content += f"## Collection: {dce_collection}\n\n"
        
            for i, doc in enumerate(dce_docs):
                # Extract content after "CONTENT:" marker and clean it
                cleaned_content = self._extract_content_after_marker(doc.page_content, "CONTENT:")
                
                # Process markdown images in the cleaned content
                processed_content = self._process_images_for_pandoc(
                    process_markdown_images(cleaned_content)
                )
            
                # Extract document metadata
                date_str = get_document_date(doc)
                source = doc.metadata.get('source', 'Inconnu')
                author = doc.metadata.get('dl_meta', {}).get('origin', {}).get('author', 'Inconnu')
                headings = doc.metadata.get('dl_meta', {}).get('headings', [])
            
                # Format document section
                content += f"### Document {i+1}\n\n"
                content += f"**Score de similarité:** {dce_scores[i]:.4f}\n\n"
                content += f"**Source:** {source}\n\n"
                content += f"**Auteur:** {author}\n\n"
                content += f"**Origine du vecteur:** {dce_origins[i] if i < len(dce_origins) else 'Non spécifié'}\n\n"
            
                if headings and isinstance(headings, list):
                    content += f"**Titres:** {' > '.join(headings)}\n\n"
            
                content += f"**Date de modification:** {date_str}\n\n"
                content += "**Contenu:**\n\n"
                content += processed_content + "\n\n"
                content += "---\n\n"

        return content
    
    def _extract_unique_images(self, content: str) -> Set[str]:
        """Extract all unique image paths from markdown content."""
        images = set()
        pattern = r'!\[.*?\]\(([^)]+)\)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            # Convert to absolute path if relative
            if not os.path.isabs(match):
                if match.startswith('./'):
                    match = match[2:]
                abs_path = os.path.abspath(os.path.join(os.getcwd(), match))
                images.add(abs_path)
            else:
                images.add(match)
        
        return images
    
    def _create_images_catalog_section(self, images: Set[str]) -> str:
        """Create a catalog section listing all images used."""
        if not images:
            return ""
        
        content = "\n\n# Catalogue des Images\n\n"
        content += f"**Nombre total d'images:** {len(images)}\n\n"
        
        for i, image_path in enumerate(sorted(images), 1):
            # Get just the filename for display
            filename = os.path.basename(image_path)
            #content += f"## Image {i}: {filename}\n\n"
            
            # Check if image exists
            if os.path.exists(image_path):
                #content += f"**Chemin:** `{image_path}`\n\n"
                """
                # Try to get image size
                try:
                    import PIL.Image
                    with PIL.Image.open(image_path) as img:
                        width, height = img.size
                        content += f"**Dimensions:** {width} x {height} pixels\n\n"
                        content += f"**Format:** {img.format}\n\n"
                except:
                    pass
                """
                # Include the image in the document
                content += f"![{filename}]({image_path})\n\n"
            else:
                content += f"**Chemin:** `{image_path}` *(fichier introuvable)*\n\n"
            
            content += "---\n\n"
        return content
    
    def _get_reference_doc(self) -> str:
        """Get path to reference document template."""
        return "templates/reference.docx"
    
    def _has_reference_doc(self) -> bool:
        """Check if reference document exists."""
        return os.path.exists(self._get_reference_doc())
    
    def get_available_files(self) -> List[dict]:
        """Get list of available exported files."""
        if not self.output_dir.exists():
            return []
        
        files = []
        for file_path in self.output_dir.glob("*.docx"):
            stats = file_path.stat()
            files.append({
                "filename": file_path.name,
                "file_path": str(file_path),
                "size": stats.st_size,
                "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })
        
        return sorted(files, key=lambda x: x["created_at"], reverse=True)
    
    def delete_file(self, filename: str) -> bool:
        """Delete a specific exported file."""
        file_path = self.output_dir / filename
        if file_path.exists() and file_path.suffix == '.docx':
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")
                return False
        return False
    
    def cleanup_old_files(self, keep_latest: int = 10) -> int:
        """Clean up old exported files, keeping only the latest N files."""
        files = self.get_available_files()
        if len(files) <= keep_latest:
            return 0
        
        files_to_delete = files[keep_latest:]
        deleted_count = 0
        
        for file_info in files_to_delete:
            if self.delete_file(file_info["filename"]):
                deleted_count += 1
        
        return deleted_count


# Global exporter instance
_backend_docx_exporter = None

def get_backend_docx_exporter(output_dir: Optional[str] = None) -> BackendDocxExporter:
    """Get or create the global backend DOCX exporter instance."""
    global _backend_docx_exporter
    if _backend_docx_exporter is None:
        _backend_docx_exporter = BackendDocxExporter(output_dir)
    return _backend_docx_exporter