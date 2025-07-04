"""
Document processing utilities for the RAG Chatbot application.
"""
import re
from typing import List
from langchain_core.documents import Document as LCDocument
from config.config import Config


def process_markdown_images(markdown_text: str, images_folder: str = None) -> str:
    """
    Replace markdown image references with full paths including the images folder.
    
    Args:
        markdown_text (str): The markdown text containing image references
        images_folder (str): The folder where images are stored (defaults to images container)
        
    Returns:
        str: Markdown text with updated image paths
    """
    # Use the images container name as default
    if images_folder is None:
        images_folder = str(Config.IMAGES_FOLDER)
    
    # Regular expression to find markdown image syntax
    pattern = r'!\[(.*?)\]\(([^/].*?)\)'
    
    def replace_path(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        # Only modify the path if it doesn't already contain the images folder
        if not image_path.startswith(images_folder):
            full_path = f"{images_folder}/{image_path}"
            return f"![{alt_text}]({full_path})"
        return match.group(0)
    
    # Replace all image references
    processed_text = re.sub(pattern, replace_path, markdown_text)
    return processed_text


def get_document_date(doc: LCDocument) -> str:
    """
    Extract last_modified_date from document metadata
    
    Args:
        doc (LCDocument): Document to extract date from
        
    Returns:
        str: The date string or '0000-00-00' if not found
    """
    try:
        date_str = doc.metadata.get('dl_meta', {}).get('origin', {}).get('creation_date', '0000-00-00')
        # Extract just the date part if it's a longer timestamp
        if date_str and len(date_str) >= 10:
            date_str = date_str[:10]
        return date_str
    except:
        return '0000-00-00'


def format_docs(docs: List[LCDocument]) -> str:
    """
    Format documents with processed markdown images
    
    Args:
        docs (List[LCDocument]): Documents to format
        
    Returns:
        str: Formatted document content
    """
    processed_docs = []
    for doc in docs:
        # Process the markdown in the document content (using default images container)
        processed_content = process_markdown_images(doc.page_content)
        processed_docs.append(processed_content)
    
    return "\n\n".join([f"Document {i+1}:\n{content}" for i, content in enumerate(processed_docs)])


async def sort_documents(docs: List[LCDocument], scores: List[float], sort_by_date: bool = False):
    """
    Sort documents by relevance or date
    
    Args:
        docs (List[LCDocument]): Documents to sort
        scores (List[float]): Similarity scores
        sort_by_date (bool): Whether to sort by date (True) or relevance (False)
        
    Returns:
        tuple: Sorted documents and their corresponding scores
    """
    if sort_by_date:
        # Create pairs of (doc, score) and sort by date
        doc_score_pairs = list(zip(docs, scores))
        sorted_pairs = sorted(
            doc_score_pairs, 
            key=lambda pair: get_document_date(pair[0]), 
            reverse=True  # Most recent first
        )
        # Unpack the sorted pairs
        sorted_docs = [doc for doc, _ in sorted_pairs]
        sorted_scores = [score for _, score in sorted_pairs]
        return sorted_docs, sorted_scores
    else:
        # Already sorted by relevance from the retriever
        return docs, scores


def extract_sources(docs: List[LCDocument]) -> set:
    """
    Extract sources from documents.
    
    Args:
        docs (List[LCDocument]): Documents to extract sources from
        
    Returns:
        set: Set of (source, page) tuples
    """
    sources = set()
    for doc in docs:
        source_page_pair = (
            doc.metadata.get('source', 'Unknown'), 
            doc.metadata.get('page', 'N/A')
        )
        sources.add(source_page_pair)
    return sources