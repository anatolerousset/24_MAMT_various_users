import os
from pathlib import Path
from typing import Dict, Optional, List
from azure.storage.blob import BlobServiceClient
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # === Qdrant Configuration ===
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # === Azure OpenAI Configuration ===
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    # === Azure BLOB Configuration ===
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    # === Model Configuration ===
    FINETUNED_DENSE_EMBED_MODEL_ID = "finetuned_models/bge-m3_finetuned_0304_10am"
    FINETUNED_BGE_RERANKER_MODEL_ID = "finetuned_models/bge-reranker-v2-m3_finetuned_0404_10am/checkpoint-711"
    
    # === Collection Names ===
    COLLECTION_NAME = "region_paca_documents"  # Technical/Region collection
    DCE_COLLECTION = "dce_documents"            # DCE collection
    
    # === Processing Settings ===
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_BATCH_SIZE = 8
    IMAGE_RESOLUTION_SCALE = 2.0
    DENSE_EMBEDDING_DIM = 1024
    
    # === Duplicate Detection Settings ===
    SIMILARITY_THRESHOLD = 0.95
    DUPLICATE_BATCH_SIZE = 100
    
    # === Performance Settings ===
    ASYNCIO_SLEEP = 0
    RESEARCH_WAITING_TIME = 7
    
    # === Provider Selection ===
    LLM_PROVIDER = "azure"  # "ollama" or "azure"
    EMBEDDING_PROVIDER = "huggingface"
    OLLAMA_MODEL = "deepseek-r1:14b"
    
    # === Search Configuration ===
    DEFAULT_SIMILARITY_THRESHOLD = 0.1
    DEFAULT_MAX_RESULTS = 15
    DEFAULT_DISPLAY_LIMIT = 8
    
    # === File Extensions ===
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls"}

    # Container names
    IMAGES_CONTAINER = "images"
    MARKDOWN_CONTAINER = "markdown"
    TOC_CONTAINER = "toc" 
    CHUNKS_CONTAINER = "chunks"
    REPORTS_CONTAINER = "reports"
    INPUT_CONTAINER = "input-files"
    EXPORTS_CONTAINER = "exports"
    ARCHIVE_CONTAINER = "archive"  

    IMAGES_FOLDER = Path("./public")
    
    # Local temp directory for processing
    TEMP_DIR = Path("./temp")
    
    # Default archiving behavior - can be overridden per ingestion
    ARCHIVE_AFTER_PROCESSING = True  # Default for region files
    ARCHIVE_ADD_TIMESTAMP = False     # Add timestamp to archived files
    ARCHIVE_PRESERVE_STRUCTURE = False  # Flatten structure in archive
    
    # Data type specific archiving rules
    ARCHIVE_RULES = {
        'dce': False,     # DCE files should NOT be archived by default
        'region': True,   # Region files SHOULD be archived by default
        'technical': True # Technical files SHOULD be archived by default
    }

    @classmethod
    def get_blob_client(cls):
        """Get Azure Blob Service Client"""
        return BlobServiceClient.from_connection_string(cls.AZURE_STORAGE_CONNECTION_STRING)
    
    @classmethod
    def ensure_containers(cls):
        """Ensure all required containers exist"""
        blob_client = cls.get_blob_client()
        containers = [
            cls.IMAGES_CONTAINER,
            cls.MARKDOWN_CONTAINER, 
            cls.TOC_CONTAINER,
            cls.CHUNKS_CONTAINER,
            cls.REPORTS_CONTAINER,
            cls.INPUT_CONTAINER,
            cls.EXPORTS_CONTAINER,
            cls.ARCHIVE_CONTAINER 
        ]
        
        for container_name in containers:
            try:
                blob_client.create_container(container_name)
                print(f"Created container: {container_name}")
            except Exception as e:
                if "ContainerAlreadyExists" not in str(e):
                    print(f"Error creating container {container_name}: {e}")
    
    @classmethod
    def get_temp_paths(cls, data_type: str, identifier: str = None):
        """
        Get temporary local paths for processing - UPDATED to support both regions and offices
        
        Args:
            data_type: Type of data ('dce', 'region', 'technical')
            identifier: Region name for 'region' data type, office name for 'dce' data type
        
        Returns:
            Dict of temporary paths
        """
        if data_type == 'region' and identifier:
            base_dir = cls.TEMP_DIR / f"region_{identifier}"
        elif data_type == 'dce' and identifier:
            base_dir = cls.TEMP_DIR / f"dce_{identifier}"  # NEW: Support for DCE offices
        else:
            base_dir = cls.TEMP_DIR / data_type
            
        paths = {
            "base": base_dir,
            "images": base_dir / "images",
            "markdown": base_dir / "markdown", 
            "toc": base_dir / "toc",
            "chunks": base_dir / "chunks"
        }
        
        # Create temp directories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        return paths
    
    @classmethod
    def should_archive_data_type(cls, data_type: str) -> bool:
        """
        Determine if files of a given data type should be archived by default.
        
        Args:
            data_type: The type of data ('dce', 'region', 'technical', etc.)
            
        Returns:
            bool: True if files should be archived, False otherwise
        """
        return cls.ARCHIVE_RULES.get(data_type.lower(), cls.ARCHIVE_AFTER_PROCESSING)
    
    @classmethod
    def get_collection_name_for_data_type(cls, data_type: str, region_name: str = None, office_name: str = None) -> str:
        """
        Get the appropriate collection name for a data type - UPDATED with enhanced office support.
        
        Args:
            data_type: The type of data ('dce', 'region', 'technical')
            region_name: Optional region name for regional data
            office_name: Optional office name for DCE data
            
        Returns:
            str: Collection name
        """
        if data_type.lower() == 'dce':
            if office_name:
                return f"dce_{office_name.lower()}_documents"
            return cls.DCE_COLLECTION
        elif data_type.lower() == 'region' and region_name:
            return f"region_{region_name.lower()}_documents"
        elif data_type.lower() == 'technical':
            return cls.COLLECTION_NAME
        else:
            return f"{data_type.lower()}_documents"
    
    @classmethod
    def load_available_regions(cls) -> List[str]:
        """Load available regions from environment variables"""
        try:
            regions_str = os.getenv('AVAILABLE_REGIONS', '["PACA"]')
            if regions_str:
                try:
                    regions = json.loads(regions_str)
                    if isinstance(regions, list):
                        return [str(region).strip() for region in regions if region]
                except json.JSONDecodeError:
                    # Fallback: manual parsing for malformed JSON
                    regions_str = regions_str.strip('[]"\'')
                    regions = [region.strip().strip('"\'') for region in regions_str.split(',')]
                    return [region for region in regions if region]
           
            return ["PACA"]  # Default fallback
           
        except Exception as e:
            print(f"Warning: Error loading regions: {e}. Using default values.")
            return ["PACA"]
    
    @classmethod
    def load_available_offices(cls) -> List[str]:
        """Load available offices from environment variables"""
        try:
            offices_str = os.getenv('AVAILABLE_OFFICES', '["CSP"]')
            if offices_str:
                try:
                    offices = json.loads(offices_str)
                    if isinstance(offices, list):
                        return [str(office).strip() for office in offices if office]
                except json.JSONDecodeError:
                    # Fallback: manual parsing for malformed JSON
                    offices_str = offices_str.strip('[]"\'')
                    offices = [office.strip().strip('"\'') for office in offices_str.split(',')]
                    return [office for office in offices if office]
           
            return ["CSP"]  # Default fallback
           
        except Exception as e:
            print(f"Warning: Error loading offices: {e}. Using default values.")
            return ["CSP"]
    
    # === Validation Methods ===
    @classmethod
    def get_active_llm_provider(cls) -> str:
        """Get the active LLM provider name for display."""
        if cls.LLM_PROVIDER.lower() == "azure" and cls.validate_azure_config():
            return "Azure OpenAI GPT-4o"
        return f"Ollama {cls.OLLAMA_MODEL}"

    @classmethod
    def validate_azure_config(cls) -> bool:
        """Validate Azure OpenAI configuration."""
        return all([
            cls.AZURE_OPENAI_API_KEY,
            cls.AZURE_OPENAI_ENDPOINT,
            cls.AZURE_OPENAI_DEPLOYMENT_NAME
        ])
    
    @classmethod
    def validate_blob_config(cls) -> bool:
        """Validate Azure Blob Storage configuration."""
        return bool(cls.AZURE_STORAGE_CONNECTION_STRING)
    
    @classmethod
    def get_config_summary(cls) -> Dict[str, any]:
        """Get a summary of current configuration for debugging."""
        return {
            "qdrant_url": cls.QDRANT_URL,
            "collections": {
                "technical": cls.COLLECTION_NAME,
                "dce": cls.DCE_COLLECTION
            },
            "available_regions": cls.load_available_regions(),
            "available_offices": cls.load_available_offices(),
            "archiving": {
                "default_behavior": cls.ARCHIVE_AFTER_PROCESSING,
                "add_timestamp": cls.ARCHIVE_ADD_TIMESTAMP,
                "preserve_structure": cls.ARCHIVE_PRESERVE_STRUCTURE,
                "archive_container": cls.ARCHIVE_CONTAINER,
                "rules": cls.ARCHIVE_RULES
            },
            "containers": {
                "input": cls.INPUT_CONTAINER,
                "archive": cls.ARCHIVE_CONTAINER,
                "images": cls.IMAGES_CONTAINER,
                "exports": cls.EXPORTS_CONTAINER
            },
            "llm": {
                "provider": cls.LLM_PROVIDER,
                "azure_configured": cls.validate_azure_config(),
                "active_provider": cls.get_active_llm_provider()
            },
            "blob_storage": {
                "configured": cls.validate_blob_config()
            }
        }