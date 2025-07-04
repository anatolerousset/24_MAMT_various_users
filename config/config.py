# config/config.py (Updated sections)
import os
from pathlib import Path
from typing import Dict, Optional
from azure.storage.blob import BlobServiceClient

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
    COLLECTION_NAME = "region_paca_documents"  # Technical/Region collection (SHARED)
    # DCE collections are now per-user: dce_user_{session_id}
    REGION_COLLECTION_BASE_NAME = "region_"
    
    # === User Management ===
    USER_SESSION_TIMEOUT = int(os.getenv("USER_SESSION_TIMEOUT", "3600"))  # 1 hour
    USER_CLEANUP_INTERVAL = int(os.getenv("USER_CLEANUP_INTERVAL", "300"))  # 5 minutes
    
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

    # Base folders - will be user-specific
    IMAGES_FOLDER = Path("./public")
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
    def get_user_temp_paths(cls, user_session_id: str, data_type: str, region_name: str = None):
        """Get temporary local paths for processing per user"""
        # Create user-specific temp directory
        if data_type == 'region' and region_name:
            base_dir = cls.TEMP_DIR / f"user_{user_session_id}" / f"region_{region_name}"
        else:
            base_dir = cls.TEMP_DIR / f"user_{user_session_id}" / data_type
            
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
    def get_user_public_folder(cls, user_session_id: str) -> Path:
        """Get user-specific public folder for images"""
        public_folder = cls.IMAGES_FOLDER / f"user_{user_session_id}"
        public_folder.mkdir(parents=True, exist_ok=True)
        return public_folder
    
    @classmethod
    def get_user_dce_collection_name(cls, user_session_id: str) -> str:
        """Get user-specific DCE collection name"""
        return f"dce_user_{user_session_id}"
    
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
    def get_collection_name_for_data_type(cls, data_type: str, region_name: str = None, user_session_id: str = None) -> str:
        """
        Get the appropriate collection name for a data type.
        
        Args:
            data_type: The type of data ('dce', 'region', 'technical')
            region_name: Optional region name for regional data
            user_session_id: User session ID for DCE collections
            
        Returns:
            str: Collection name
        """
        if data_type.lower() == 'dce':
            if user_session_id:
                return cls.get_user_dce_collection_name(user_session_id)
            else:
                # Fallback to generic DCE collection
                return "dce_documents"
        elif data_type.lower() == 'region' and region_name:
            return f"region_{region_name.lower()}_documents"
        elif data_type.lower() == 'technical':
            return cls.COLLECTION_NAME
        else:
            return f"{data_type.lower()}_documents"
    
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
                "dce_pattern": "dce_user_{session_id}",
                "user_specific": True
            },
            "user_management": {
                "session_timeout": cls.USER_SESSION_TIMEOUT,
                "cleanup_interval": cls.USER_CLEANUP_INTERVAL
            },
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