"""
LLM initialization and management for the RAG Chatbot application.
"""
import gc
import torch
from langchain_ollama import OllamaLLM
from langchain_openai import AzureChatOpenAI

from config.config import Config

class LLMManager:
    """Manages LLM initialization and configuration."""
    
    def __init__(self):
        self.llm = None
        self.provider = Config.LLM_PROVIDER.lower()
    
    def initialize_llm(self):
        """
        Initialize the appropriate LLM based on the selected provider.
        
        Returns:
            LLM: An instance of OllamaLLM or AzureChatOpenAI
        """
        # Clean GPU memory before initializing
        torch.cuda.empty_cache()
        gc.collect()
        
        if self.provider == "azure":
            return self._initialize_azure_llm()
        else:
            return self._initialize_ollama_llm()
    
    def _initialize_azure_llm(self):
        """Initialize Azure OpenAI LLM."""
        if not Config.validate_azure_config():
            print("WARNING: Azure OpenAI credentials are missing or incomplete. Falling back to Ollama.")
            return self._initialize_ollama_llm()
        
        try:
            return AzureChatOpenAI(
                azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
                openai_api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
                api_key=Config.AZURE_OPENAI_API_KEY,
                temperature=0.7,
                streaming=True
            )
        except Exception as e:
            print(f"Error initializing Azure OpenAI: {e}")
            print("Falling back to Ollama.")
            return self._initialize_ollama_llm()
    
    def _initialize_ollama_llm(self):
        """Initialize Ollama LLM."""
        return OllamaLLM(model=Config.OLLAMA_MODEL)
    
    def get_provider_name(self) -> str:
        """Get the active LLM provider name for display."""
        if self.provider == "azure" and Config.validate_azure_config():
            return "Azure OpenAI GPT-4o"
        return f"Ollama {Config.OLLAMA_MODEL}"


def initialize_llm():
    """
    Factory function to initialize LLM.
    Maintains backward compatibility with the original code.
    """
    manager = LLMManager()
    return manager.initialize_llm()