"""
Backend-compatible response generation functionality for the RAG system.
Removes Chainlit dependencies and provides pure API-based response generation.
"""
import gc
import torch
from typing import List, Set, Optional, Tuple, Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, HumanMessage, SystemMessage
from langchain_core.documents import Document as LCDocument

from core.templates import SYSTEM_RESPONSE_TEMPLATE, STATUS_MESSAGES
from utils.document_processing import format_docs, extract_sources
from config.config import Config


class BackendResponseGenerator:
    """Manages LLM response generation for backend API without Chainlit dependencies."""
    
    def __init__(self, llm):
        self.llm = llm
        # Create system prompt template
        self.system_prompt = ChatPromptTemplate.from_template(SYSTEM_RESPONSE_TEMPLATE)
        # Create user specifications template
        self.user_template = """Spécifications supplémentaires:
{user_specifications}

Veuillez tenir compte de ces spécifications lors de la génération de votre réponse technique."""
    
    async def generate_response_with_specifications(
        self, 
        query: str, 
        technical_docs: List[LCDocument], 
        dce_docs: List[LCDocument], 
        user_specifications: Optional[str] = None,
        use_both: bool = True,
        use_technical: bool = False
    ) -> Tuple[str, Set, Dict[str, Any]]:
        """
        Generate LLM response with optional user specifications.
        
        Args:
            query (str): User query
            technical_docs (List[LCDocument]): Technical documents
            dce_docs (List[LCDocument]): DCE documents
            user_specifications (str, optional): Additional user specifications
            use_both (bool): Whether to use both collections
            use_technical (bool): Whether to use only technical collection
            
        Returns:
            Tuple[str, Set, Dict]: (response_text, sources, metadata)
        """
        # Ensure GPU memory is clean before generating response
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Check if any documents are available
        if not technical_docs and not dce_docs:
            return "Aucun document disponible pour générer une réponse.", set(), {
                "status": "no_documents",
                "collections_used": []
            }
        
        # Prepare contexts and metadata
        technical_context, dce_context, collections_used = self._prepare_contexts(
            technical_docs, dce_docs, use_both, use_technical
        )
        
        # Create system message with context
        system_message_content = self.system_prompt.format(
            technical_context=technical_context,
            dce_context=dce_context,
            question=query
        )
        
        # Create messages list
        messages = [SystemMessage(content=system_message_content)]
        
        # Add user specifications if provided
        if user_specifications and user_specifications.strip():
            user_message_content = self.user_template.format(
                user_specifications=user_specifications.strip()
            )
            messages.append(HumanMessage(content=user_message_content))
        else:
            # Add a simple user message to trigger the response
            messages.append(HumanMessage(content="Veuillez générer la réponse technique."))
        
        # Generate response
        try:
            # Use async invoke for proper response generation
            response = await self.llm.ainvoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Extract sources
            sources = self._extract_response_sources(
                technical_docs, dce_docs, use_both, use_technical
            )
            
            # Prepare metadata
            metadata = {
                "status": "success",
                "collections_used": collections_used,
                "specifications_used": bool(user_specifications and user_specifications.strip()),
                "technical_docs_count": len(technical_docs) if technical_docs else 0,
                "dce_docs_count": len(dce_docs) if dce_docs else 0,
                "total_sources": len(sources)
            }
            
            return response_text, sources, metadata
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
            return error_msg, set(), {
                "status": "error",
                "error": str(e),
                "collections_used": collections_used
            }
    
    def _prepare_contexts(
        self, 
        technical_docs: List[LCDocument], 
        dce_docs: List[LCDocument], 
        use_both: bool, 
        use_technical: bool
    ) -> Tuple[str, str, List[str]]:
        """Prepare technical and DCE contexts based on selection."""
        collections_used = []
        
        # Determine which documents to use
        if use_both:
            # Use both collections
            technical_context = format_docs(technical_docs) if technical_docs else "N/A"
            dce_context = format_docs(dce_docs) if dce_docs else "N/A"
            
            if technical_docs:
                collections_used.append(Config.COLLECTION_NAME)
            if dce_docs:
                collections_used.append(Config.DCE_COLLECTION)
                
        elif use_technical:
            # Use only technical collection
            technical_context = format_docs(technical_docs) if technical_docs else "N/A"
            dce_context = "N/A"
            
            if technical_docs:
                collections_used.append(Config.COLLECTION_NAME)
                
        else:
            # Use only DCE collection
            technical_context = "N/A"
            dce_context = format_docs(dce_docs) if dce_docs else "N/A"
            
            if dce_docs:
                collections_used.append(Config.DCE_COLLECTION)
        
        return technical_context, dce_context, collections_used
    
    def _extract_response_sources(
        self, 
        technical_docs: List[LCDocument], 
        dce_docs: List[LCDocument], 
        use_both: bool, 
        use_technical: bool
    ) -> Set:
        """Extract sources from the documents used in response generation."""
        sources = set()
        
        # Only include sources from collections that were used
        if use_both:
            if technical_docs:
                sources.update(extract_sources(technical_docs))
            if dce_docs:
                sources.update(extract_sources(dce_docs))
        elif use_technical:
            if technical_docs:
                sources.update(extract_sources(technical_docs))
        else:
            if dce_docs:
                sources.update(extract_sources(dce_docs))
        
        return sources


# Streaming response generator for real-time responses
class StreamingResponseGenerator(BackendResponseGenerator):
    """Extended generator that supports streaming responses."""
    
    async def generate_streaming_response(
        self, 
        query: str, 
        technical_docs: List[LCDocument], 
        dce_docs: List[LCDocument], 
        user_specifications: Optional[str] = None,
        use_both: bool = True,
        use_technical: bool = False
    ):
        """
        Generate streaming LLM response.
        
        Yields response chunks as they're generated.
        """
        # Ensure GPU memory is clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Check if any documents are available
        if not technical_docs and not dce_docs:
            yield {
                "type": "error",
                "content": "Aucun document disponible pour générer une réponse.",
                "sources": [],
                "metadata": {"status": "no_documents"}
            }
            return
        
        # Prepare contexts
        technical_context, dce_context, collections_used = self._prepare_contexts(
            technical_docs, dce_docs, use_both, use_technical
        )
        
        # Create system message
        system_message_content = self.system_prompt.format(
            technical_context=technical_context,
            dce_context=dce_context,
            question=query
        )
        
        # Create messages list
        messages = [SystemMessage(content=system_message_content)]
        
        # Add user specifications if provided
        if user_specifications and user_specifications.strip():
            user_message_content = self.user_template.format(
                user_specifications=user_specifications.strip()
            )
            messages.append(HumanMessage(content=user_message_content))
        else:
            messages.append(HumanMessage(content="Veuillez générer la réponse technique."))
        
        # Generate streaming response
        try:
            accumulated_response = ""
            
            # Stream the response
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    accumulated_response += chunk.content
                    yield {
                        "type": "chunk",
                        "content": chunk.content,
                        "accumulated": accumulated_response
                    }
            
            # Extract sources after generation is complete
            sources = self._extract_response_sources(
                technical_docs, dce_docs, use_both, use_technical
            )
            
            # Final metadata
            metadata = {
                "status": "success",
                "collections_used": collections_used,
                "specifications_used": bool(user_specifications and user_specifications.strip()),
                "technical_docs_count": len(technical_docs) if technical_docs else 0,
                "dce_docs_count": len(dce_docs) if dce_docs else 0,
                "total_sources": len(sources)
            }
            
            # Send final completion message
            yield {
                "type": "complete",
                "content": accumulated_response,
                "sources": list(sources),
                "metadata": metadata
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "content": f"Erreur lors de la génération: {str(e)}",
                "sources": [],
                "metadata": {"status": "error", "error": str(e)}
            }


# Factory functions for backward compatibility and API use
async def generate_llm_response(
    query: str, 
    technical_docs: List[LCDocument], 
    dce_docs: List[LCDocument], 
    user_specifications: Optional[str] = None,
    use_both: bool = True,
    use_technical: bool = False,
    llm = None
) -> Tuple[str, Set, Dict[str, Any]]:
    """
    Factory function to generate response with user specifications for backend API.
    """
    if llm is None:
        raise ValueError("LLM instance is required for backend response generation")
    
    try:
        generator = BackendResponseGenerator(llm)
        
        print(f"DEBUG: Calling generator with query: {query[:100]}...")
        print(f"DEBUG: Technical docs: {len(technical_docs)}, DCE docs: {len(dce_docs)}")
        print(f"DEBUG: User specifications: {user_specifications is not None}")
        
        response_text, sources, metadata = await generator.generate_response_with_specifications(
            query, technical_docs, dce_docs, user_specifications, use_both, use_technical
        )
        
        print(f"DEBUG: Generator returned - Response length: {len(response_text)}, Sources: {type(sources)}, {len(sources) if sources else 0}")
        
        # Ensure sources is a set of tuples
        if not isinstance(sources, set):
            print(f"WARNING: Sources is not a set, converting from {type(sources)}")
            if isinstance(sources, (list, tuple)):
                sources = set(sources)
            else:
                sources = set()
        
        # Validate each source in the set
        validated_sources = set()
        for source in sources:
            try:
                if isinstance(source, tuple) and len(source) >= 2:
                    validated_sources.add((str(source[0]), str(source[1])))
                elif isinstance(source, (list, tuple)) and len(source) >= 2:
                    validated_sources.add((str(source[0]), str(source[1])))
                elif isinstance(source, str):
                    validated_sources.add((source, "N/A"))
                else:
                    print(f"WARNING: Skipping invalid source: {source}")
            except Exception as e:
                print(f"WARNING: Error processing source {source}: {e}")
                continue
        
        print(f"DEBUG: Validated sources count: {len(validated_sources)}")
        
        return response_text, validated_sources, metadata
        
    except Exception as e:
        print(f"ERROR: In generate_llm_response: {e}")
        import traceback
        traceback.print_exc()
        # Return safe defaults instead of raising
        return f"Error generating response: {str(e)}", set(), {"error": str(e)}

async def generate_streaming_llm_response(
    query: str, 
    technical_docs: List[LCDocument], 
    dce_docs: List[LCDocument], 
    user_specifications: Optional[str] = None,
    use_both: bool = True,
    use_technical: bool = False,
    llm = None
):
    """
    Factory function to generate streaming response for backend API.
    
    Yields response chunks as they're generated.
    """
    if llm is None:
        raise ValueError("LLM instance is required for backend response generation")
    
    generator = StreamingResponseGenerator(llm)
    async for chunk in generator.generate_streaming_response(
        query, technical_docs, dce_docs, user_specifications, use_both, use_technical
    ):
        yield chunk


# Utility function for Chainlit compatibility
def create_chainlit_compatible_generator(llm):
    """
    Create a response generator that's compatible with Chainlit sessions.
    
    This function can be used in environments where Chainlit is available.
    """
    try:
        import chainlit as cl
        
        class ChainlitResponseGenerator(BackendResponseGenerator):
            """Chainlit-compatible response generator."""
            
            async def generate_response_with_specifications(
                self, 
                query: str, 
                technical_docs: List[LCDocument], 
                dce_docs: List[LCDocument], 
                user_specifications: Optional[str] = None,
                use_both: bool = True,
                use_technical: bool = False
            ) -> Set:
                """Generate response with Chainlit message streaming."""
                
                # Get current action from user session if available
                action = cl.user_session.get("current_action") if hasattr(cl, 'user_session') else None
                
                # Ensure GPU memory is clean
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Check documents
                if not technical_docs and not dce_docs:
                    await cl.Message(content="Aucun document disponible pour générer une réponse.").send()
                    return set()
                
                # Prepare contexts
                technical_context, dce_context, collections_used = self._prepare_contexts(
                    technical_docs, dce_docs, use_both, use_technical
                )
                
                # Display generation info
                collections_text = ", ".join(collections_used) if collections_used else "Aucune"
                await cl.Message(content=f"**Génération basée sur:** {collections_text}").send()
                
                # Create messages
                system_message_content = self.system_prompt.format(
                    technical_context=technical_context,
                    dce_context=dce_context,
                    question=query
                )
                
                messages = [SystemMessage(content=system_message_content)]
                
                if user_specifications and user_specifications.strip():
                    user_message_content = self.user_template.format(
                        user_specifications=user_specifications.strip()
                    )
                    messages.append(HumanMessage(content=user_message_content))
                else:
                    messages.append(HumanMessage(content="Veuillez générer la réponse technique."))
                
                # Stream response
                response_msg = cl.Message(content="")
                
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, 'content'):
                        await response_msg.stream_token(chunk.content)
                    else:
                        await response_msg.stream_token(str(chunk))
                
                await response_msg.send()
                
                # Extract sources
                return self._extract_response_sources(
                    technical_docs, dce_docs, use_both, use_technical
                )
        
        return ChainlitResponseGenerator(llm)
    
    except ImportError:
        # Chainlit not available, return backend generator
        return BackendResponseGenerator(llm)


# Response validation utilities
def validate_response_quality(response_text: str, sources: Set) -> Dict[str, Any]:
    """
    Validate the quality of generated response.
    
    Args:
        response_text: Generated response
        sources: Sources used
        
    Returns:
        Dict with validation results
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "metrics": {}
    }
    
    # Check response length
    if len(response_text) < 100:
        validation["warnings"].append("Response is very short")
    elif len(response_text) > 5000:
        validation["warnings"].append("Response is very long")
    
    # Check if sources are used
    if not sources:
        validation["warnings"].append("No sources identified")
    
    # Check for code formatting
    if "```" in response_text:
        validation["metrics"]["has_code_blocks"] = True
    
    # Check for French content
    french_indicators = ["le", "la", "les", "de", "du", "des", "et", "ou", "dans"]
    french_count = sum(1 for word in french_indicators if word in response_text.lower())
    validation["metrics"]["french_indicators"] = french_count
    
    if french_count < 3:
        validation["warnings"].append("Response may not be in French")
    
    # Overall validation
    if len(validation["warnings"]) > 2:
        validation["is_valid"] = False
    
    return validation