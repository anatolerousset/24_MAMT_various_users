"""
Chainlit frontend that communicates with the backend API with user session support
"""
import os
import httpx
import chainlit as cl
from typing import Dict, Any, List, Set, Tuple, Optional
from datetime import datetime
import json
import re
import asyncio
from config.config import Config

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Backend URL from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")
IMAGES_URL = os.getenv("IMAGES_URL", "http://localhost:8001")

class UserSessionManager:
    def __init__(self, user_session_id: str = None):
        self.user_session_id = user_session_id
        self.response_history = []
        self.last_search_results = {}
        self.current_query = ""
        self.user_specifications = None
        self.user_info = None
    
    def add_response(self, content: str, query: str, sources: List[Tuple[str, int]], specifications: str = None):
        """Add response to history"""
        entry = {
            "content": content,
            "query": query,
            "sources": sources,
            "timestamp": datetime.now().strftime("%d/%m/%Y à %H:%M"),
            "specifications": specifications,
            "user_session_id": self.user_session_id
        }
        self.response_history.append(entry)
    
    def get_history(self):
        """Get response history"""
        return self.response_history
    
    def clear_history(self):
        """Clear response history"""
        self.response_history = []

async def initialize_user_session():
    """Initialize or get user session from backend"""
    try:
        # Try to get existing session from cl.user_session
        stored_session_id = cl.user_session.get("user_session_id")
        
        if stored_session_id:
            # Verify session is still valid
            async with httpx.AsyncClient() as client:
                headers = {"X-User-Session": stored_session_id}
                response = await client.get(f"{BACKEND_URL}/api/user/info", headers=headers, timeout=10.0)
                
                if response.status_code == 200:
                    user_info = response.json()
                    _log.info(f"Using existing user session: {stored_session_id}")
                    return stored_session_id, user_info
        
        # Create new session
        async with httpx.AsyncClient() as client:
            # You could pass user identifier here if available
            response = await client.post(
                f"{BACKEND_URL}/api/user/session", 
                json={"user_identifier": None},  # Could be email, username, etc.
                timeout=10.0
            )
            
            if response.status_code == 200:
                session_data = response.json()
                session_id = session_data["session_id"]
                
                # Store in cl.user_session
                cl.user_session.set("user_session_id", session_id)
                
                _log.info(f"Created new user session: {session_id}")
                return session_id, session_data
            else:
                _log.error(f"Failed to create user session: {response.status_code}")
                return None, None
                
    except Exception as e:
        _log.error(f"Error initializing user session: {e}")
        return None, None

async def update_progress_bar(status_msg, duration_seconds=9):
    """Update progress bar over specified duration"""
    steps = 50  # Number of progress steps
    step_duration = duration_seconds / steps
    
    for i in range(steps + 1):
        progress = i / steps
        filled = int(progress * 30)  # 30 characters wide progress bar
        bar = "█" * filled + "░" * (30 - filled)
        percentage = int(progress * 100)
        
        # Different messages for different progress stages
        if percentage < 1:
            message = "🔍 **Recherche hybride en cours...** Initialisation"
        elif percentage < 5:
            message = "🔍 **Recherche hybride en cours...** Recherche dans les mémoires techniques"
        elif percentage < 50:
            message = "🔍 **Recherche hybride en cours...** Reclassement des mémoires techniques"
        elif percentage < 55:
            message = "🔍 **Recherche hybride en cours...** Recherche dans le DCE"
        elif percentage < 99:
            message = "🔍 **Recherche hybride en cours...** Reclassement du DCE"
        else:
            message = "🔍 **Recherche hybride en cours...** Finalisation"
        
        status_msg.content = f"{message}\n\n[{bar}] {percentage}%"
        await status_msg.update()
        
        if i < steps:  # Don't sleep after the last update
            await asyncio.sleep(step_duration)

def process_images_for_chainlit(content: str, user_session_id: str = None) -> str:
    """
    Process image URLs in content to ensure they work with Chainlit.
    Convert backend image URLs to proper format for Chainlit display with user support.
    
    Args:
        content: Content with image references
        user_session_id: User session ID for user-specific images
        
    Returns:
        Content with Chainlit-compatible image references
    """
    # Pattern to match markdown images
    image_pattern = re.compile(r'!\[(.*?)\]\(([^)]+)\)')
    
    def replace_image_url(match):
        alt_text = match.group(1)
        image_url = match.group(2)

        # Extract filename from path
        if '/' in image_url:
            filename = image_url.split('/')[-1]
            # Check if it's already a user-specific path
            if user_session_id and f"user_{user_session_id}" in image_url:
                # Already user-specific, use as is but convert to HTTP URL
                new_path = f"{IMAGES_URL}/images/user_{user_session_id}/{filename}"
            elif user_session_id:
                # Not user-specific, make it user-specific
                new_path = f"{IMAGES_URL}/images/user_{user_session_id}/{filename}"
            else:
                # No user session, use general path
                new_path = f"{IMAGES_URL}/images/{filename}"
        else:
            filename = image_url
            if user_session_id:
                new_path = f"{IMAGES_URL}/images/user_{user_session_id}/{filename}"
            else:
                new_path = f"{IMAGES_URL}/images/{filename}"

        if user_session_id:
            _log.info(f"Image URL processing for user {user_session_id}: {image_url} -> {new_path}")
        else:
            _log.info(f"Image URL processing: {image_url} -> {new_path}")
        return f"![{alt_text}]({new_path})"

    processed_content = image_pattern.sub(replace_image_url, content)
    return processed_content

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with user support"""
    # Initialize user session
    user_session_id, user_info = await initialize_user_session()
    
    if not user_session_id:
        await cl.Message(content="❌ **Erreur d'initialisation de session utilisateur**\n\nImpossible de créer une session utilisateur.").send()
        return
    
    # Initialize session manager with user session
    session_manager = UserSessionManager(user_session_id)
    session_manager.user_info = user_info
    cl.user_session.set("session_manager", session_manager)
    
    try:
        # Check backend health and get configuration
        async with httpx.AsyncClient() as client:
            headers = {"X-User-Session": user_session_id}
            
            health_response = await client.get(f"{BACKEND_URL}/health", timeout=10.0)
            config_response = await client.get(f"{BACKEND_URL}/api/config", timeout=10.0)
            
            if health_response.status_code == 200 and config_response.status_code == 200:
                health_data = health_response.json()
                config_data = config_response.json()
                
                status = health_data.get("status", "unknown")
                services = health_data.get("services", {})
                
                if status == "healthy":
                    welcome_msg = "🟢 **Mon assistant mémoire technique opérationnel**\n\n"
                    
                    # Display user session info
                    if user_info:
                        welcome_msg += f"**Session utilisateur:** `{user_session_id}`\n"
                        welcome_msg += f"**Collection DCE personnelle:** `{user_info.get('dce_collection', 'N/A')}`\n\n"
                    
                    # Display service status
                    service_status = []
                    if services.get("qdrant") == "healthy":
                        service_status.append("✓ Base vectorielle")
                    if services.get("llm") == "healthy":
                        service_status.append("✓ Modèle de langage")
                    if services.get("blob_storage") == "healthy":
                        service_status.append("✓ Stockage cloud")
                    
                    welcome_msg += "**Services actifs:** " + ", ".join(service_status) + "\n\n"
                    
                    # Display configuration
                    collections = config_data.get("collections", {})
                    llm_info = config_data.get("llm", {})
                    
                    welcome_msg += f"**Collections disponibles:**\n"
                    welcome_msg += f"- Technique (partagée): `{collections.get('technical', 'N/A')}`\n"
                    welcome_msg += f"- DCE (personnelle): `{user_info.get('dce_collection') if user_info else 'N/A'}`\n\n"
                    welcome_msg += f"**LLM actif:** {llm_info.get('provider', 'N/A')}\n\n"
                    
                else:
                    welcome_msg = "🟡 **Système en cours d'initialisation**\n\nLe système démarre, veuillez patienter quelques instants."
            else:
                welcome_msg = "🔴 **Système non disponible**\n\nImpossible de se connecter au backend."
    except Exception as e:
        welcome_msg = f"🔴 **Erreur de connexion**\n\nErreur: {str(e)}"
    
    # Add user session cleanup option
    welcome_msg += "\n\n💡 **Note:** Votre session est personnelle et sera automatiquement nettoyée après inactivité."
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {"X-User-Session": user_session_id}
            response = await client.get(f"{BACKEND_URL}/api/collections", headers=headers, timeout=10.0)
            if response.status_code == 200:
                collections_data = response.json()
                collections = collections_data.get("collections", [])
                if collections:
                    _log.info(f"Collections vectorielles disponibles: {', '.join(collections)}")
            else:
                _log.info(f"⚠️ Impossible de récupérer les collections.")
    except Exception as e:
        _log.info(f"⚠️ Erreur lors de la récupération des collections: {str(e)}")

    _log.info(f"Backend URL: {BACKEND_URL}")
    _log.info(f"User Session ID: {user_session_id}")

    current_host = os.getenv("CHAINLIT_HOST", "localhost")
    streamlit_frontend_url = f"http://{current_host}:8501"
    welcome_msg += f"\n\n📁 **Interface d'ingestion Streamlit** {streamlit_frontend_url}"

    welcome_msg += "\n\n**Écrivez le titre du paragraphe à générer pour commencer !**"
    
    await cl.Message(content=welcome_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with dual search and progress bar with user support"""
    query = message.content
    session_manager = cl.user_session.get("session_manager")
    
    if not session_manager or not session_manager.user_session_id:
        await cl.Message(content="❌ **Erreur de session**\n\nSession utilisateur invalide.").send()
        return
    
    session_manager.current_query = query
    user_session_id = session_manager.user_session_id
    
    # Show initial status message with progress bar
    status_msg = await cl.Message(content="🔍 **Recherche hybride en cours...** Initialisation\n\n[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%").send()
    
    # Initialize progress_task to None
    progress_task = None
    
    try:
        # Start progress bar task
        progress_task = asyncio.create_task(update_progress_bar(status_msg, Config.RESEARCH_WAITING_TIME)) 
        
        # Perform dual search (both collections) with user session
        async with httpx.AsyncClient() as client:
            search_payload = {
                "query": query,
                "threshold": 0.1,  # Using backend default
                "max_results": 15,
                "use_reranker": True
            }
            
            headers = {"X-User-Session": user_session_id}
            
            # Start the search request
            search_task = asyncio.create_task(
                client.post(
                    f"{BACKEND_URL}/api/dual-search",
                    json=search_payload,
                    headers=headers,
                    timeout=60.0  # Increased timeout for image processing
                )
            )
            
            # Wait for search to complete, but don't wait for progress bar
            done, pending = await asyncio.wait(
                [progress_task, search_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Get the search response
            response = await search_task
            
            # Cancel the progress bar if search completed first
            if progress_task and not progress_task.done():
                progress_task.cancel()
            
            if response.status_code == 200:
                search_results = response.json()
                
                technical_results = search_results.get("technical_results", {})
                dce_results = search_results.get("dce_results", {})
                total_documents = search_results.get("total_documents", 0)
                
                # Store results in session
                session_manager.last_search_results = {
                    "technical_results": technical_results,
                    "dce_results": dce_results,
                    "query": query
                }
                
                # Update final status
                status_msg.content = f"✅ **Recherche terminée** - {total_documents} documents trouvés "
                await status_msg.update()
                
                # Display results with user-specific image processing
                await display_search_results(technical_results, dce_results, user_session_id)
                
                if total_documents == 0:
                    await cl.Message(content="❌ **Aucun document trouvé**\n\nVeuillez reformuler votre titre de paragraphe.").send()
                    return
                
                # Automatically ask for specifications
                await ask_for_specifications()
                
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get("detail", f"Erreur HTTP {response.status_code}")
                status_msg.content = f"❌ **Erreur de recherche:** {error_msg}"
                await status_msg.update()
        
    except httpx.TimeoutException:
        # Cancel progress task if still running
        if progress_task and not progress_task.done():
            progress_task.cancel()
        status_msg.content = "⏱️ **Timeout** - La recherche a pris trop de temps. Réessayez."
        await status_msg.update()
    except Exception as e:
        # Cancel progress task if still running
        if progress_task and not progress_task.done():
            progress_task.cancel()
        status_msg.content = f"❌ **Erreur:** {str(e)}"
        await status_msg.update()

async def display_search_results(technical_results: Dict, dce_results: Dict, user_session_id: str):
    """Display search results from both collections with user-specific image handling"""
    
    # Display technical collection results
    if technical_results.get("documents"):
        collection_name = technical_results.get("collection_name", "Collection Technique")
        docs = technical_results["documents"]
        scores = technical_results["scores"]
        origins = technical_results["origins"]
        
        await display_collection_results(docs, scores, origins, collection_name, 
                                        technical_results.get("reranked", False), user_session_id)
    
    # Display DCE collection results
    if dce_results.get("documents"):
        collection_name = dce_results.get("collection_name", "Collection DCE")
        docs = dce_results["documents"]
        scores = dce_results["scores"]
        origins = dce_results["origins"]
        
        await display_collection_results(docs, scores, origins, collection_name,
                                        dce_results.get("reranked", False), user_session_id)

async def display_collection_results(docs: List[Dict], scores: List[float], 
                                   origins: List[str], collection_name: str, 
                                   reranked: bool, user_session_id: str):
    """Display results from a specific collection with user-specific image processing"""
    if not docs:
        return
    
    # Header message
    rerank_info = " (avec re-classement)" if reranked else ""
    header = f"📚 **{collection_name}**{rerank_info} - {len(docs)} documents\n\n"
    
    # Limit display to first 5 documents
    display_limit = min(5, len(docs))
    
    for i in range(display_limit):
        doc = docs[i]
        score = scores[i] if i < len(scores) else 0.0
        origin = origins[i] if i < len(origins) else "unknown"
        
        # Extract metadata
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Unknown")
        dl_meta = metadata.get("dl_meta", {})
        
        # Format document display
        content = f"**Document {i+1}** (Score: {score:.4f})\n"
        content += f"**Source**: {source}\n"
        content += f"**Origine**: {origin}\n"
        
        # Add headings if available
        headings = dl_meta.get("headings", [])
        if headings and isinstance(headings, list):
            content += f"**Titres**: {' > '.join(headings)}\n"
        
        # Add content preview with user-specific image processing
        page_content = doc.get('page_content', '')
        
        # Process images for Chainlit display with user session
        processed_content = process_images_for_chainlit(page_content, user_session_id)
        
        content += f"\n{processed_content}"
                
        if i == 0:
            content = header + content
        
        await cl.Message(content=content).send()

async def ask_for_specifications():
    """Ask for user specifications"""
    try:
        # Ask for user input using AskUserMessage
        specifications_response = await cl.AskUserMessage(
            content="**Spécifications supplémentaires:**\n\nEntrez des spécifications techniques ou de style pour personnaliser la réponse. \n*Ecrire **aucunes** sinon*"
        ).send()
        
        user_specifications = specifications_response.get("output", "").strip()

        if user_specifications != "aucunes":
            session_manager = cl.user_session.get("session_manager")
            session_manager.user_specifications = user_specifications if user_specifications else None
            
            if user_specifications:
                await show_collection_selection_with_specs(user_specifications)
            else:
                await show_collection_selection_without_specs()
        else:
            await show_collection_selection_without_specs()
            
    except Exception as e:
        print(f"Error asking for specifications: {e}")
        await show_collection_selection_without_specs()

async def show_collection_selection_without_specs():
    """Show collection selection options without specifications"""
    session_manager = cl.user_session.get("session_manager")
    search_results = session_manager.last_search_results
    
    technical_docs = search_results.get("technical_results", {}).get("documents", [])
    dce_docs = search_results.get("dce_results", {}).get("documents", [])
    
    actions = []
    
    # Button for using both collections
    if technical_docs and dce_docs:
        actions.append(
            cl.Action(
                name="generate_response",
                payload={"use_both": True, "use_specifications": False},
                label="2️⃣ Utiliser les deux collections"
            )
        )
    
    # Button for using only technical collection
    if technical_docs:
        actions.append(
            cl.Action(
                name="generate_response",
                payload={"use_both": False, "use_technical": True, "use_specifications": False},
                label="📚 Utiliser uniquement la collection MT"
            )
        )
    
    # Button for using only DCE collection
    if dce_docs:
        actions.append(
            cl.Action(
                name="generate_response",
                payload={"use_both": False, "use_technical": False, "use_specifications": False},
                label="📋 Utiliser uniquement la collection DCE"
            )
        )
    
    # Cancel button
    actions.append(
        cl.Action(
            name="cancel_generation",
            payload={},
            label="❌ Annuler"
        )
    )
    
    total_docs = len(technical_docs) + len(dce_docs)
    
    response_msg = cl.Message(
        content=f"**{total_docs} documents trouvés. Choisissez la collection à utiliser pour la génération:**",
        actions=actions
    )
    
    await response_msg.send()

async def show_collection_selection_with_specs(user_specifications: str):
    """Show collection selection options with specifications"""
    session_manager = cl.user_session.get("session_manager")
    search_results = session_manager.last_search_results
    
    technical_docs = search_results.get("technical_results", {}).get("documents", [])
    dce_docs = search_results.get("dce_results", {}).get("documents", [])
    
    actions = []
    
    # Button for using both collections with specifications
    if technical_docs and dce_docs:
        actions.append(
            cl.Action(
                name="generate_response",
                payload={"use_both": True, "use_specifications": True},
                label="2️⃣ Utiliser les deux collections"
            )
        )
    
    # Button for using only technical collection with specifications
    if technical_docs:
        actions.append(
            cl.Action(
                name="generate_response",
                payload={"use_both": False, "use_technical": True, "use_specifications": True},
                label="📚 Utiliser uniquement la collection MT"
            )
        )
    
    # Button for using only DCE collection with specifications
    if dce_docs:
        actions.append(
            cl.Action(
                name="generate_response",
                payload={"use_both": False, "use_technical": False, "use_specifications": True},
                label="📋 Utiliser uniquement la collection DCE"
            )
        )
    
    # Cancel button
    actions.append(
        cl.Action(
            name="cancel_generation",
            payload={},
            label="❌ Annuler"
        )
    )
    
    specs_preview = user_specifications[:200] + "..." if len(user_specifications) > 200 else user_specifications
    
    response_msg = cl.Message(
        content=f"**Spécifications reçues:**\n\n{specs_preview}\n\n**Choisissez la collection à utiliser:**",
        actions=actions
    )
    
    await response_msg.send()

async def stream_chat_response_text_only(query: str, technical_docs: list, dce_docs: list, 
                                        user_specifications: str = None, use_both: bool = True, 
                                        use_technical: bool = False):
    """Simple text-only streaming without SSE parsing with user support"""
    session_manager = cl.user_session.get("session_manager")
    user_session_id = session_manager.user_session_id if session_manager else None
    
    chat_payload = {
        "query": query,
        "technical_docs": technical_docs,
        "dce_docs": dce_docs,
        "user_specifications": user_specifications,
        "use_both": use_both,
        "use_technical": use_technical
    }
    
    headers = {"X-User-Session": user_session_id} if user_session_id else {}
    
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    accumulated_content = ""  # Store the full response
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/chat/stream-text",
                json=chat_payload,
                headers=headers,
                timeout=300.0
            ) as response:
                
                if response.status_code != 200:
                    await cl.Message(
                        content=f"❌ **Erreur:** HTTP {response.status_code}"
                    ).send()
                    return
                
                async for chunk in response.aiter_text():
                    if "[STREAM_COMPLETE]" in chunk:
                        break
                    elif chunk.startswith("[ERROR:"):
                        await cl.Message(content=f"❌ {chunk}").send()
                        return
                    else:
                        accumulated_content += chunk  
                        await response_msg.stream_token(chunk)
        
        await response_msg.update()
        
        # Extract sources from documents
        sources = set()
        for doc in technical_docs + dce_docs:
            if doc and isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Unknown")
                page = metadata.get("page", "N/A")
                sources.add((source, page))
        
        # Get search results from session manager
        search_results = session_manager.last_search_results if session_manager else {}
        
        # Show export options with the accumulated content and user session
        _log.info(f"ACCUMULATED CONTENT: {accumulated_content}")

        await show_export_options(
            content=accumulated_content,
            query=query,
            sources=list(sources),
            technical_docs=technical_docs,
            dce_docs=dce_docs,
            search_results=search_results
        )
    
    except Exception as e:
        await cl.Message(content=f"❌ **Erreur:** {str(e)}").send()

@cl.action_callback("generate_response")
async def on_generate_response_action_streaming(action: cl.Action):
    """Handle response generation with streaming support and user session"""
    session_manager = cl.user_session.get("session_manager")
    
    if not session_manager or not session_manager.user_session_id:
        await cl.Message(content="❌ **Erreur de session**\n\nSession utilisateur invalide.").send()
        return
    
    search_results = session_manager.last_search_results
    query = session_manager.current_query
    
    # Extract action parameters
    use_both = action.payload.get("use_both", False)
    use_technical = action.payload.get("use_technical", False)
    use_specifications = action.payload.get("use_specifications", False)
    
    # Get specifications if needed
    user_specifications = None
    if use_specifications:
        user_specifications = session_manager.user_specifications
    
    # Prepare documents for the request
    technical_docs = search_results.get("technical_results", {}).get("documents", [])
    dce_docs = search_results.get("dce_results", {}).get("documents", [])
    
    # Filter documents based on selection
    if not use_both:
        if use_technical:
            dce_docs = []
        else:
            technical_docs = []
    
    # Show generation status
    gen_msg = await cl.Message(content="**Démarrage de la génération ...**").send()
    
    # Use streaming response generation
    await stream_chat_response_text_only(
        query=query,
        technical_docs=technical_docs,
        dce_docs=dce_docs,
        user_specifications=user_specifications,
        use_both=use_both,
        use_technical=use_technical
    )
    
    # Update the generation message
    gen_msg.content = "✅ **Génération streaming terminée**"
    await gen_msg.update()

@cl.action_callback("cancel_generation")
async def on_cancel_generation_action(action: cl.Action):
    """Handle generation cancellation"""
    await cl.Message(content="❌ **Génération annulée.** Vous pouvez demander un nouveau paragraphe.").send()

async def show_export_options(content: str, query: str, sources: List[Tuple[str, int]],
                            technical_docs: List[Dict], dce_docs: List[Dict], search_results: Dict):
    """Show export options after response generation with user session support"""
    session_manager = cl.user_session.get("session_manager")
    user_session_id = session_manager.user_session_id if session_manager else None
    
    actions = []
    
    tech_scores = search_results.get("technical_results", {}).get("scores", [])
    tech_origins = search_results.get("technical_results", {}).get("origins", [])
    dce_scores = search_results.get("dce_results", {}).get("scores", [])
    dce_origins = search_results.get("dce_results", {}).get("origins", [])

    # Simple response export with download (user session included in payload)
    actions.append(
        cl.Action(
            name="export_response_only_download",  
            payload={
                "content": content,
                "query": query,
                "sources": sources,
                "technical_docs": technical_docs,
                "technical_scores": tech_scores,
                "technical_origins": tech_origins,
                "dce_docs": dce_docs,
                "dce_scores": dce_scores,
                "dce_origins": dce_origins,
                "user_session_id": user_session_id
            },
            label="📄 Exporter le paragraphe généré (DOCX)"
        )
    )
    
    actions.append(
        cl.Action(
            name="export_complete_document", 
            payload={
                "content": content,
                "query": query,
                "sources": sources,
                "technical_docs": technical_docs,
                "technical_scores": tech_scores,
                "technical_origins": tech_origins,
                "dce_docs": dce_docs,
                "dce_scores": dce_scores,
                "dce_origins": dce_origins,
                "user_session_id": user_session_id
            },
            label="📑 Exporter tous les paragraphes récupérés (DOCX)"
        )
    )
    
    export_msg = cl.Message(
        content="**Options d'export disponibles:**",
        actions=actions
    )
    await export_msg.send()

@cl.action_callback("export_response_only_download")
async def on_export_response_only_download_action(action: cl.Action):
    """Handle simple response export with direct download and user session support"""
    try:
        content = action.payload.get("content", "")
        query = action.payload.get("query", "")
        sources = action.payload.get("sources", [])
        technical_docs = action.payload.get("technical_docs", [])
        technical_scores = action.payload.get("technical_scores", [])
        technical_origins = action.payload.get("technical_origins", [])
        dce_docs = action.payload.get("dce_docs", [])
        dce_scores = action.payload.get("dce_scores", [])
        dce_origins = action.payload.get("dce_origins", [])
        user_session_id = action.payload.get("user_session_id")
        
        if not content:
            await cl.Message(content="❌ Aucun contenu à exporter.").send()
            return
        
        export_status = await cl.Message(content="📄 **Export DOCX en cours...**").send()
        
        # Use the direct download endpoint with user session
        async with httpx.AsyncClient() as client:
            export_payload = {
                "content": content,
                "query": query,
                "sources": sources,
                "technical_docs": technical_docs,
                "technical_scores": technical_scores,
                "technical_origins": technical_origins,
                "dce_docs": dce_docs,
                "dce_scores": dce_scores,
                "dce_origins": dce_origins,
                "include_retrieved_docs": False,
                "include_images_catalog": True
            }
            
            headers = {"X-User-Session": user_session_id} if user_session_id else {}
            _log.info(f"Content envoyé: {content}")
            
            response = await client.post(
                f"{BACKEND_URL}/api/export/docx/direct",
                json=export_payload,
                headers=headers,
                timeout=60.0
            )
            
            if response.status_code == 200:
                # Get the binary file content
                file_content = response.content
                
                # Extract filename from headers
                content_disposition = response.headers.get('content-disposition', '')
                filename = "document.docx"
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                
                export_status.content = "✅ **Export réussi !** Téléchargement du document..."
                await export_status.update()
                
                # Create download element for Chainlit
                file_element = cl.File(
                    content=file_content,
                    name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    display="inline"
                )
                
                success_msg = f"📄 **Export terminé**\n\n"
                success_msg += f"**Fichier:** {filename}\n"
                success_msg += f"**Taille:** {len(file_content) / 1024:.1f} KB\n"
                if user_session_id:
                    success_msg += f"**Session:** {user_session_id}\n"
                success_msg += f"**Téléchargement disponible ci-dessous**"
                
                download_msg = cl.Message(content=success_msg, elements=[file_element])
                await download_msg.send()
                
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get("detail", f"Erreur HTTP {response.status_code}")
                await cl.Message(content=f"❌ **Erreur d'export:** {error_msg}").send()
    
    except Exception as e:
        await cl.Message(content=f"❌ **Erreur d'export:** {str(e)}").send()

@cl.action_callback("export_complete_document")
async def on_export_complete_document_action(action: cl.Action):
    """Handle complete document export with user session support"""
    try:
        content = action.payload.get("content", "")
        query = action.payload.get("query", "")
        sources = action.payload.get("sources", [])
        technical_docs = action.payload.get("technical_docs", [])
        technical_scores = action.payload.get("technical_scores", [])
        technical_origins = action.payload.get("technical_origins", [])
        dce_docs = action.payload.get("dce_docs", [])
        dce_scores = action.payload.get("dce_scores", [])
        dce_origins = action.payload.get("dce_origins", [])
        user_session_id = action.payload.get("user_session_id")
        
        if not content:
            await cl.Message(content="❌ Aucun contenu à exporter.").send()
            return
        
        export_status = await cl.Message(content="📑 **Export document complet en cours...**").send()
        
        # Use the direct download endpoint with user session
        async with httpx.AsyncClient() as client:
            export_payload = {
                "content": content,
                "query": query,
                "sources": sources,
                "technical_docs": technical_docs,
                "technical_scores": technical_scores,
                "technical_origins": technical_origins,
                "dce_docs": dce_docs,
                "dce_scores": dce_scores,
                "dce_origins": dce_origins,
                "include_retrieved_docs": True,
                "include_images_catalog": True
            }
            
            headers = {"X-User-Session": user_session_id} if user_session_id else {}
            
            response = await client.post(
                f"{BACKEND_URL}/api/export/docx/direct",
                json=export_payload,
                headers=headers,
                timeout=120.0  # Timeout plus long pour export complet
            )
            
            if response.status_code == 200:
                # Get binary file content
                file_content = response.content
                
                # Extract filename from headers
                content_disposition = response.headers.get('content-disposition', '')
                filename = "document_complet.docx"
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                
                export_status.content = "✅ **Export complet réussi !** Téléchargement du document..."
                await export_status.update()
                
                # Create download element for Chainlit
                file_element = cl.File(
                    content=file_content,
                    name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    display="inline"
                )
                
                success_msg = f"📑 **Export complet terminé**\n\n"
                success_msg += f"**Fichier:** {filename}\n"
                success_msg += f"**Taille:** {len(file_content) / 1024:.1f} KB\n"
                success_msg += f"**Documents techniques:** {len(technical_docs)}\n"
                success_msg += f"**Documents DCE:** {len(dce_docs)}\n"
                if user_session_id:
                    success_msg += f"**Session:** {user_session_id}\n"
                success_msg += f"**Téléchargement disponible ci-dessous**"
                
                download_msg = cl.Message(content=success_msg, elements=[file_element])
                await download_msg.send()
                
            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get("detail", f"Erreur HTTP {response.status_code}")
                await cl.Message(content=f"❌ **Erreur d'export complet:** {error_msg}").send()
    
    except Exception as e:
        await cl.Message(content=f"❌ **Erreur d'export complet:** {str(e)}").send()


if __name__ == "__main__":
    print("Starting Chainlit Frontend with User Support...")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Images will be served from: {BACKEND_URL}/images/user_{{session_id}}/")