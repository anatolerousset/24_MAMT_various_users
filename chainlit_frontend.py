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

# Session storage for response history
class SessionManager:
    def __init__(self):
        self.response_history = []
        self.last_search_results = {}
        self.current_query = ""
        self.user_specifications = None
        self.selected_region = None
        self.selected_office = None
        self.technical_collection = Config.COLLECTION_NAME
        self.dce_collection = Config.DCE_COLLECTION
    
    def add_response(self, content: str, query: str, sources: List[Tuple[str, int]], specifications: str = None):
        """Add response to history"""
        entry = {
            "content": content,
            "query": query,
            "sources": sources,
            "timestamp": datetime.now().strftime("%d/%m/%Y à %H:%M"),
            "specifications": specifications,
            "region": self.selected_region,
            "office": self.selected_office,
            "technical_collection": self.technical_collection,
            "dce_collection": self.dce_collection
        }
        self.response_history.append(entry)
    
    def set_region_and_office(self, region: str, office: str):
        """Set the selected region and office and update collection names"""
        self.selected_region = region
        self.selected_office = office
        
        if region:
            self.technical_collection = f"region_{region.lower()}_documents"
        else:
            self.technical_collection = Config.COLLECTION_NAME
            
        if office:
            self.dce_collection = f"dce_{office.lower()}_documents"
        else:
            self.dce_collection = Config.DCE_COLLECTION
            
        _log.info(f"Region set to: {region}, Office set to: {office}")
        _log.info(f"Technical Collection: {self.technical_collection}, DCE Collection: {self.dce_collection}")
    
    def get_history(self):
        """Get response history"""
        return self.response_history
    
    def clear_history(self):
        """Clear response history"""
        self.response_history = []

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

def process_images_for_chainlit(content: str) -> str:
    """
    Process image URLs in content to ensure they work with Chainlit.
    Convert backend image URLs to proper format for Chainlit display.
    
    Args:
        content: Content with image references
        
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
        else:
            filename = image_url

        new_path = f"{IMAGES_URL}/images/{filename}"
        _log.info(f"ancien chemin : {image_url}")
        _log.info(f"filename chemin : {filename}")
        _log.info(f"nouveau chemin : {new_path}")
        return f"![{alt_text}]({new_path})"

    processed_content = image_pattern.sub(replace_image_url, content)
    return processed_content

async def create_region_office_selector_element() -> cl.CustomElement:
    """Create a region and office selector custom element"""
    available_regions = Config.load_available_regions()
    available_offices = Config.load_available_offices()
    session_manager = cl.user_session.get("session_manager")
    
    # Get current selections from session or default to first available
    current_region = session_manager.selected_region if session_manager else None
    current_office = session_manager.selected_office if session_manager else None
    
    if not current_region and available_regions:
        current_region = available_regions[0]
    if not current_office and available_offices:
        current_office = available_offices[0]
        
    if session_manager and current_region and current_office:
        session_manager.set_region_and_office(current_region, current_office)
    
    region_office_element = cl.CustomElement(
        name="RegionOfficeSelector",
        props={
            "available_regions": available_regions,
            "available_offices": available_offices,
            "selected_region": current_region,
            "selected_office": current_office,
            "default_technical_collection": Config.COLLECTION_NAME,
            "default_dce_collection": Config.DCE_COLLECTION,
            "technical_collection_name": f"region_{current_region.lower()}_documents" if current_region else Config.COLLECTION_NAME,
            "dce_collection_name": f"dce_{current_office.lower()}_documents" if current_office else Config.DCE_COLLECTION
        }
    )
    
    # Store the element in session for later updates
    cl.user_session.set("region_office_selector", region_office_element)
    
    return region_office_element

@cl.action_callback("region_office_selected")
async def on_region_office_selected(action: cl.Action):
    """Handle region and office selection from the custom element"""
    try:
        region = action.payload.get("region")
        office = action.payload.get("office")
        technical_collection_name = action.payload.get("technical_collection_name")
        dce_collection_name = action.payload.get("dce_collection_name")
        
        if not region or not office:
            await cl.Message(content="❌ Région et bureau non spécifiés").send()
            return
        
        # Update session manager
        session_manager = cl.user_session.get("session_manager")
        if session_manager:
            session_manager.set_region_and_office(region, office)
        
        """
        # Send confirmation message
        await cl.Message(
            content=f"✅ **Sélection confirmée:**\n\n"
                   f"**Région:** {region}\n"
                   f"**Bureau:** {office}\n\n"
                   f"**Collection technique active:** `{technical_collection_name}`\n"
                   f"**Collection DCE active:** `{dce_collection_name}`\n\n"
                   f"Vous pouvez maintenant effectuer vos recherches avec cette configuration."
        ).send()
        """
        _log.info(f"Region/Office selected: {region}/{office}, Collections: {technical_collection_name}, {dce_collection_name}")
        
    except Exception as e:
        _log.error(f"Error handling region/office selection: {e}")
        await cl.Message(content=f"❌ Erreur lors de la sélection: {str(e)}").send()

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with region selector"""
    # Initialize session manager
    session_manager = SessionManager()
    cl.user_session.set("session_manager", session_manager)
    
    try:
        # Check backend health and get configuration
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"{BACKEND_URL}/health", timeout=10.0)
            config_response = await client.get(f"{BACKEND_URL}/api/config", timeout=10.0)
            
            if health_response.status_code == 200 and config_response.status_code == 200:
                health_data = health_response.json()
                config_data = config_response.json()
                
                status = health_data.get("status", "unknown")
                services = health_data.get("services", {})
                
                if status == "healthy":
                    welcome_msg = "🟢 **Mon assistant mémoire technique opérationnel**\n\n"
                    
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
                    
                    #welcome_msg += f"**Collections disponibles:**\n"
                    #welcome_msg += f"- Technique: `{collections.get('technical', 'N/A')}`\n"
                    #welcome_msg += f"- DCE: `{collections.get('dce', 'N/A')}`\n\n"
                    #welcome_msg += f"**LLM actif:** {llm_info.get('provider', 'N/A')}\n\n"
                    
                else:
                    welcome_msg = "🟡 **Système en cours d'initialisation**\n\nLe système démarre, veuillez patienter quelques instants."
            else:
                welcome_msg = "🔴 **Système non disponible**\n\nImpossible de se connecter au backend."
    except Exception as e:
        welcome_msg = f"🔴 **Erreur de connexion**\n\nErreur: {str(e)}"
    
    # Display collections info
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/collections", timeout=10.0)
            if response.status_code == 200:
                collections_data = response.json()
                collections = collections_data.get("collections", [])
                if collections:
                    _log.info(f"\n**Collections vectorielles:** {', '.join(collections)}")
    except Exception as e:
        _log.info(f"\n⚠️ Erreur lors de la récupération des collections: {str(e)}")

    # Add backend and streamlit URLs
    current_host = os.getenv("CHAINLIT_HOST", "localhost")
    streamlit_frontend_url = f"http://{current_host}:8501"
    welcome_msg += f"\n📁 **Interface d'ingestion Streamlit** {streamlit_frontend_url}\n\n"
    
    # Send welcome message
    await cl.Message(content=welcome_msg).send()
    
    # Create and send region selector
    region_selector = await create_region_office_selector_element()
    await cl.Message(
        content="",#"🗺️ **Sélection de région**\n\nVeuillez d'abord sélectionner la région pour configurer la collection technique appropriée:",
        elements=[region_selector]
    ).send()
    
    # Instructions
    await cl.Message(
        content="**Après avoir sélectionné votre région, écrivez le titre du paragraphe à générer pour commencer !**"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages with dual search and progress bar"""
    query = message.content
    session_manager = cl.user_session.get("session_manager")
    session_manager.current_query = query
    
    # Check if region and office are selected
    if not session_manager.selected_region or not session_manager.selected_office:
        await cl.Message(
            content="⚠️ **Veuillez d'abord sélectionner une région et un bureau** via le sélecteur ci-dessus avant de faire votre recherche."
        ).send()
        return
    
    # Show initial status message with progress bar
    status_msg = await cl.Message(
        content=f"🔍 **Recherche hybride en cours...**\n\n"
               f"Région: **{session_manager.selected_region}** | Bureau: **{session_manager.selected_office}**\n\n"
               f"Collection Technique: `{session_manager.technical_collection}`\n"
               f"Collection DCE: `{session_manager.dce_collection}`\n\n"
               f"[░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%"
    ).send()
    
    # Initialize progress_task to None
    progress_task = None
    
    try:
        # Start progress bar task
        progress_task = asyncio.create_task(update_progress_bar(status_msg, Config.RESEARCH_WAITING_TIME)) 
        
        # Perform dual search with selected collections
        async with httpx.AsyncClient() as client:
            search_payload = {
                "query": query,
                "technical_collection": session_manager.technical_collection,
                "dce_collection": session_manager.dce_collection,
                "threshold": 0.1,
                "max_results": 15,
                "use_reranker": True
            }
            
            # Start the search request
            search_task = asyncio.create_task(
                client.post(
                    f"{BACKEND_URL}/api/dual-search",
                    json=search_payload,
                    timeout=60.0
                )
            )
            
            # Wait for search to complete
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
                status_msg.content = (f"✅ **Recherche terminée**\n\n"
                                    f"Région: **{session_manager.selected_region}** | Bureau: **{session_manager.selected_office}**\n\n"
                                    f"Collection Technique: `{session_manager.technical_collection}`\n"
                                    f"Collection DCE: `{session_manager.dce_collection}`\n\n"
                                    f"{total_documents} documents trouvés")
                await status_msg.update()
                
                # Display results
                await display_search_results(technical_results, dce_results)
                
                if total_documents == 0:
                    await cl.Message(
                        content=f"❌ **Aucun document trouvé**\n\n"
                               f"Région: {session_manager.selected_region} | Bureau: {session_manager.selected_office}\n\n"
                               f"Veuillez reformuler votre titre de paragraphe ou essayer une autre configuration."
                    ).send()
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

async def display_search_results(technical_results: Dict, dce_results: Dict):
    """Display search results from both collections with proper image handling"""
    
    # Display technical collection results
    if technical_results.get("documents"):
        collection_name = technical_results.get("collection_name", "Collection Technique")
        docs = technical_results["documents"]
        scores = technical_results["scores"]
        origins = technical_results["origins"]
        
        await display_collection_results(docs, scores, origins, collection_name, 
                                       technical_results.get("reranked", False))
    
    # Display DCE collection results
    if dce_results.get("documents"):
        collection_name = dce_results.get("collection_name", "Collection DCE")
        docs = dce_results["documents"]
        scores = dce_results["scores"]
        origins = dce_results["origins"]
        
        await display_collection_results(docs, scores, origins, collection_name,
                                       dce_results.get("reranked", False))

async def display_collection_results(docs: List[Dict], scores: List[float], 
                                   origins: List[str], collection_name: str, reranked: bool):
    """Display results from a specific collection with image processing"""
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
        
        # Add content preview with image processing
        page_content = doc.get('page_content', '')
        
        # Process images for Chainlit display
        processed_content = process_images_for_chainlit(page_content)
        
        content += f"\n{processed_content}" #[:500]}..."
                
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
    """Simple text-only streaming without SSE parsing"""
    
    chat_payload = {
        "query": query,
        "technical_docs": technical_docs,
        "dce_docs": dce_docs,
        "user_specifications": user_specifications,
        "use_both": use_both,
        "use_technical": use_technical
    }
    
    response_msg = cl.Message(content="")
    await response_msg.send()
    
    accumulated_content = ""  # Store the full response
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{BACKEND_URL}/api/chat/stream-text",
                json=chat_payload,
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
        session_manager = cl.user_session.get("session_manager")
        search_results = session_manager.last_search_results if session_manager else {}
        
        # Show export options with the accumulated content
        _log.info(f"ACCUMULATED CONTENET: {accumulated_content}")

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
    """Handle response generation with streaming support"""
    session_manager = cl.user_session.get("session_manager")
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
    """Show export options after response generation"""
    actions = []
    
    tech_scores = search_results.get("technical_results", {}).get("scores", [])
    tech_origins = search_results.get("technical_results", {}).get("origins", [])
    dce_scores = search_results.get("dce_results", {}).get("scores", [])
    dce_origins = search_results.get("dce_results", {}).get("origins", [])

    # Simple response export with download
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
                    "dce_origins": dce_origins
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
                "dce_origins": dce_origins
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
    """Handle simple response export with direct download"""
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
        
        if not content:
            await cl.Message(content="❌ Aucun contenu à exporter.").send()
            return
        
        export_status = await cl.Message(content="📄 **Export DOCX en cours...**").send()
        
        # Use the direct download endpoint
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
            _log.info(f"Content envoyé: {content}")
            
            response = await client.post(
                f"{BACKEND_URL}/api/export/docx/direct",
                json=export_payload,
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
    """Handle complete document export"""
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
        
        if not content:
            await cl.Message(content="❌ Aucun contenu à exporter.").send()
            return
        
        export_status = await cl.Message(content="📑 **Export document complet en cours...**").send()
        
        # Utiliser l'endpoint de téléchargement direct
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
            
            response = await client.post(
                f"{BACKEND_URL}/api/export/docx/direct",
                json=export_payload,
                timeout=120.0  # Timeout plus long pour export complet
            )
            
            if response.status_code == 200:
                # Récupérer le fichier binaire
                file_content = response.content
                
                # Extraire le nom du fichier depuis les headers
                content_disposition = response.headers.get('content-disposition', '')
                filename = "document_complet.docx"
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                
                export_status.content = "✅ **Export complet réussi !** Téléchargement du document..."
                await export_status.update()
                
                # Créer un élément de téléchargement pour Chainlit
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
    print("Starting Chainlit Frontend ...")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Images will be served from: {BACKEND_URL}/images/")