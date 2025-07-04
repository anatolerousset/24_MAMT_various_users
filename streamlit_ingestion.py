# streamlit_ingestion.py (Complete version with user management)
import streamlit as st
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
import os
import sys
import httpx
import asyncio
import time
import json
import threading
from datetime import datetime

from config.config import Config
from utils.blob_utils import BlobStorageManager

import logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Backend URL from environment
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8001")

# Initialize user session management in Streamlit
def initialize_user_session():
    """Initialize or get user session for Streamlit"""
    
    # Check if we already have a session
    if 'user_session_id' in st.session_state and st.session_state.user_session_id:
        return st.session_state.user_session_id
    
    # Create new session
    try:
        response = asyncio.run(create_user_session_async())
        if response and response.get("success"):
            session_id = response["data"]["session_id"]
            st.session_state.user_session_id = session_id
            st.session_state.user_info = response["data"]
            return session_id
        else:
            st.error("❌ Impossible de créer une session utilisateur")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la création de session: {e}")
        return None

async def create_user_session_async():
    """Create user session asynchronously"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/user/session",
                json={"user_identifier": None},  # Could be customized
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def get_user_info_async(user_session_id: str):
    """Get user info asynchronously"""
    try:
        async with httpx.AsyncClient() as client:
            headers = {"X-User-Session": user_session_id}
            response = await client.get(
                f"{BACKEND_URL}/api/user/info",
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def cleanup_user_session_async(user_session_id: str):
    """Cleanup user session asynchronously"""
    try:
        async with httpx.AsyncClient() as client:
            headers = {"X-User-Session": user_session_id}
            response = await client.delete(
                f"{BACKEND_URL}/api/user/session",
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Ingestion function with user support
async def start_basic_ingestion_user(ingestion_params: dict, user_session_id: str):
    """Start basic ingestion without progress tracking with user session support"""
    try:
        headers = {"X-User-Session": user_session_id} if user_session_id else {}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/ingestion",
                json=ingestion_params,
                headers=headers,
                timeout=300.0
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_user_info():
    """Display user session information in sidebar"""
    user_session_id = st.session_state.get('user_session_id')
    
    if user_session_id:
        with st.sidebar:
            st.markdown("### 👤 Session Utilisateur")
            
            # Try to get fresh user info
            try:
                user_info_response = asyncio.run(get_user_info_async(user_session_id))
                if user_info_response.get("success"):
                    user_info = user_info_response["data"]
                    
                    st.info(f"**ID de session:** `{user_session_id[:12]}...`")
                    st.info(f"**Collection DCE:** `{user_info.get('dce_collection', 'N/A')}`")
                    
                    if user_info.get('processed_files_count', 0) > 0:
                        st.success(f"**Fichiers traités:** {user_info['processed_files_count']}")
                    
                    # Session cleanup button
                    if st.button("🧹 Nettoyer ma session", help="Supprimer les données temporaires"):
                        cleanup_result = asyncio.run(cleanup_user_session_async(user_session_id))
                        if cleanup_result.get("success"):
                            st.success("✅ Session nettoyée")
                            # Reset session state
                            for key in ['user_session_id', 'user_info']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                        else:
                            st.error("❌ Erreur lors du nettoyage")
                else:
                    st.warning("⚠️ Session invalide, création d'une nouvelle session...")
                    # Reset and create new session
                    for key in ['user_session_id', 'user_info']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Erreur session: {e}")

def display_launch_section(uploaded_files, validation_errors, data_type, region_name, 
                          recreate_collection, remove_duplicates, archive_processed_files, user_session_id):
    """Launch section with basic loading and user session support"""
    
    # Initialize session state for ingestion status
    if 'ingestion_in_progress' not in st.session_state:
        st.session_state.ingestion_in_progress = False
    
    # Centered launch button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Button is disabled if there are validation errors OR if ingestion is in progress OR no user session
        is_disabled = bool(validation_errors) or st.session_state.ingestion_in_progress or not user_session_id
        
        # Change button text based on ingestion status
        if not user_session_id:
            button_text = "❌ Session utilisateur requise"
        elif st.session_state.ingestion_in_progress:
            button_text = "🔄 Ingestion en cours..."
        else:
            button_text = "👷 Démarrer l'Ingestion"
        
        launch_button = st.button(
            button_text,
            disabled=is_disabled,
            use_container_width=True,
            type="primary"
        )
    
    # Handle launch button click
    if launch_button and not st.session_state.ingestion_in_progress and user_session_id:
        try:
            # Set ingestion status to in progress
            st.session_state.ingestion_in_progress = True
            
            # Force a rerun to update the button state immediately
            st.rerun()

        except Exception as e:
            # Reset ingestion status on error
            st.session_state.ingestion_in_progress = False
            st.error(f"❌ Erreur lors du démarrage de l'ingestion: {str(e)}")
    
    # If ingestion is in progress, run the actual ingestion process
    if st.session_state.ingestion_in_progress:
        run_ingestion_process(uploaded_files, data_type, region_name, 
                             recreate_collection, remove_duplicates, archive_processed_files, user_session_id)

def run_ingestion_process(uploaded_files, data_type, region_name, 
                         recreate_collection, remove_duplicates, archive_processed_files, user_session_id):
    """Separate function to handle the actual ingestion process with user session support"""
    try:
        blob_manager = BlobStorageManager()
        
        with st.spinner("📤 Téléchargement des fichiers..."):
            uploaded_blob_names = upload_files_to_blob(uploaded_files, blob_manager)
            
            if not uploaded_blob_names:
                st.error("❌ Aucun fichier n'a pu être téléchargé")
                st.session_state.ingestion_in_progress = False
                return
        
        # Prepare ingestion parameters
        ingestion_params = {
            'data_type': data_type,
            'recreate_collection': recreate_collection,
            'remove_duplicates': remove_duplicates,
            'archive_processed_files': archive_processed_files,
            'user_session_id': user_session_id  # Add user session to params
        }
        
        # Determine collection name
        if data_type == "dce":
            final_collection_name = "dce_documents"
        elif data_type == "region" and region_name:
            final_collection_name = f"region_{region_name}_documents"
            ingestion_params['region_name'] = region_name
        
        ingestion_params['collection_name'] = final_collection_name
        
        # Start ingestion with loading spinner and timer
        start_time = datetime.now()
        
        with st.spinner("🔄 Ingestion en cours... "):
            # Show timer in sidebar or below spinner
            timer_placeholder = st.empty()
            
            try:
                # Run ingestion in a separate thread to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(start_basic_ingestion_user(ingestion_params, user_session_id))
                    )
                    
                    # Update timer while waiting
                    while not future.done():
                        elapsed_time = datetime.now() - start_time
                        elapsed_seconds = int(elapsed_time.total_seconds())
                        elapsed_minutes = elapsed_seconds // 60
                        elapsed_seconds = elapsed_seconds % 60
                        
                        timer_placeholder.info(
                            f"⏱️ Temps écoulé: {elapsed_minutes:02d}:{elapsed_seconds:02d} - "
                            f"Le traitement peut prendre plusieurs minutes selon la taille des fichiers et leur type..."
                        )
                        
                        time.sleep(1)
                    
                    # Get the result
                    ingestion_result = future.result()
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'ingestion: {str(e)}")
                st.session_state.ingestion_in_progress = False
                return
            finally:
                timer_placeholder.empty()
        
        # Reset ingestion status
        st.session_state.ingestion_in_progress = False
        
        # Handle result
        if ingestion_result.get("success"):
            st.success("✅ Ingestion terminée avec succès!")
            
            # Update user info in session state
            if 'user_info' in st.session_state:
                try:
                    user_info_response = asyncio.run(get_user_info_async(user_session_id))
                    if user_info_response.get("success"):
                        st.session_state.user_info = user_info_response["data"]
                except Exception as e:
                    _log.warning(f"Could not update user info: {e}")
            
            # Show restart button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Nouvelle Ingestion", type="secondary", use_container_width=True):
                    # Reset any relevant session state
                    if 'ingestion_in_progress' in st.session_state:
                        del st.session_state.ingestion_in_progress
                    st.rerun()
        else:
            error_msg = ingestion_result.get("error", "Erreur inconnue")
            st.error(f"❌ Échec de l'ingestion: {error_msg}")
            
            # Show retry button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Réessayer", type="secondary", use_container_width=True):
                    # Reset ingestion status for retry
                    st.session_state.ingestion_in_progress = False
                    st.rerun()
            
    except Exception as e:
        # Reset ingestion status on any error
        st.session_state.ingestion_in_progress = False
        st.error(f"❌ Erreur lors du démarrage de l'ingestion: {str(e)}")

# Helper functions (unchanged from original)
def load_regions_from_env() -> List[str]:
    """Load available regions from environment variables"""
    try:
        regions_str = os.getenv('AVAILABLE_REGIONS', '')
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
       
        return regions if regions else ["PACA", "LR"]
       
    except Exception as e:
        st.warning(f"Error loading regions: {e}. Using default values.")
        return ["PACA", "LR"]

def get_allowed_file_types(data_type: str) -> tuple:
    """Return allowed file types based on data type"""
    if data_type == "region":
        return (['docx'], "DOCX uniquement pour les données de type région")
    elif data_type == "dce":
        return (['pdf', 'docx', 'xlsx', 'xls'], "PDF, DOCX, XLSX, XLS pour les données de type DCE. <strong>Pour les DCE merci de mettre RC et CCTP en une seule fois.</strong>")
    else:
        return (['pdf', 'docx', 'xlsx', 'xls'], "Tous les formats pris en charge")

def validate_uploaded_files(uploaded_files: List, data_type: str) -> List[str]:
    """Validate that uploaded files match the selected data type"""
    allowed_types, _ = get_allowed_file_types(data_type)
    validation_errors = []
    
    if uploaded_files:
        for file in uploaded_files:
            file_extension = file.name.split('.')[-1].lower()
            if file_extension not in allowed_types:
                validation_errors.append(
                    f"Fichier '{file.name}': type '{file_extension}' non autorisé pour le type '{data_type}'. "
                    f"Types autorisés: {', '.join(allowed_types).upper()}"
                )
    
    return validation_errors

async def check_backend_health():
    """Check if backend is healthy"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def get_collections_from_backend():
    """Get available collections from backend"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/collections", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {"collections": []}
    except Exception as e:
        return {"collections": []}

def add_navigation_button():
    """Add navigation button to the top right"""
    current_host = st.get_option("server.address") or "localhost"
    
    if current_host == "0.0.0.0":
        current_host = os.getenv("STREAMLIT_HOST", "localhost")
    
    second_frontend_url = f"http://{current_host}:8002"
    
    st.markdown(f"""
    <div class="nav-button-container">
        <a href="{second_frontend_url}" target="_blank" class="nav-button">
            Interface chat mon assistant MT
        </a>
    </div>
    """, unsafe_allow_html=True)

def upload_files_to_blob(uploaded_files: List, blob_manager: BlobStorageManager) -> List[str]:
    """Upload files to blob storage"""
    uploaded_blob_names = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for uploaded_file in uploaded_files:
            file_path = temp_path / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                blob_name = blob_manager.upload_file_to_blob(
                    file_path, 
                    Config.INPUT_CONTAINER, 
                    uploaded_file.name
                )
                uploaded_blob_names.append(blob_name)
            except Exception as e:
                st.error(f"Erreur lors du téléchargement de {uploaded_file.name}: {str(e)}")
    
    return uploaded_blob_names

def display_validation_errors(errors: List[str]) -> None:
    """Display validation errors with consistent styling"""
    if not errors:
        return
    
    error_html = '<div class="status-box status-warning">'
    
    for error in errors:
        error_html += f'<div style="margin-left: 20px;">• {error}</div>'
    
    error_html += '</div>'
    st.markdown(error_html, unsafe_allow_html=True)

def display_sidebar_config():
    """Display configuration in sidebar with consistent styling"""
    with st.sidebar:
        st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
        
        try:
            health_data = asyncio.run(check_backend_health())
            
            if health_data.get("status") == "healthy":
                st.success("✅ Backend opérationnel")
                
                collections_data = asyncio.run(get_collections_from_backend())
                collections = collections_data.get("collections", [])
                if collections:
                    _log.info(f"📚 Collections disponibles: {', '.join(collections)}")
            else:
                st.error(f"❌ Backend non opérationnel: {health_data.get('error', 'Erreur inconnue')}")
                st.warning("Veuillez vous assurer que le backend fonctionne et est accessible.")
        except Exception as e:
            st.error(f"❌ Impossible de se connecter au backend: {str(e)}")

        st.divider()

        # Data type section
        st.markdown("**📚 Type de collection**")
        data_type = st.selectbox(
            "Data type",
            options=["dce", "region"],
            help="Sélectionnez le type de données à traiter",
            label_visibility="collapsed"
        )
        
        # Region section
        region_name = None
        if data_type == "region":
            st.markdown("**🌍 Sélection de la Region**")
            
            available_regions = load_regions_from_env()
            
            if available_regions:
                region_name = st.selectbox(
                    "Region",
                    options=available_regions,
                    help="Sélectionnez la région à traiter",
                    label_visibility="collapsed"
                )
                
                if region_name:
                    region_name = region_name.lower().strip()
            else:
                st.error("❌ Aucune région disponible. Vérifiez la configuration des variables d'environnement.")
        
        st.divider()
        
        # Advanced options section
        st.markdown("**🔧 Options Avancées**")
        
        default_recreate = data_type == "dce"
        default_archive = data_type == "region"
      
        st.checkbox(
            "🆕 Recréer la collection",
            value=default_recreate,
            help="Supprimer et recréer complètement la collection",
            disabled=True
        )
        
        st.checkbox(
            "♻️ Supprimer les doublons",
            value=True,
            help="Analyser et supprimer les documents en double après ingestion",
            disabled=True
        )
        
        st.checkbox(
            "📦 Archiver les documents traités",
            value=default_archive,
            help="Les fichiers DCE ne sont pas archivés, mais les MT le sont",
            disabled=True
        )
        
        recreate_collection = default_recreate
        remove_duplicates = True
        archive_processed_files = default_archive
        
        return data_type, region_name, recreate_collection, remove_duplicates, archive_processed_files

def display_upload_section(data_type: str):
    """Display upload section with consistent styling"""
    st.markdown('<div class="section-header"> </div>', unsafe_allow_html=True)
    
    allowed_types, help_message = get_allowed_file_types(data_type)

    uploaded_files = st.file_uploader(
        "Select files",
        type=allowed_types,
        accept_multiple_files=True,
        help=help_message,
        label_visibility="collapsed"
    )
    
    st.markdown(
        f'<div class="file-type-notice">📋 <strong>Extensions autorisées:</strong> {help_message}. </div>',
        unsafe_allow_html=True
    )
        
    return uploaded_files

def main():
    """Main interface with user session management"""
    
    # Initialize user session first
    user_session_id = initialize_user_session()
    
    # Add navigation button
    add_navigation_button()
    
    # Main header
    st.markdown('<h1 class="main-header">📁 Interface d\'ingestion des documents</h1>', unsafe_allow_html=True)
    
    # Display user info in sidebar
    display_user_info()
    
    # Configuration from sidebar
    data_type, region_name, recreate_collection, remove_duplicates, archive_processed_files = display_sidebar_config()
    
    # Upload section with conditional file types
    uploaded_files = display_upload_section(data_type)
    
    # Parameter validation
    validation_errors = []
    
    if not uploaded_files:
        validation_errors.append("Aucun fichier sélectionné")
    else:
        # Validate that files match the selected data type
        file_validation_errors = validate_uploaded_files(uploaded_files, data_type)
        validation_errors.extend(file_validation_errors)
    
    if data_type == "region" and not region_name:
        validation_errors.append("Une région doit être sélectionnée pour le type 'region'")
    
    if not user_session_id:
        validation_errors.append("Session utilisateur non initialisée")
    
    # Display validation errors
    display_validation_errors(validation_errors)
    
    # Launch section - now with user session support
    display_launch_section(
        uploaded_files, validation_errors, data_type, region_name,
        recreate_collection, remove_duplicates, archive_processed_files, user_session_id
    )

# Page configuration
st.set_page_config(
    page_title="Ingestion des documents",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styles (unchanged from original)
st.markdown("""
<style>
    :root {
        --primary-color: #2E86AB;
        --success-color: #28a745;
        --error-color: #dc3545;
        --warning-color: #ffc107;
        --info-color: #17a2b8;
        --border-radius: 8px;
        --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        --padding-standard: 15px;
        --margin-standard: 15px;
    }
    
    .nav-button-container {
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 999999;
    }

    .nav-button {
        background-color: var(--primary-color);
        color: white !important;
        padding: 12px 20px;
        border: none;
        border-radius: var(--border-radius);
        font-size: 14px;
        font-weight: 600;
        text-decoration: none !important;
        box-shadow: var(--box-shadow);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }

    .nav-button:hover {
        background-color: #246A87;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: white !important;
        text-decoration: none !important;
    }
    
    .main-header {
        text-align: center;
        color: var(--primary-color);
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        padding: var(--padding-standard);
        border-bottom: 3px solid var(--primary-color);
        background: linear-gradient(135deg, var(--card), var(--accent));
        border-radius: var(--border-radius) var(--border-radius) 0 0;
    }
    
    .section-header {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: 600;
        margin: var(--margin-standard) 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
        background: linear-gradient(90deg, var(--primary-color), transparent);
        background-clip: text;
        -webkit-background-clip: text;
    }
    
    .status-box {
        padding: var(--padding-standard);
        border-radius: var(--border-radius);
        margin: var(--margin-standard) 0;
        box-shadow: var(--box-shadow);
        font-weight: 500;
        border: 1px solid var(--border);
    }
    
    .status-success {
        background-color: hsla(120, 65%, 45%, 0.1);
        border-left: 4px solid var(--success-color);
        color: var(--success-color);
        border-color: var(--success-color);
    }
    
    .status-error {
        background-color: hsla(0, 65%, 55%, 0.1);
        border-left: 4px solid var(--error-color);
        color: var(--error-color);
        border-color: var(--error-color);
    }
    
    .status-warning {
        background-color: hsla(45, 100%, 50%, 0.1);
        border-left: 4px solid var(--warning-color);
        color: hsl(45, 80%, 25%);
        border-color: var(--warning-color);
    }
    
    .status-info {
        background-color: hsla(210, 65%, 75%, 0.1);
        border-left: 4px solid var(--info-color);
        color: var(--info-color);
        border-color: var(--info-color);
    }
    
    .file-type-notice {
        background-color: hsla(210, 65%, 75%, 0.1);
        border: 1px solid var(--info-color);
        border-radius: var(--border-radius);
        padding: 10px var(--padding-standard);
        margin: 10px 0;
        color: var(--info-color);
        font-size: 0.9rem;
        border-left: 4px solid var(--info-color);
    }
    
    .info-card {
        background-color: hsla(210, 65%, 75%, 0.05);
        border-radius: var(--border-radius);
        padding: var(--padding-standard);
        margin: var(--margin-standard) 0;
        border: 1px solid hsla(210, 65%, 75%, 0.2);
    }
    
    .stDeployButton {
        display: none;
    }
    
    [data-testid="stToolbar"] {
        display: none;
    }
    
    .stActionButton {
        display: none;
    }
    
    button[title="Deploy this app"] {
        display: none;
    }
    
    /* Style metric containers */
    [data-testid="metric-container"] {
        background-color: hsla(210, 65%, 75%, 0.05);
        border: 1px solid hsla(210, 65%, 75%, 0.2);
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* French File Uploader Styles */

/* Hide the original "Browse files" button text and replace with French */
[data-testid="stFileUploader"] button[kind="secondary"] {
    font-size: 0;
}

[data-testid="stFileUploader"] button[kind="secondary"]:after {
    content: "Parcourir les fichiers";
    font-size: 14px;
    font-weight: 500;
}

/* Change the drag and drop text */
[data-testid="stFileUploaderDropzoneInstructions"] .st-emotion-cache-9ycgxx {
    font-size: 0;
}

[data-testid="stFileUploaderDropzoneInstructions"] .st-emotion-cache-9ycgxx::before {
    content: "Glissez et déposez vos fichiers ici";
    font-size: 16px;
    color: inherit;
    font-weight: 500;
}

/* Change the file limit text */
[data-testid="stFileUploaderDropzoneInstructions"] .st-emotion-cache-1rpn56r {
    font-size: 0;
}

[data-testid="stFileUploaderDropzoneInstructions"] .st-emotion-cache-1rpn56r::before {
    content: "Limite de 200 Mo par fichier • PDF, DOCX, XLSX, XLS";
    font-size: 12px;
    color: inherit;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()