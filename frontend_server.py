"""
Frontend FastAPI server with single user access control integration
"""
import os
import subprocess
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
import json
from typing import Dict, Any, List
from sse_starlette.sse import EventSourceResponse

# Import single user access control
from utils.single_user_manager import get_single_user_manager, require_single_user_access

# Global variables to track subprocess
streamlit_process = None
chainlit_process = None
backend_url = os.getenv("BACKEND_URL", "http://backend:8001")
start_time = time.time()

def start_streamlit():
    """Start Streamlit application"""
    global streamlit_process
    try:
        print("Starting Streamlit with access control...")
        streamlit_process = subprocess.Popen([
            "streamlit", "run", "streamlit_ingestion.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        print("Streamlit started successfully")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

def start_chainlit():
    """Start Chainlit application"""
    global chainlit_process
    try:
        print("Starting Chainlit with access control...")
        # Set backend URL for Chainlit
        env = os.environ.copy()
        env["BACKEND_URL"] = backend_url
        
        chainlit_process = subprocess.Popen([
            "chainlit", "run", "chainlit_frontend.py",
            "--port", "8002",
            "--host", "0.0.0.0"
        ], env=env)
        print("Chainlit started successfully")
    except Exception as e:
        print(f"Error starting Chainlit: {e}")

async def get_current_session(request: Request):
    """FastAPI dependency to get current session info"""
    user_ip = request.client.host
    session_id = request.headers.get("x-session-id")
    
    if not session_id:
        raise HTTPException(
            status_code=401,
            detail={"error": "no_session", "message": "Aucune session fournie"}
        )
    
    manager = get_single_user_manager()
    if not manager.validate_session(session_id, user_ip, "frontend"):
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_session", "message": "Session invalide ou expirée"}
        )
    
    return {
        "session_id": session_id,
        "user_ip": user_ip
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting frontend applications with single user access control...")
    
    # Check backend availability first
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/health", timeout=10.0)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ Backend is {health_data.get('status', 'unknown')}")
            else:
                print(f"⚠️ Backend responded with status {response.status_code}")
    except Exception as e:
        print(f"⚠️ Backend health check failed: {e}")
    
    # Start Streamlit in a separate thread
    streamlit_thread = threading.Thread(target=start_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Wait a bit for Streamlit to start
    await asyncio.sleep(3)
    
    # Start Chainlit in a separate thread
    chainlit_thread = threading.Thread(target=start_chainlit)
    chainlit_thread.daemon = True
    chainlit_thread.start()
    
    # Wait for both to be ready
    await asyncio.sleep(5)
    print("Frontend applications started with access control")
    
    yield
    
    # Shutdown
    global streamlit_process, chainlit_process
    
    print("Shutting down frontend applications...")
    
    if streamlit_process:
        streamlit_process.terminate()
        streamlit_process.wait()
        print("Streamlit stopped")
    
    if chainlit_process:
        chainlit_process.terminate()
        chainlit_process.wait()
        print("Chainlit stopped")

app = FastAPI(title="RAG Frontend Server with Single User Access", version="2.2.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - service selection with access control info"""
    # Get backend status
    backend_status = "unknown"
    backend_info = {}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/health", timeout=5.0)
            if response.status_code == 200:
                health_data = response.json()
                backend_status = health_data.get("status", "unknown")
                backend_info = health_data
    except Exception as e:
        backend_status = f"error: {str(e)}"
    
    # Get session status
    manager = get_single_user_manager()
    session_status = manager.get_session_status()
    
    return {
        "message": "RAG Frontend Server with Single User Access Control",
        "version": "2.2.0",
        "backend_status": backend_status,
        "backend_url": backend_url,
        "access_control": {
            "enabled": True,
            "mode": "single_user",
            "session_active": session_status.get("active", False),
            "session_info": session_status.get("session_info", {}) if session_status.get("active") else None
        },
        "services": {
            "ingestion": {
                "url": "http://localhost:8501",
                "description": "Streamlit interface for document ingestion (requires authentication)",
                "protected": True
            },
            "chat": {
                "url": "http://localhost:8002", 
                "description": "Chainlit chat interface with full RAG features (requires authentication)",
                "protected": True
            },
            "health": {
                "url": "http://localhost:8000/health",
                "description": "Frontend health status (requires authentication)",
                "protected": True
            },
            "auth": {
                "request_access": "http://localhost:8000/api/auth/request-access",
                "session_status": "http://localhost:8000/api/auth/session-status", 
                "logout": "http://localhost:8000/api/auth/logout",
                "description": "Authentication endpoints"
            }
        },
        "features": [
            "Single user access control",
            "Session management with timeout",
            "Dual collection search",
            "Image processing integration",
            "Azure Blob Storage exports",
            "Complete DOCX document export",
            "Real-time LLM response generation"
        ],
        "backend_info": backend_info
    }

# NEW: Access control endpoints (proxy to backend)
@app.post("/api/auth/request-access")
async def request_system_access(request: Request):
    """Request access to the system (proxy to backend)"""
    try:
        async with httpx.AsyncClient() as client:
            # Forward the request to backend
            response = await client.post(
                f"{backend_url}/api/auth/request-access",
                headers=dict(request.headers),
                timeout=10.0
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 423:
                # Return the locked response as-is
                return JSONResponse(
                    status_code=423,
                    content=response.json()
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                )
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")

@app.get("/api/auth/session-status")
async def get_session_status(request: Request):
    """Get current session status (proxy to backend)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{backend_url}/api/auth/session-status",
                headers=dict(request.headers),
                timeout=10.0
            )
            
            return response.json() if response.status_code == 200 else {"status": {"active": False}}
    except Exception:
        return {"status": {"active": False}}

@app.post("/api/auth/logout")
async def logout(request: Request):
    """Logout from the system (proxy to backend)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend_url}/api/auth/logout",
                headers=dict(request.headers),
                timeout=10.0
            )
            
            return response.json() if response.status_code == 200 else {"success": False}
    except Exception:
        return {"success": False}

@app.post("/api/auth/force-logout")
async def force_logout(request: Request):
    """Force logout (admin function, proxy to backend)"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend_url}/api/auth/force-logout",
                headers=dict(request.headers),
                timeout=10.0
            )
            
            return response.json() if response.status_code == 200 else {"success": False}
    except Exception:
        return {"success": False}

@app.get("/health")
@require_single_user_access("frontend")
async def health_check(request: Request, session: dict = Depends(get_current_session)):
    """Protected health check endpoint"""
    services_status = {}
    
    # Check Streamlit
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8501", timeout=5.0)
            services_status["streamlit"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services_status["streamlit"] = "unhealthy"
    
    # Check Chainlit
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8002", timeout=5.0)
            services_status["chainlit"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services_status["chainlit"] = "unhealthy"
    
    # Check Backend with session
    backend_detailed_status = {}
    try:
        async with httpx.AsyncClient() as client:
            headers = {"x-session-id": session["session_id"]}
            response = await client.get(f"{backend_url}/health", headers=headers, timeout=5.0)
            if response.status_code == 200:
                backend_data = response.json()
                services_status["backend"] = backend_data.get("status", "unknown")
                backend_detailed_status = backend_data.get("services", {})
            else:
                services_status["backend"] = f"http_error_{response.status_code}"
    except Exception as e:
        services_status["backend"] = f"connection_error"
        backend_detailed_status["error"] = str(e)
    
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values()
    ) else "degraded"
    
    return {
        "status": overall_status,
        "frontend_services": services_status,
        "backend_services": backend_detailed_status,
        "backend_url": backend_url,
        "session_info": session,
        "access_control": {
            "enabled": True,
            "session_active": True,
            "session_id": session["session_id"][:8] + "..."
        },
        "timestamp": time.time(),
        "uptime_info": {
            "streamlit_running": streamlit_process is not None and streamlit_process.poll() is None,
            "chainlit_running": chainlit_process is not None and chainlit_process.poll() is None
        }
    }

@app.get("/ingestion")
async def redirect_to_streamlit():
    """Redirect to Streamlit ingestion interface"""
    return RedirectResponse(url="http://localhost:8501")

@app.get("/chat")
async def redirect_to_chainlit():
    """Redirect to Chainlit chat interface"""
    return RedirectResponse(url="http://localhost:8002")

@app.post("/api/chat/stream")
@require_single_user_access("chat")
async def proxy_streaming_chat(request: Request, session: dict = Depends(get_current_session)):
    """Protected proxy streaming chat requests to backend with SSE support"""
    try:
        data = await request.json()
        
        async def stream_proxy():
            async with httpx.AsyncClient() as client:
                headers = {"x-session-id": session["session_id"]}
                
                async with client.stream(
                    "POST",
                    f"{backend_url}/api/chat/stream",
                    json=data,
                    headers=headers,
                    timeout=300.0
                ) as response:
                    if response.status_code != 200:
                        yield {
                            "data": json.dumps({
                                "type": "error",
                                "content": f"Backend error: HTTP {response.status_code}"
                            })
                        }
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield {"data": line[6:]}
                        elif line.strip():
                            yield {"data": line}
        
        return EventSourceResponse(stream_proxy())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming proxy error: {str(e)}")

@app.post("/api/chat/stream-text")
@require_single_user_access("chat")
async def proxy_text_streaming_chat(request: Request, session: dict = Depends(get_current_session)):
    """Protected proxy text streaming chat requests to backend"""
    try:
        data = await request.json()
        
        async def text_stream_proxy():
            async with httpx.AsyncClient() as client:
                headers = {"x-session-id": session["session_id"]}
                
                async with client.stream(
                    "POST",
                    f"{backend_url}/api/chat/stream-text",
                    json=data,
                    headers=headers,
                    timeout=300.0
                ) as response:
                    if response.status_code != 200:
                        yield f"[ERROR: Backend returned HTTP {response.status_code}]"
                        return
                    
                    async for chunk in response.aiter_text():
                        yield chunk
        
        return StreamingResponse(
            text_stream_proxy(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Streaming timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming proxy error: {str(e)}")

# Protected proxy endpoints for backend communication
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@require_single_user_access("api")
async def proxy_to_backend(path: str, request: Request, session: dict = Depends(get_current_session)):
    """Protected proxy requests to backend with session authentication"""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{backend_url}/api/{path}"
            headers = {"x-session-id": session["session_id"]}
            
            # Get request data
            if request.method in ["POST", "PUT"]:
                try:
                    data = await request.json()
                except:
                    data = None
            else:
                data = None
            
            # Forward request to backend with extended timeout for complex operations
            timeout = 120.0 if path.startswith(('dual-search', 'chat', 'export')) else 30.0
            
            response = await client.request(
                method=request.method,
                url=url,
                json=data,
                params=dict(request.query_params),
                headers=headers,
                timeout=timeout
            )
            
            # Return response with proper content type
            if response.headers.get('content-type', '').startswith('application/json'):
                return response.json()
            else:
                return JSONResponse(
                    content={"detail": "Non-JSON response from backend"},
                    status_code=response.status_code
                )
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Backend timeout for {path}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Backend unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/backend-status")
@require_single_user_access("status")
async def get_backend_status(request: Request, session: dict = Depends(get_current_session)):
    """Protected backend status endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            headers = {"x-session-id": session["session_id"]}
            
            # Get health status
            health_response = await client.get(f"{backend_url}/health", headers=headers, timeout=10.0)
            
            # Get configuration
            config_response = await client.get(f"{backend_url}/api/config", headers=headers, timeout=10.0)
            
            # Get collections
            collections_response = await client.get(f"{backend_url}/api/collections", headers=headers, timeout=10.0)
            
            result = {
                "backend_url": backend_url,
                "timestamp": time.time(),
                "session_info": session
            }
            
            if health_response.status_code == 200:
                result["health"] = health_response.json()
            else:
                result["health"] = {"error": f"HTTP {health_response.status_code}"}
            
            if config_response.status_code == 200:
                result["config"] = config_response.json()
            else:
                result["config"] = {"error": f"HTTP {config_response.status_code}"}
            
            if collections_response.status_code == 200:
                result["collections"] = collections_response.json()
            else:
                result["collections"] = {"error": f"HTTP {collections_response.status_code}"}
            
            return result
    
    except Exception as e:
        return {
            "error": str(e),
            "backend_url": backend_url,
            "timestamp": time.time(),
            "session_info": session
        }

# Debug and monitoring endpoints with protection
@app.get("/debug/processes")
@require_single_user_access("debug")
async def debug_processes(request: Request, session: dict = Depends(get_current_session)):
    """Protected debug endpoint to check process status"""
    return {
        "streamlit": {
            "process_exists": streamlit_process is not None,
            "is_running": streamlit_process is not None and streamlit_process.poll() is None,
            "pid": streamlit_process.pid if streamlit_process else None,
            "return_code": streamlit_process.poll() if streamlit_process else None
        },
        "chainlit": {
            "process_exists": chainlit_process is not None,
            "is_running": chainlit_process is not None and chainlit_process.poll() is None,
            "pid": chainlit_process.pid if chainlit_process else None,
            "return_code": chainlit_process.poll() if chainlit_process else None
        },
        "session_info": session
    }

@app.get("/config")
async def get_frontend_config():
    """Get frontend configuration (public endpoint)"""
    # Get session status without requiring authentication
    manager = get_single_user_manager()
    session_status = manager.get_session_status()
    
    return {
        "backend_url": backend_url,
        "frontend_version": "2.2.0",
        "access_control": {
            "enabled": True,
            "mode": "single_user",
            "session_timeout_minutes": 30,
            "session_active": session_status.get("active", False)
        },
        "services": {
            "streamlit_port": 8501,
            "chainlit_port": 8002,
            "frontend_port": 8000
        },
        "features": {
            "dual_search": True,
            "image_processing": True,
            "blob_storage": True,
            "docx_export": True,
            "session_management": True,
            "single_user_access": True
        },
        "timeouts": {
            "dual_search": 120,
            "chat_generation": 180,
            "docx_export": 120,
            "blob_operations": 60,
            "health_check": 5
        }
    }

# Metrics and monitoring with protection
@app.get("/metrics")
@require_single_user_access("metrics")
async def get_metrics(request: Request, session: dict = Depends(get_current_session)):
    """Protected metrics endpoint"""
    manager = get_single_user_manager()
    session_status = manager.get_session_status()
    
    return {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - start_time,
        "processes": {
            "streamlit_running": streamlit_process is not None and streamlit_process.poll() is None,
            "chainlit_running": chainlit_process is not None and chainlit_process.poll() is None
        },
        "backend_url": backend_url,
        "version": "2.2.0",
        "access_control": {
            "enabled": True,
            "current_session": session,
            "session_status": session_status
        }
    }

if __name__ == "__main__":
    print("Starting RAG Frontend Server with Single User Access Control...")
    print(f"Backend URL: {backend_url}")
    print("Services will be available at:")
    print("  - Frontend API: http://localhost:8000")
    print("  - Streamlit (Ingestion): http://localhost:8501 [PROTECTED]")
    print("  - Chainlit (Chat): http://localhost:8002 [PROTECTED]")
    print("")
    print("🔐 SINGLE USER MODE ENABLED")
    print("Only one user can access the system at a time.")
    print("Session timeout: 30 minutes")
    
    uvicorn.run(
        "frontend_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )