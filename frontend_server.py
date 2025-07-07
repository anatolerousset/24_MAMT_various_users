"""
Frontend FastAPI server to serve Streamlit and Chainlit applications
with full integration to the backend API
"""
import os
import subprocess
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
import json
from typing import Dict, Any, List
from sse_starlette.sse import EventSourceResponse

# Global variables to track subprocess
streamlit_process = None
chainlit_process = None
backend_url = os.getenv("BACKEND_URL", "http://backend:8001")
start_time = time.time()

def start_streamlit():
    """Start Streamlit application"""
    global streamlit_process
    try:
        print("Starting Streamlit...")
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
        print("Starting Chainlit...")
        # Set backend URL for Chainlit
        env = os.environ.copy()
        env["BACKEND_URL"] = backend_url
        
        chainlit_process = subprocess.Popen([
            "chainlit", "run", "chainlit_frontend.py",
            "--port", "8502",
            "--host", "0.0.0.0"
        ], env=env)
        print("Chainlit started successfully")
    except Exception as e:
        print(f"Error starting Chainlit: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting frontend applications...")
    
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
    print("Frontend applications started")
    
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

app = FastAPI(title="RAG Frontend Server", version="2.0.0", lifespan=lifespan)

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
    """Root endpoint - service selection"""
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
    
    return {
        "message": "RAG Frontend Server",
        "version": "2.0.0",
        "backend_status": backend_status,
        "backend_url": backend_url,
        "services": {
            "ingestion": {
                "url": "http://localhost:8501",
                "description": "Streamlit interface for document ingestion"
            },
            "chat": {
                "url": "http://localhost:8502", 
                "description": "Chainlit chat interface with full RAG features"
            },
            "health": {
                "url": "http://localhost:8000/health",
                "description": "Frontend health status"
            },
            "backend_health": {
                "url": f"{backend_url}/health",
                "description": "Backend health and configuration"
            }
        },
        "features": [
            "Dual collection search",
            "Image processing integration",
            "Azure Blob Storage exports",
            "Complete DOCX document export",
            "Session management",
            "Real-time LLM response generation"
        ],
        "backend_info": backend_info
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
            response = await client.get("http://localhost:8502", timeout=5.0)
            services_status["chainlit"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services_status["chainlit"] = "unhealthy"
    
    # Check Backend with detailed status
    backend_detailed_status = {}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{backend_url}/health", timeout=5.0)
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
    return RedirectResponse(url="http://localhost:8502")

@app.post("/api/chat/stream")
async def proxy_streaming_chat(request: Request):
    """Proxy streaming chat requests to backend with SSE support"""
    try:
        data = await request.json()
        
        async def stream_proxy():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{backend_url}/api/chat/stream",
                    json=data,
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
                            # Forward the SSE data
                            yield {"data": line[6:]}  # Remove "data: " prefix
                        elif line.strip():
                            # Handle other SSE lines
                            yield {"data": line}
        
        return EventSourceResponse(stream_proxy())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming proxy error: {str(e)}")

@app.post("/api/chat/stream-text")
async def proxy_text_streaming_chat(request: Request):
    """Proxy text streaming chat requests to backend"""
    try:
        data = await request.json()
        
        async def text_stream_proxy():
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{backend_url}/api/chat/stream-text",
                    json=data,
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

@app.get("/backend-status")
async def get_backend_status():
    """Get detailed backend status and configuration"""
    try:
        async with httpx.AsyncClient() as client:
            # Get health status
            health_response = await client.get(f"{backend_url}/health", timeout=10.0)
            
            # Get configuration
            config_response = await client.get(f"{backend_url}/api/config", timeout=10.0)
            
            # Get collections
            collections_response = await client.get(f"{backend_url}/api/collections", timeout=10.0)
            
            result = {
                "backend_url": backend_url,
                "timestamp": time.time()
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
            "timestamp": time.time()
        }

# Proxy endpoints for backend communication
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_backend(path: str, request: Request):
    """Proxy requests to backend with better error handling"""
    try:
        async with httpx.AsyncClient() as client:
            url = f"{backend_url}/api/{path}"
            
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


@app.post("/api/dual-search")
async def proxy_dual_search(request: Request):
    """Proxy dual search requests with error handling"""
    try:
        data = await request.json()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend_url}/api/dual-search",
                json=data,
                timeout=120.0  # Extended timeout for image processing
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Dual search failed"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=error_detail)
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Dual search timeout - operation may be processing large images")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dual search error: {str(e)}")

@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy chat requests with LLM timeout handling"""
    try:
        data = await request.json()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend_url}/api/chat",
                json=data,
                timeout=180.0  # Extended timeout for LLM generation
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Chat generation failed"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=error_detail)
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM generation timeout - the model may be processing a complex request")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/export/docx")
async def proxy_docx_export(request: Request):
    """Proxy DOCX export requests"""
    try:
        data = await request.json()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend_url}/api/export/docx",
                json=data,
                timeout=120.0  # Extended timeout for document generation
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "DOCX export failed"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=error_detail)
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DOCX export timeout - document generation taking longer than expected")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX export error: {str(e)}")

@app.post("/api/blob/export")
async def proxy_blob_export(request: Request):
    """Proxy blob storage export requests"""
    try:
        data = await request.json()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{backend_url}/api/blob/export",
                json=data,
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Blob export failed"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=error_detail)
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Blob export timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blob export error: {str(e)}")

@app.get("/api/blob/list")
async def proxy_blob_list():
    """Proxy blob storage list requests"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{backend_url}/api/blob/list",
                timeout=30.0
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = "Blob list failed"
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", error_detail)
                except:
                    pass
                raise HTTPException(status_code=response.status_code, detail=error_detail)
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Blob list timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blob list error: {str(e)}")

# Debug and monitoring endpoints
@app.get("/debug/processes")
async def debug_processes():
    """Debug endpoint to check process status"""
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
        }
    }

@app.post("/debug/restart-streamlit")
async def restart_streamlit():
    """Debug endpoint to restart Streamlit"""
    global streamlit_process
    
    try:
        # Stop existing process
        if streamlit_process and streamlit_process.poll() is None:
            streamlit_process.terminate()
            streamlit_process.wait()
        
        # Start new process
        start_streamlit()
        await asyncio.sleep(3)  # Wait for startup
        
        return {
            "success": True,
            "message": "Streamlit restarted",
            "new_pid": streamlit_process.pid if streamlit_process else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/debug/restart-chainlit")
async def restart_chainlit():
    """Debug endpoint to restart Chainlit"""
    global chainlit_process
    
    try:
        # Stop existing process
        if chainlit_process and chainlit_process.poll() is None:
            chainlit_process.terminate()
            chainlit_process.wait()
        
        # Start new process
        start_chainlit()
        await asyncio.sleep(5)  # Wait for startup
        
        return {
            "success": True,
            "message": "Chainlit restarted",
            "new_pid": chainlit_process.pid if chainlit_process else None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Configuration endpoints
@app.get("/config")
async def get_frontend_config():
    """Get frontend configuration"""
    return {
        "backend_url": backend_url,
        "frontend_version": "2.0.0",
        "services": {
            "streamlit_port": 8501,
            "chainlit_port": 8502,
            "frontend_port": 8000
        },
        "features": {
            "dual_search": True,
            "image_processing": True,
            "blob_storage": True,
            "docx_export": True,
            "session_management": True
        },
        "timeouts": {
            "dual_search": 120,
            "chat_generation": 180,
            "docx_export": 120,
            "blob_operations": 60,
            "health_check": 5
        }
    }

@app.put("/config/backend-url")
async def update_backend_url(request: Request):
    """Update backend URL (for debugging)"""
    global backend_url
    
    try:
        data = await request.json()
        new_url = data.get("backend_url")
        
        if not new_url:
            raise HTTPException(status_code=400, detail="backend_url is required")
        
        old_url = backend_url
        backend_url = new_url
        
        # Test the new URL
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{backend_url}/health", timeout=5.0)
                if response.status_code != 200:
                    backend_url = old_url  # Rollback
                    raise HTTPException(status_code=400, detail="New backend URL is not responding correctly")
        except httpx.RequestError:
            backend_url = old_url  # Rollback
            raise HTTPException(status_code=400, detail="Cannot connect to new backend URL")
        
        return {
            "success": True,
            "old_url": old_url,
            "new_url": backend_url,
            "message": "Backend URL updated successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration update error: {str(e)}")

# Metrics and monitoring
@app.get("/metrics")
async def get_metrics():
    """Get basic metrics about the frontend server"""
    return {
        "timestamp": time.time(),
        "uptime_seconds": time.time() - start_time,
        "processes": {
            "streamlit_running": streamlit_process is not None and streamlit_process.poll() is None,
            "chainlit_running": chainlit_process is not None and chainlit_process.poll() is None
        },
        "backend_url": backend_url,
        "version": "2.0.0"
    }

if __name__ == "__main__":
    print("Starting RAG Frontend Server...")
    print(f"Backend URL: {backend_url}")
    print("Services will be available at:")
    print("  - Frontend API: http://localhost:8000")
    print("  - Streamlit (Ingestion): http://localhost:8501")
    print("  - Chainlit (Chat): http://localhost:8502")
    
    uvicorn.run(
        "frontend_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True
    )