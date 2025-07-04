#!/bin/bash

# Start backend services script - Modified for external Qdrant

echo "Starting Backend Services (using external Qdrant)..."

# Create necessary directories
mkdir -p /app/data /app/temp

# Start the FastAPI backend server
echo "Starting FastAPI backend server..."
python backend_server.py