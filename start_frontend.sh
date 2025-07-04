#!/bin/bash

# Start frontend services script

echo "Starting Frontend Services..."

# Create necessary directories
mkdir -p /app/public /app/temp /app/data

# Start the FastAPI server that manages Streamlit and Chainlit
echo "Starting FastAPI frontend server..."
python frontend_server.py