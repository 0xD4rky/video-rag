#!/bin/bash

# Video RAG Web Interface Startup Script

echo "Starting Video RAG Web Interface"

mkdir -p /Users/darky/Documents/video-rag/data/uploads
mkdir -p /Users/darky/Documents/video-rag/data/output
mkdir -p /Users/darky/Documents/video-rag/data/logs

echo "Checking main project dependencies"
pip install -r ../requirements.txt

echo "Starting Flask application"
python3 app.py
