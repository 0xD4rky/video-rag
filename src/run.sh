
#!/usr/bin/env bash
set -e

echo "Clearing previous data..."
rm -rf /Users/darky/Documents/video-rag/data/output/*
rm -rf /Users/darky/Documents/video-rag/data/logs/*
rm -rf /Users/darky/Documents/video-rag/data/faiss/*

echo "Creating directories..."
mkdir -p /Users/darky/Documents/video-rag/data/{faiss,output,logs}

echo "Running video search..."
python3 agent.py