
#!/usr/bin/env bash
set -e

read -p "Enter the video path: " VIDEO_PATH
read -p "Enter the information to be retrieved: " QUERY

TOP_N="${TOP_N:-3}"

echo "Clearing previous data..."
rm -rf /Users/darky/Documents/video-rag/data/output/*
rm -rf /Users/darky/Documents/video-rag/data/logs/*
rm -rf /Users/darky/Documents/video-rag/data/faiss/*

echo "Creating directories..."
mkdir -p /Users/darky/Documents/video-rag/data/{faiss,output,logs}

echo ""
echo "Running video search..."
echo "Video: $VIDEO_PATH"
echo "Query: $QUERY"
echo "Top N: $TOP_N"
echo ""

echo "Running video search..."
python3 agent.py "$VIDEO_PATH" "$QUERY" --top-n "$TOP_N"