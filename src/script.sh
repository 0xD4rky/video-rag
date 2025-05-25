#!/bin/bash

VIDEO_PATH="/Users/darky/Documents/video-rag/data/test_1.mp4"
QUERY="a man drinking water from glass"

python3 embeddings.py <<EOF
$VIDEO_PATH
$QUERY
EOF 
