#!/bin/bash
set -e
VIDEO="${1:-}"
QUERY="${2:-}"
if [ -z "$VIDEO" ]; then
  read -r -p "Enter video path: " VIDEO
fi
if [ -z "$QUERY" ]; then
  read -r -p "Enter query: " QUERY
fi
python -m agent --video "$VIDEO" --query "$QUERY" "${@:3}"