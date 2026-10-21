#!/usr/bin/env bash
# Starts the Argus API and the Streamlit UI together. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .venv ]; then
  source .venv/bin/activate
fi

if [ ! -f .env ]; then
  echo "No .env found — copy .env.example to .env and add your API keys." >&2
  exit 1
fi

uvicorn api.server:app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

streamlit run ui/streamlit_app.py --server.port 8501
