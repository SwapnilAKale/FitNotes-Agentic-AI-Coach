#!/bin/sh
# Container entrypoint: prepare the data volume, then run the app exactly as a
# developer would (`python server.py` → API on 127.0.0.1:8000, which starts the
# UI + same-origin proxy on :3000 and the MCP tool server).
set -e
cd /app
python deploy/prepare_data.py
exec python server.py "$@"
