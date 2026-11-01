# syntax=docker/dockerfile:1
#
# FitNotes Coach — one container running `python server.py`:
#   API (127.0.0.1:8000) + UI / same-origin proxy (:3000) + MCP tool server.
# Only :3000 is meant to be reached, and only through Caddy (docker-compose.yml).
#
# Multi-arch: builds natively on linux/arm64 (Oracle Ampere A1) and linux/amd64.
# Python 3.12 has wheels for every dependency on both, torch included.
#
#   docker compose build                     # normal path, see DEPLOY.md
#   docker build -t fitnotes-coach .         # image only
#   docker buildx build --platform linux/arm64,linux/amd64 -t you/fitnotes-coach .

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Embedding / reranker models are baked into the image here.
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# 1) CPU-only PyTorch first, so sentence-transformers does not pull the CUDA
#    build (several GB of unused GPU libraries). The CPU index has both
#    manylinux aarch64 and x86_64 wheels. Override only if that host is
#    unreachable from your build machine: --build-arg TORCH_INDEX_URL=https://pypi.org/simple
#    works, but PyPI's torch >= 2.10 pulls the CUDA libraries on Linux (both
#    arches); on arm64, TORCH_SPEC=torch==2.9.1 from PyPI is CPU-only.
#    TORCH_SPEC pins a version if you need one, e.g. --build-arg TORCH_SPEC=torch==2.9.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch
RUN pip install --index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}"

# 2) Everything else. torch is already satisfied, so it is not replaced.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Download the sentence-transformers models at BUILD time, so a cold start
#    never waits on (or depends on) huggingface.co. Same names as src/rag.py
#    and src/memory.py.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 4) Build the RAG index from the tracked corpus/raw/*.json — offline: no
#    PubMed / Wikipedia calls. It lives outside data/ (the volume) and is
#    copied in on first start by deploy/prepare_data.py.
COPY corpus/ corpus/
COPY scripts/build_corpus.py scripts/build_corpus.py
RUN CHROMA_DB_PATH=/app/seed/chroma_db python scripts/build_corpus.py --offline \
    && test -f /app/seed/chroma_db/chroma.sqlite3

# 5) The application.
COPY . .

# Run as an unprivileged user. data/ is the named volume's mount point; Docker
# copies this ownership into a fresh volume. ontology/ is writable because the
# app appends newly seen exercises to its review queue there.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data /app/ontology /app/seed \
    && chmod +x deploy/entrypoint.sh

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    # Container defaults (docker-compose.yml sets them too).
    OPEN_BROWSER=0 \
    BACKEND_HOST=127.0.0.1 \
    FRONTEND_PROXY_TIMEOUT=900

USER app
VOLUME ["/app/data"]
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/status', timeout=8)"

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
