# Week 9: containerized API service. Only what api.py's serving path needs
# (see requirements-api.txt) -- eval tooling and the one-time CMS extraction
# script have their own dependencies and don't belong in this image.

FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
# CPU-only torch first -- otherwise sentence-transformers pulls the default
# CUDA build (no GPU on Cloud Run or most free-tier hosts), which alone
# bloats the image past 8GB with nvidia-* packages nothing here ever uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake the cross-encoder reranker model into the image so a cold container
# start doesn't also pay for a ~90MB download from Hugging Face on the
# first request.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src/ ./src/
COPY data/real_docs/ ./data/real_docs/

WORKDIR /app/src

ENV EMBED_PROVIDER=openai \
    LLM_PROVIDER=anthropic \
    RETRIEVAL_MODE=hybrid \
    VECTOR_STORE=pgvector \
    RAG_DATA_DIR=real_docs \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
