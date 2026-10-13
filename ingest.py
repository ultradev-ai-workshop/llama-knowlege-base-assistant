"""Ingest all supported files under DATA_DIR into PostgreSQL/pgvector."""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from document_ingestion import iter_supported_files, index_paths

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("rag.ingest")


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def main() -> int:
    try:
        connection = required_env("POSTGRES_CONNECTION_STRING")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.200:11434").strip().rstrip("/")
        embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
        collection = os.getenv("PGVECTOR_COLLECTION_NAME", "knowledge_base_ollama").strip()
        data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        paths = iter_supported_files(data_dir)
        if not paths:
            logger.warning("No supported documents found under %s. Supported: txt, md, pdf, doc, docx, csv, xls, xlsx", data_dir)
            return 0
        results = index_paths(
            paths, data_dir=data_dir, connection=connection,
            ollama_url=ollama_url, embedding_model=embedding_model,
            collection=collection,
        )
        total_chunks = sum(results.values())
        logger.info("Processed %d files; indexed %d chunks into collection %s", len(results), total_chunks, collection)
        return 0
    except Exception:
        logger.exception("Ingestion failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
