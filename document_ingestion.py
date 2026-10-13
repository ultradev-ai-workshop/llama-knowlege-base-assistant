"""Shared document loading, chunking, embedding and PGVector indexing utilities."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from urllib.parse import quote
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, text

load_dotenv()
logger = logging.getLogger("rag.ingestion")
BATCH_SIZE = 100
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".pdf", ".doc", ".docx",
    ".csv", ".xls", ".xlsx",
}


def source_link_for(path: Path, data_dir: Path) -> str:
    base_url = os.getenv("SOURCE_BASE_URL", "").strip().rstrip("/")
    try:
        relative = path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        relative = path.name
    if base_url:
        return f"{base_url}/" + "/".join(quote(part) for part in relative.split("/"))
    return path.resolve().as_uri()


def _load_spreadsheet(path: Path) -> list[Document]:
    """Load each workbook sheet as a document while preserving its sheet name."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=str).items()
    documents: list[Document] = []
    for sheet_name, frame in sheets:
        frame = frame.fillna("")
        if frame.empty:
            continue
        # Include column names and row values in a readable, searchable form.
        content = frame.to_csv(index=False, lineterminator="\n")
        documents.append(Document(page_content=content, metadata={"sheet_name": str(sheet_name)}))
    return documents


def load_one_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext or '(none)'}")

    if ext in {".txt", ".md", ".markdown"}:
        docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    elif ext == ".pdf":
        docs = PyPDFLoader(str(path)).load()  # One document per PDF page
    elif ext == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif ext == ".doc":
        # Legacy .doc support depends on the system's document-conversion tools.
        docs = UnstructuredWordDocumentLoader(str(path)).load()
    elif ext == ".csv":
        docs = CSVLoader(str(path), autodetect_encoding=True).load()
    elif ext in {".xls", ".xlsx"}:
        docs = _load_spreadsheet(path)
    else:  # guarded above
        raise ValueError(f"Unsupported file extension: {ext}")

    for doc in docs:
        doc.metadata["file_name"] = path.name
        doc.metadata["file_type"] = ext.lstrip(".")
    return docs


def _old_ids_for_source(connection: str, collection: str, source_path: str) -> list[str]:
    """Find previously indexed IDs for this source in the configured collection."""
    engine = create_engine(connection)
    try:
        query = text("""
            SELECT e.id
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = :collection
              AND e.cmetadata ->> 'source_path' = :source_path
        """)
        with engine.connect() as conn:
            return [str(row[0]) for row in conn.execute(query, {
                "collection": collection, "source_path": source_path
            }).fetchall()]
    finally:
        engine.dispose()


def index_paths(
    paths: Iterable[Path], *, data_dir: Path, connection: str,
    ollama_url: str, embedding_model: str, collection: str,
) -> dict[str, int]:
    """Index the supplied files. Re-indexing a file replaces its old chunks."""
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    store = PGVector(
        embeddings=embeddings, collection_name=collection,
        connection=connection, use_jsonb=True,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    results: dict[str, int] = {}

    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = path.relative_to(data_dir.resolve()).as_posix() if path.is_relative_to(data_dir.resolve()) else path.name
        try:
            documents = load_one_file(path)
            for doc in documents:
                doc.metadata.update({
                    "file_name": path.name,
                    "source_path": relative,
                    "source_link": source_link_for(path, data_dir),
                })
            chunks = splitter.split_documents(documents)
            if not chunks:
                logger.warning("No text extracted from %s", path)
                results[str(path)] = 0
                continue

            ids: list[str] = []
            for chunk in chunks:
                # Page/sheet/row metadata and start index keep chunk IDs deterministic.
                identity = (
                    f"{relative}:{chunk.metadata.get('page', '')}:"
                    f"{chunk.metadata.get('sheet_name', '')}:{chunk.metadata.get('row', '')}:"
                    f"{chunk.metadata.get('start_index', 0)}:"
                    f"{hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()}"
                )
                ids.append(hashlib.sha256(identity.encode("utf-8")).hexdigest())

            old_ids = _old_ids_for_source(connection, collection, relative)
            for offset in range(0, len(chunks), BATCH_SIZE):
                store.add_documents(
                    documents=chunks[offset:offset + BATCH_SIZE],
                    ids=ids[offset:offset + BATCH_SIZE],
                )
            # New content is stored first; then remove obsolete chunks for this source.
            obsolete_ids = list(set(old_ids) - set(ids))
            if obsolete_ids:
                store.delete(ids=obsolete_ids)
            results[str(path)] = len(chunks)
            logger.info("Indexed %s: %d chunks (%d obsolete chunks removed)", path, len(chunks), len(obsolete_ids))
        except Exception:
            logger.exception("Failed to ingest %s", path)
            raise
    return results


def iter_supported_files(data_dir: Path) -> list[Path]:
    return sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(data_dir).parts)
    )
