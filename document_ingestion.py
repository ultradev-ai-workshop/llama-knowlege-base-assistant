"""Shared document loading, chunking, embedding and PGVector indexing utilities."""
from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import socket
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, urlunparse
from typing import Iterable

import requests
from bs4 import BeautifulSoup

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
MAX_URL_BYTES = 2_000_000
URL_TIMEOUT_SECONDS = 15
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


def _validated_public_url(raw_url: str) -> str:
    """Normalize a public HTTP(S) URL and reject local/private destinations."""
    parsed = urlparse(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not supported.")

    host = parsed.hostname.rstrip(".").lower()
    # Optional allowlist: comma-separated hostnames. Subdomains are allowed.
    allowed = [item.strip().lower().lstrip(".") for item in os.getenv("URL_ALLOWED_DOMAINS", "").split(",") if item.strip()]
    if allowed and not any(host == domain or host.endswith("." + domain) for domain in allowed):
        raise ValueError(f"The domain {host!r} is not in URL_ALLOWED_DOMAINS.")

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve URL host {host!r}.") from exc
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Only publicly reachable web pages can be indexed.")

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", parsed.query, ""))


def load_web_page(raw_url: str) -> tuple[str, str]:
    """Fetch a bounded HTML page and return its title and readable text."""
    url = _validated_public_url(raw_url)
    headers = {"User-Agent": "KnowledgeBaseAssistant/1.0 (+URL ingestion)"}
    for _ in range(6):
        with requests.get(url, headers=headers, timeout=URL_TIMEOUT_SECONDS, stream=True, allow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("The page redirected without a destination.")
                url = _validated_public_url(urljoin(url, location))
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise ValueError("The URL must point to an HTML page.")
            data = bytearray()
            for block in response.iter_content(64 * 1024):
                data.extend(block)
                if len(data) > MAX_URL_BYTES:
                    raise ValueError("The page is larger than the 2 MB indexing limit.")
            encoding = response.encoding or "utf-8"
            html = bytes(data).decode(encoding, errors="replace")
            break
    else:
        raise ValueError("The page redirected too many times.")

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else urlparse(url).netloc
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        element.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    content = main.get_text("\n", strip=True)
    if not content:
        raise ValueError("No readable page text was found.")
    return title[:300], content


def index_web_urls(
    urls: Iterable[str], *, connection: str, ollama_url: str,
    embedding_model: str, collection: str,
) -> dict[str, int]:
    """Fetch and index public HTML pages, replacing old chunks for each URL."""
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_url)
    store = PGVector(embeddings=embeddings, collection_name=collection, connection=connection, use_jsonb=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    results: dict[str, int] = {}

    for raw_url in urls:
        url = _validated_public_url(raw_url)
        title, content = load_web_page(url)
        source_id = f"url:{url}"
        docs = [Document(page_content=content, metadata={
            "file_name": title, "file_type": "html", "source_path": source_id,
            "source_link": url, "source_type": "url",
        })]
        chunks = splitter.split_documents(docs)
        ids = [hashlib.sha256((f"{source_id}:{chunk.metadata.get('start_index', 0)}:" + hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()).encode("utf-8")).hexdigest() for chunk in chunks]
        old_ids = _old_ids_for_source(connection, collection, source_id)
        for offset in range(0, len(chunks), BATCH_SIZE):
            store.add_documents(documents=chunks[offset:offset + BATCH_SIZE], ids=ids[offset:offset + BATCH_SIZE])
        obsolete_ids = list(set(old_ids) - set(ids))
        if obsolete_ids:
            store.delete(ids=obsolete_ids)
        results[url] = len(chunks)
        logger.info("Indexed URL %s: %d chunks (%d obsolete removed)", url, len(chunks), len(obsolete_ids))
    return results


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
