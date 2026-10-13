# PostgreSQL + pgvector RAG Knowledge Base

A small, production-minded starter project using LangChain, PostgreSQL/pgvector, OpenAI embeddings, and Streamlit.

## Requirements

- Python 3.10+
- PostgreSQL with the `vector` extension enabled
- OpenAI API key
- PostgreSQL database and credentials with permission to create/use tables and indexes

## Project layout

```text
rag_knowledge_base/
├── app.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
└── data/
    └── example.txt
```

## Setup

1. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables.

   ```bash
   cp .env.example .env
   ```

   Set `OPENAI_API_KEY` and `POSTGRES_CONNECTION_STRING`. URL-encode reserved characters in the database username/password. Example format:

   ```text
   postgresql+psycopg://rag_user:password@localhost:5432/ragdb
   ```

4. Make sure the database exists and pgvector is enabled:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

   Run this as a database administrator if the application database user lacks permission. The extension must be installed on the PostgreSQL server.

5. Put `.txt` files in `data/` (nested folders are supported). An example file is included.

6. Ingest files:

   ```bash
   python ingest.py
   ```

7. Start the chat UI:

   ```bash
   streamlit run app.py
   ```

## Source links

By default, `source_link` uses a `file://` URI for each local file. Browsers often block local-file navigation from web apps, and `file://` links are not shareable with other users. For clickable, remotely accessible sources, set `SOURCE_BASE_URL` to a document-serving proxy/base URL that maps relative paths to accessible documents. This project does not expose local files over HTTP by itself.

## Operational notes

- The ingestion process batches writes and generates deterministic chunk IDs to make repeated ingestion of unchanged content idempotent.
- If a source file changes substantially, chunks at new offsets/content get new IDs; stale chunks from prior versions are not automatically deleted. For a strict mirror, add a reconciliation step that deletes prior IDs for each source before re-ingesting it.
- Keep `.env` out of source control. Use a secret manager in deployed environments.
- Restrict database network access, use TLS where appropriate, and grant the application database role only the permissions it needs.
- Add authentication/authorization in front of Streamlit before exposing it to users.
- Retrieved text is treated as untrusted input in the prompt. Prompt instructions reduce but do not eliminate prompt-injection risk.
- File URI links may not open in a browser; use a controlled proxy for production source access.
- Monitor OpenAI usage and set budgets/rate limits in your OpenAI account.
- Consider pinning exact dependency versions after testing in your target environment.

## Troubleshooting

- **No files indexed:** ensure files have `.txt` extensions under `data/` and `DATA_DIR` points to the correct folder.
- **Database connection fails:** verify database, credentials, network access, and the `postgresql+psycopg://` scheme.
- **Vector extension error:** ensure `CREATE EXTENSION vector` has been run in the target database.
- **Missing environment variables:** `.env.example` is a template; copy it to `.env` and fill in secrets.
