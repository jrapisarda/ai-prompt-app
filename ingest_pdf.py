#!/usr/bin/env python
"""
Bulk-load a single PDF into your existing Chroma DB (./chroma_data) using the
*same* OpenAI embedding model as the Flask app.

USAGE
-----
pip install -r requirements.txt          # you already have these, but just FYI
python ingest_pdf.py \
    --pdf Freire_PedagogyoftheOppressed.pdf \
    --collection freire_pedagogy         # or my_collection if you prefer
"""
import argparse, datetime as dt, os, sys
from dotenv import load_dotenv

load_dotenv()

import pdfplumber, chromadb, tiktoken
from chromadb.utils import embedding_functions

# ── CLI ─────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--pdf",        required=True, help="PDF file to ingest")
ap.add_argument("--db-dir",     default="./chroma_data",
                help="Folder that already contains chroma.sqlite3")
ap.add_argument("--collection", default="freire_pedagogy",
                help="Collection name (new or existing)")
ap.add_argument("--max-tokens", type=int, default=400,
                help="Chunk size (tokens)")
ap.add_argument("--overlap",    type=int, default=50,
                help="Token overlap between chunks")
args = ap.parse_args()

# ── helper functions ───────────────────────────────────────────────────────
def pdf_to_text(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return " ".join(" ".join(p.split()) for p in pages)

def token_chunks(text: str, tokenizer, max_tokens=400, overlap=50):
    ids = tokenizer.encode(text)
    step = max_tokens - overlap
    for i in range(0, len(ids), step):
        yield tokenizer.decode(ids[i : i + max_tokens])

# ── 1. read & chunk ────────────────────────────────────────────────────────
tok = tiktoken.get_encoding("cl100k_base")      # matches text-embedding-3-small
raw = pdf_to_text(args.pdf)
chunks = list(token_chunks(raw, tok,
                           max_tokens=args.max_tokens,
                           overlap=args.overlap))
print(f"✅ Extracted {len(chunks)} chunks from {args.pdf}")

# ── 2. connect to the SAME database folder ─────────────────────────────────
client = chromadb.PersistentClient(path=args.db_dir)      # no legacy settings!

# ── 3. use the SAME embedding model as the Flask app ───────────────────────
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),       # already set for the webapp
    model_name="text-embedding-3-small"        # see app.py  :contentReference[oaicite:1]{index=1}
)

col = client.get_or_create_collection(args.collection, embedding_function=embed_fn)

# ── 4. write vectors ───────────────────────────────────────────────────────
ids = [f"{os.path.basename(args.pdf)}_chunk_{i}" for i in range(len(chunks))]
col.add(ids=ids,
        documents=chunks,
        metadatas=[{
            "source_pdf": os.path.basename(args.pdf),
            "ts": dt.datetime.utcnow().isoformat()
        } for _ in chunks])

print(f"🎉 Finished ingest: collection “{args.collection}” now has {col.count()} items.")
