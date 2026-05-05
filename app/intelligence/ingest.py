"""
ingest.py — Hierarchical hybrid ingestion pipeline.

Supports both PDF and Markdown (.md) files.
For Markdown files, uses a 4-layer strategy:
    Layer 1: Header-based splits (## / ###) as primary chunks
    Layer 2: Synthetic Questions indexed as semantic anchors pointing to parent chunk
    Layer 3: Table rows as micro-chunks for precise numerical/factual lookup
    Layer 4: Critical Correct Answers as hard overrides (stored with type=critical_fact)

For PDF files, uses the original fixed-size chunking strategy.

Usage (from backend folder):
    # Ingest markdown into dbu_policy collection
    uv run python -m app.intelligence.ingest --collection dbu_policy --file DBU_Policy.md

    # Ingest all PDFs into ai_policies collection
    uv run python -m app.intelligence.ingest --collection ai_policies

    # Reset and re-ingest
    uv run python -m app.intelligence.ingest --collection dbu_policy --file DBU_Policy.md --reset
"""

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from app.intelligence.vector_store import VectorStore

DOCS_PATH     = Path(__file__).parent.parent / "docs"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# ── Critical facts — hard override layer ─────────────────────────────────────
# These are injected into the vector store with type=critical_fact.
# The RAG pipeline checks these first before returning any answer.
CRITICAL_FACTS = {
    "cairo reports to":          "The CAIRO reports directly to the Provost (Vice President for Academic Affairs), with a dotted-line relationship to the CIO. NOT the CIO.",
    "agentic ai permitted":      "Agentic AI is PROHIBITED on DBU WiFi and infrastructure without an approved CAIRO exception. It is NOT permitted.",
    "chatgpt allowed":           "ChatGPT in conversational mode is permitted at DBU. It is listed as an approved Level 3 tool. Autonomous use requires a CAIRO exception.",
    "red badge means":           "A red badge means sandbox access is suspended — it triggers a brief targeted review (estimated 20 minutes). It is NOT a permanent punishment.",
    "student rep votes":         "The student representative on the AI Governance Committee holds a formal VOTING seat — not an advisory role.",
    "algc certification valid":  "ALGC certification is valid for 2 years, plus a brief term-start review (10-15 minutes) at the start of each academic term.",
    "tier 3 approval":           "Tier 3 sandbox access requires approval from the CAIRO, CIO, Data Privacy Officer, and — if student-facing — the President's Cabinet.",
    "incident reporting time":   "Security incidents must be reported to the CAIRO office within ONE HOUR of detection.",
}


# ── PDF ingestion (original strategy with improved chunk size) ────────────────

def load_pdf(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text.strip(), "page": i + 1})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_pdf(path: Path, store: VectorStore, chunk_index: int, collection_name: str) -> int:
    print(f"\nProcessing PDF: {path.name}")
    pages = load_pdf(path)
    print(f"  Pages extracted: {len(pages)}")

    all_chunks, all_metadatas, all_ids = [], [], []

    for page_data in pages:
        chunks = chunk_text(page_data["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source":     path.name,
                "page":       page_data["page"],
                "collection": collection_name,
                "type":       "pdf_chunk",
            })
            all_ids.append(f"{collection_name}_chunk_{chunk_index}")
            chunk_index += 1

    _batch_store(store, all_chunks, all_metadatas, all_ids)
    return chunk_index


# ── Markdown ingestion (4-layer hierarchical strategy) ───────────────────────

def parse_markdown_sections(content: str) -> list[dict]:
    """
    Splits a markdown document on ## and ### headers.
    Returns a list of sections, each with:
        - header:    the header text
        - level:     2 (##) or 3 (###)
        - part:      parent ## section name
        - text:      full section text including header
        - tables:    list of table blocks found in the section
        - synthetic: list of synthetic questions found in the section
    """
    lines = content.split("\n")
    sections = []
    current_section = None
    current_part = ""

    for line in lines:
        h2_match = re.match(r'^## (.+)', line)
        h3_match = re.match(r'^### (.+)', line)

        if h2_match:
            if current_section:
                sections.append(current_section)
            current_part = h2_match.group(1).strip()
            current_section = {
                "header":    current_part,
                "level":     2,
                "part":      current_part,
                "text":      line + "\n",
                "tables":    [],
                "synthetic": [],
            }
        elif h3_match:
            if current_section:
                sections.append(current_section)
            header = h3_match.group(1).strip()
            current_section = {
                "header":    header,
                "level":     3,
                "part":      current_part,
                "text":      line + "\n",
                "tables":    [],
                "synthetic": [],
            }
        else:
            if current_section is not None:
                current_section["text"] += line + "\n"

    if current_section:
        sections.append(current_section)

    # Post-process: extract tables and synthetic questions from each section
    for section in sections:
        section["tables"]    = extract_tables(section["text"])
        section["synthetic"] = extract_synthetic_questions(section["text"])

    return sections


def extract_tables(text: str) -> list[list[str]]:
    """Extracts markdown tables as lists of row strings."""
    tables = []
    current_table = []
    in_table = False

    for line in text.split("\n"):
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                in_table = True
                current_table = []
            # Skip separator rows (|---|---|)
            if not re.match(r'^\|[-| :]+\|$', line.strip()):
                current_table.append(line.strip())
        else:
            if in_table and current_table:
                tables.append(current_table)
                current_table = []
            in_table = False

    if current_table:
        tables.append(current_table)

    return tables


def extract_synthetic_questions(text: str) -> list[str]:
    """
    Extracts synthetic questions from sections.
    Looks for blocks labeled 'Synthetic Questions', 'Sample Questions', or numbered/bulleted questions.
    """
    questions = []
    in_synthetic_block = False

    for line in text.split("\n"):
        line_lower = line.lower()
        if any(marker in line_lower for marker in ["synthetic questions", "sample questions", "test questions"]):
            in_synthetic_block = True
            continue

        if in_synthetic_block:
            # Stop at the next header or empty section
            if line.startswith("#"):
                in_synthetic_block = False
                continue
            # Extract question lines
            clean = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
            if clean and len(clean) > 10 and "?" in clean:
                questions.append(clean)

    return questions


def ingest_markdown(path: Path, store: VectorStore, chunk_index: int, collection_name: str) -> int:
    print(f"\nProcessing Markdown: {path.name}")
    content = path.read_text(encoding="utf-8")
    sections = parse_markdown_sections(content)
    print(f"  Sections found: {len(sections)}")

    all_chunks, all_metadatas, all_ids = [], [], []

    for section in sections:
        header  = section["header"]
        part    = section["part"]
        section_text = section["text"].strip()

        if not section_text:
            continue

        # ── Layer 1: Header-based section chunk ───────────────────────────
        all_chunks.append(section_text)
        section_id = f"{collection_name}_chunk_{chunk_index}"
        all_metadatas.append({
            "source": path.name,
            "page": 0,
            "collection": collection_name,
            "type": "section",
            "header": header,
            "part": part,
            "chunk_id": section_id,  # ← add this
        })
        all_ids.append(section_id)
        chunk_index += 1

        # ── Layer 2: Synthetic questions as semantic anchors ──────────────
        for question in section["synthetic"]:
            all_chunks.append(question)
            all_metadatas.append({
                "source":     path.name,
                "page":       0,
                "collection": collection_name,
                "type":       "synthetic_question",
                "header":     header,
                "part":       part,
                "parent_id":  section_id,
            })
            all_ids.append(f"{collection_name}_chunk_{chunk_index}")
            chunk_index += 1

        # ── Layer 3: Table rows as micro-chunks ───────────────────────────
        for table in section["tables"]:
            if len(table) < 2:
                continue
            header_row = table[0]
            for row in table[1:]:
                micro_chunk = f"{header} | {header_row} | {row}"
                all_chunks.append(micro_chunk)
                all_metadatas.append({
                    "source":     path.name,
                    "page":       0,
                    "collection": collection_name,
                    "type":       "table_row",
                    "header":     header,
                    "part":       part,
                })
                all_ids.append(f"{collection_name}_chunk_{chunk_index}")
                chunk_index += 1

    # ── Layer 4: Critical facts as hard overrides ─────────────────────────
    print(f"  Injecting {len(CRITICAL_FACTS)} critical facts...")
    for key, fact in CRITICAL_FACTS.items():
        all_chunks.append(fact)
        all_metadatas.append({
            "source":     "CRITICAL_FACTS",
            "page":       0,
            "collection": collection_name,
            "type":       "critical_fact",
            "key":        key,
        })
        all_ids.append(f"{collection_name}_critical_{key.replace(' ', '_')}")

    syn_count   = sum(len(s["synthetic"]) for s in sections)
    table_count = sum(len(t) - 1 for s in sections for t in s["tables"] if len(t) > 1)
    print(f"  Section chunks: {len(sections)}")
    print(f"  Synthetic question anchors: {syn_count}")
    print(f"  Table row micro-chunks: {table_count}")
    print(f"  Critical fact overrides: {len(CRITICAL_FACTS)}")

    _batch_store(store, all_chunks, all_metadatas, all_ids)
    return chunk_index


def _batch_store(
    store: VectorStore,
    chunks: list[str],
    metadatas: list[dict],
    ids: list[str],
    batch_size: int = 50,
) -> None:
    print(f"  Total chunks to embed: {len(chunks)}")
    print("  Embedding and storing...")
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        store.add_chunks(
            chunks=chunks[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"    Stored {i + 1} to {end}...")


# ── Main entry point ──────────────────────────────────────────────────────────

def ingest_documents(
    reset: bool = False,
    collection_name: str = "ai_policies",
    single_file: str | None = None,
) -> None:
    store = VectorStore(collection_name=collection_name)

    if reset:
        print(f"Resetting collection '{collection_name}'...")
        store.reset()
        print("Collection cleared.")

    if not store.is_empty():
        print(f"Collection '{collection_name}' already has {store.count()} chunks.")
        print("Use --reset flag to re-ingest from scratch.")
        return

    if not DOCS_PATH.exists():
        print(f"Docs folder not found at: {DOCS_PATH}")
        sys.exit(1)

    if single_file:
        file_path = DOCS_PATH / single_file
        if not file_path.exists():
            print(f"File not found: {file_path}")
            sys.exit(1)
        files = [file_path]
    else:
        files = list(DOCS_PATH.glob("*.pdf")) + list(DOCS_PATH.glob("*.md"))

    if not files:
        print(f"No PDF or Markdown files found in {DOCS_PATH}")
        sys.exit(1)

    print(f"Collection: '{collection_name}'")
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  - {f.name}")

    chunk_index = 0

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            chunk_index = ingest_pdf(file_path, store, chunk_index, collection_name)
        elif suffix == ".md":
            chunk_index = ingest_markdown(file_path, store, chunk_index, collection_name)
        else:
            print(f"  Skipping unsupported file type: {file_path.name}")

    print(f"\nDone. {store.count()} total chunks stored in '{collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest AI policy documents into ChromaDB.")
    parser.add_argument("--reset",      action="store_true", help="Clear collection before ingesting.")
    parser.add_argument("--collection", type=str, default="ai_policies", help="Collection name.")
    parser.add_argument("--file",       type=str, default=None, help="Single file to ingest (PDF or .md).")
    args = parser.parse_args()
    ingest_documents(reset=args.reset, collection_name=args.collection, single_file=args.file)