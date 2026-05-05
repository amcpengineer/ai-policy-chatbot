import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List

# Path where ChromaDB persists its data locally
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"

# Lightweight embedding model — runs fully local, no API key needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default collection name — can be overridden per instance
DEFAULT_COLLECTION = "ai_policies"


class VectorStore:
    def __init__(self, collection_name: str = DEFAULT_COLLECTION):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )

    def add_chunks(
        self,
        chunks: List[str],
        metadatas: List[dict],
        ids: List[str],
    ) -> None:
        """
        Embeds and stores document chunks into ChromaDB.

        Args:
            chunks:    List of text chunks to embed and store.
            metadatas: List of dicts with source info (doc name, page number).
            ids:       Unique string ID for each chunk.
        """
        embeddings = self.embedding_model.encode(chunks).tolist()
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query: str, n_results: int = 3) -> List[dict]:
        query_embedding = self.embedding_model.encode([query]).tolist()

        # Stage 1: search ONLY synthetic questions — they are semantic bridges
        syn_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            where={"type": "synthetic_question"},
            include=["documents", "metadatas", "distances"],
        )

        # Collect parent section IDs from matching synthetic questions
        parent_ids = set()
        if syn_results["documents"][0]:
            for meta, distance in zip(syn_results["metadatas"][0], syn_results["distances"][0]):
                if (1 - distance) >= 0.5:  # only strong synthetic matches
                    parent_id = meta.get("parent_id", "")
                    if parent_id:
                        parent_ids.add(parent_id)

        # Stage 2: search ALL chunk types for broad retrieval
        all_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results + 10,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for doc, meta, distance in zip(
                all_results["documents"][0],
                all_results["metadatas"][0],
                all_results["distances"][0],
        ):
            chunk_type = meta.get("type", "chunk")

            # Skip synthetic questions from final context
            if chunk_type == "synthetic_question":
                continue

            score = round(1 - distance, 4)

            # Boost sections that were pointed to by synthetic question matches
            chunk_id = meta.get("chunk_id", "")
            if chunk_id in parent_ids:
                score = min(1.0, score + 0.25)

            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "relevance_score": score,
                "type": chunk_type,
                "header": meta.get("header", ""),
                "part": meta.get("part", ""),
            })

        # Sort by boosted score, critical facts always first
        retrieved.sort(key=lambda x: (
            0 if x["type"] == "critical_fact" else 1,
            -x["relevance_score"]
        ))

        return retrieved[:n_results]

    def is_empty(self) -> bool:
        """Returns True if no documents have been ingested yet."""
        return self.collection.count() == 0

    def count(self) -> int:
        """Returns total number of chunks stored."""
        return self.collection.count()

    def reset(self) -> None:
        """Deletes all chunks from the collection. Use before re-ingesting."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

