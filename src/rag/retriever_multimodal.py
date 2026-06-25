import os
import json
import pickle
import hashlib
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from src.rag.embedder import get_embeddings

"""Multimodal variant of the starter hybrid retriever.

This version adds lightweight figure-priority behavior for visual queries,
while keeping the same FAISS+BM25 structure as the base retriever.
"""

load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
CACHE_DIR = "./data/cache"
BATCH_SIZE = 50

FIGURE_BOOST = 1.2  # Modest boost for figure chunks in a multimodal index


class Retriever:
    """
    Hybrid retriever combining FAISS (cosine similarity)
    and BM25 (keyword search) with Reciprocal Rank Fusion or union mode.
    Supports figure boosting for multimodal indexes.
    """

    def __init__(self):
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.bm25_retriever = None
        self.documents = []

    def _get_cache_dir(self, chunks: list[dict]) -> str:
        content = json.dumps([c["text"] for c in chunks], sort_keys=True)
        hash_key = hashlib.md5(content.encode()).hexdigest()[:8]
        return os.path.join(CACHE_DIR, f"index_{hash_key}")

    def build_index(self, chunks: list[dict], index_name: str = None) -> None:
        """
        Build hybrid index from chunks, with FAISS persistence.

        Args:
            chunks: List of chunk dicts with text and metadata.
            index_name: Optional name for the index folder.
                        If None, uses MD5 hash of chunk content.
        """
        if index_name:
            cache_dir = os.path.join(CACHE_DIR, index_name)
        else:
            cache_dir = self._get_cache_dir(chunks)

        faiss_path = os.path.join(cache_dir, "faiss_index")
        bm25_path = os.path.join(cache_dir, "bm25.pkl")

        if os.path.exists(faiss_path) and os.path.exists(bm25_path):
            logger.info(f"Loading cached index from {cache_dir}...")
            self.vectorstore = FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.COSINE
            )
            with open(bm25_path, "rb") as f:
                cached = pickle.load(f)
                self.documents = cached["documents"]
                self.bm25_retriever = cached["bm25_retriever"]
            logger.info("Cached index loaded successfully")
            return

        self.documents = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]

        logger.info(f"Building hybrid index for {len(self.documents)} chunks...")

        first_batch = self.documents[:BATCH_SIZE]
        self.vectorstore = FAISS.from_documents(
            first_batch,
            self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        logger.info(f"Embedded {min(BATCH_SIZE, len(self.documents))}/{len(self.documents)} chunks")

        for i in range(BATCH_SIZE, len(self.documents), BATCH_SIZE):
            batch = self.documents[i:i + BATCH_SIZE]
            self.vectorstore.add_documents(batch)
            logger.info(f"Embedded {min(i + BATCH_SIZE, len(self.documents))}/{len(self.documents)} chunks")

        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = TOP_K

        os.makedirs(cache_dir, exist_ok=True)
        self.vectorstore.save_local(faiss_path)
        with open(bm25_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "bm25_retriever": self.bm25_retriever
            }, f)
        logger.info(f"Index saved to {cache_dir}")
        logger.info("Hybrid index built successfully")

    def _rrf_search(self, query: str, top_k: int, bm25_results, embedding_results) -> list[dict]:
        """
        RRF fusion from BM25 and embedding results.
        Applies a modest boost to figure chunks in the multimodal index.
        """
        scores = {}

        for rank, doc in enumerate(bm25_results):
            key = doc.page_content
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1 / (rank + RRF_K)

        for rank, (doc, _) in enumerate(embedding_results):
            key = doc.page_content
            if key not in scores:
                scores[key] = {"doc": doc, "score": 0.0}
            scores[key]["score"] += 1 / (rank + RRF_K)

        # Apply a modest boost to figure chunks in multimodal retrieval.
        # This keeps figure metadata visible while preserving BM25/embedding relevance.
        for item in scores.values():
            if item["doc"].metadata.get("chunk_type") == "figure":
                item["score"] *= FIGURE_BOOST

        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [
            {
                "text": item["doc"].page_content,
                "metadata": item["doc"].metadata,
                "score": round(item["score"], 4)
            }
            for item in sorted_results[:top_k]
        ]

    def _union_search(self, top_k: int, bm25_results, embedding_results) -> list[dict]:
        """Union of BM25 and embedding results without fusion."""
        seen = set()
        combined = []

        for doc in bm25_results[:top_k]:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": None,
                    "source": "bm25"
                })

        for doc, score in embedding_results[:top_k]:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": round(float(score), 4),
                    "source": "embedding"
                })

        return combined

    def search(self, query: str, top_k: int = TOP_K, use_rrf: bool = True) -> dict:
        """
        Search using hybrid retrieval (BM25 + embeddings).

        Args:
            query: Search query text.
            top_k: Number of top results to return.
            use_rrf: If True uses RRF fusion, if False returns union.

        Returns:
            Dict with hybrid, bm25 and embedding results.
        """
        if not self.bm25_retriever or not self.vectorstore:
            logger.error("Index not built — call build_index first")
            return {}

        bm25_results = self.bm25_retriever.invoke(query)
        bm25_chunks = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "rank": rank,
                "score": round(1 / (rank + RRF_K), 4)
            }
            for rank, doc in enumerate(bm25_results[:top_k])
        ]

        embedding_results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        embedding_chunks = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "rank": rank,
                "score": round(float(cosine_score), 4)
            }
            for rank, (doc, cosine_score) in enumerate(embedding_results)
        ]

        if use_rrf:
            hybrid_chunks = self._rrf_search(query, top_k, bm25_results, embedding_results)
            mode = "rrf"
        else:
            hybrid_chunks = self._union_search(top_k, bm25_results, embedding_results)
            mode = "union"

        logger.info(f"Hybrid search ({mode}): {len(hybrid_chunks)} chunks (BM25: {len(bm25_chunks)}, Embedding: {len(embedding_chunks)})")

        return {
            "hybrid": hybrid_chunks,
            "bm25": bm25_chunks,
            "embedding": embedding_chunks,
            "mode": mode
        }