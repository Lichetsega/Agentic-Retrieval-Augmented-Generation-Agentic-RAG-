"""
Query Cache Module for VisitEthiopia RAG Chatbot.
Provides exact-hash and semantic embedding-based caching for user queries.
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

class QueryCache:
    """
    High-performance Q&A response cache supporting exact matching and
    normalized question matching with thread-safe persistence.
    """

    def __init__(self, cache_file: Optional[str] = None, max_size: int = 200, ttl_seconds: int = 86400 * 7):
        if cache_file is None:
            base_dir = Path(__file__).resolve().parent
            cache_file = str(base_dir / "data" / "query_cache.json")

        self.cache_file = cache_file
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        """Loads cached entries from disk in a thread-safe manner."""
        with getattr(self, "_lock", threading.Lock()):
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            if isinstance(data, list):
                                self.entries = data
                except Exception as e:
                    print(f"[WARNING] QueryCache load error: {e}")
                    self.entries = []

    def _save_cache(self):
        """Persists cached entries to disk."""
        # Called within lock context by set() or _load_cache()
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARNING] QueryCache save error: {e}")

    def _normalize(self, text: str) -> str:
        """Normalizes question string for matching using core_utils."""
        from core_utils import normalize_text
        return normalize_text(text)

    def _hash(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode("utf-8")).hexdigest()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates semantic & fuzzy similarity score between two normalized question strings.
        Combines sequence ratio and token overlap Jaccard score.
        """
        import difflib
        n1, n2 = self._normalize(text1), self._normalize(text2)
        if not n1 or not n2:
            return 0.0
        if n1 == n2:
            return 1.0

        # Token set Jaccard Similarity
        t1, t2 = set(n1.split()), set(n2.split())
        if not t1 or not t2:
            return 0.0
        intersection = t1 & t2
        union = t1 | t2
        jaccard = len(intersection) / len(union)

        # Sequence Matcher Ratio
        seq_ratio = difflib.SequenceMatcher(None, n1, n2).ratio()

        # Weighted combined score
        return (0.6 * jaccard) + (0.4 * seq_ratio)

    def get(self, question: str, similarity_threshold: float = 0.82) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached response if an exact, normalized, or semantically similar match exists
        and has not expired (thread-safe). Guaranteed 0 API calls & 0 quota used on cache hit.
        """
        norm_q = self._normalize(question)
        q_hash = self._hash(question)
        now = time.time()

        best_match = None
        highest_score = 0.0

        with self._lock:
            for entry in self.entries:
                # Check TTL
                if now - entry.get("timestamp", 0) > self.ttl_seconds:
                    continue

                cached_norm = entry.get("normalized_question") or self._normalize(entry.get("question", ""))
                cached_hash = entry.get("hash")

                # Step 1: Exact Hash or Exact Normalized Question Match (100% confidence)
                if cached_hash == q_hash or cached_norm == norm_q:
                    print(f"⚡ [CACHE HIT - EXACT] Served exact query match for: '{question}' (0 API calls, 0 quota used)")
                    response = dict(entry.get("response", {}))
                    response["from_cache"] = True
                    response["cache_type"] = "exact"
                    return response

                # Step 2: Calculate Semantic Similarity Score
                sim_score = self._calculate_similarity(question, entry.get("question", ""))
                if sim_score > highest_score:
                    highest_score = sim_score
                    best_match = entry

            # Step 3: Semantic Cache Threshold Evaluation
            if best_match and highest_score >= similarity_threshold:
                matched_q = best_match.get("question", "")
                print(
                    f"⚡ [CACHE HIT - SEMANTIC] Served semantically similar query match for: '{question}' "
                    f" (matched: '{matched_q}', similarity score: {round(highest_score, 2)}) -> 0 API calls, 0 quota used"
                )
                response = dict(best_match.get("response", {}))
                response["from_cache"] = True
                response["cache_type"] = "semantic"
                response["similarity_score"] = round(highest_score, 2)
                return response

        return None

    def set(self, question: str, response: Dict[str, Any]):
        """Caches a question and its generated response (thread-safe)."""
        if not question or not response:
            return

        norm_q = self._normalize(question)
        q_hash = self._hash(question)

        # Do not cache error responses
        if response.get("route") == "error":
            return

        # Ensure docs are JSON serializable
        raw_docs = response.get("docs", [])
        serializable_docs = []
        for d in raw_docs:
            if hasattr(d, "page_content"):
                doc_dict = {"page_content": getattr(d, "page_content", "")}
                if hasattr(d, "metadata"):
                    doc_dict["metadata"] = getattr(d, "metadata", {})
                serializable_docs.append(doc_dict)
            elif isinstance(d, dict):
                serializable_docs.append(d)
            else:
                serializable_docs.append(str(d))

        entry = {
            "question": question,
            "normalized_question": norm_q,
            "hash": q_hash,
            "timestamp": time.time(),
            "response": {
                "answer": response.get("answer", ""),
                "route": response.get("route", "retrieve"),
                "docs": serializable_docs,
                "is_grounded": response.get("is_grounded", True),
            }
        }

        with self._lock:
            # Remove existing match if any
            self.entries = [e for e in self.entries if e.get("hash") != q_hash]

            # Enforce max size limit (FIFO eviction)
            if len(self.entries) >= self.max_size:
                self.entries.pop(0)

            self.entries.append(entry)
            self._save_cache()

    def clear(self):
        """Clears all cached entries (thread-safe)."""
        with self._lock:
            self.entries = []
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except Exception:
                    pass

# Global Singleton Instance
query_cache = QueryCache()
