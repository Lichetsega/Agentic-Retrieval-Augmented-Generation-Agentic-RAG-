"""
Hybrid retrieval engine.

This module combines vector similarity (semantic search) and BM25
keyword scoring, then merges both signals into a single ranked result list.
"""

from rank_bm25 import BM25Okapi  # BM25 is the industry standard for keyword matching (like Google Search)
import re
from typing import List, Optional, Dict
from thefuzz import fuzz, process
from langchain_core.documents import Document


class HybridRetriever:
    """
    Combines 'Semantic Search' (Vector) with 'Keyword Search' (BM25).
    This ensures the bot understands the MEANING of a question while also
    finding EXACT terms like 'Lalibela' or 'Fasil Ghebbi'.
    """

    def __init__(self, vector_store):
        """Initialize retriever state and lazy BM25 structures."""
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []
        self.metadatas = []
        self.doc_map = {}
        self.is_initialized = False

    def initialize_bm25(self):
        """
        Extracts all text from the Vector Database and builds a Keyword Index.
        This allows us to search by words even if we don't use the AI model.
        """
        print("🔧 Building BM25 index...")

        try:
            # Fetch all documents currently stored in ChromaDB
            all_data = self.vector_store.get(include=["documents", "metadatas"])
            self.documents = all_data['documents'] or []
            self.metadatas = all_data.get('metadatas') or []

            # Create a map to quickly find the ID of a document by its text
            # Create maps to quickly find the index of a document by content and url
            self.doc_map = {}
            for i, content in enumerate(self.documents):
                meta = self.metadatas[i] if i < len(self.metadatas) and isinstance(self.metadatas[i], dict) else {}
                url = meta.get("url", "")
                self.doc_map[(content, url)] = i
                if content not in self.doc_map:
                    self.doc_map[content] = i

            print(f"📚 Processing {len(self.documents)} documents...")

             # No corpus yet: keep retriever usable, but skip BM25 build.
            if not self.documents:
                self.bm25 = None
                self.doc_map = {}
                self.is_initialized = True
                print("⚠️ BM25 skipped: no documents available in the vector store.")
                return

            # Tokenization: Cleaning text and splitting it into a list of words.
            # Guard against empty-token documents to avoid BM25 division errors.
            tokenized_docs = []
            for d in self.documents:
                tokens = re.sub(r'[^\w\s]', ' ', (d or "").lower()).split()
                tokenized_docs.append(tokens if tokens else ["__empty__"])

            # Build standard BM25 keyword index
            self.bm25 = BM25Okapi(tokenized_docs)

            # Build fuzzy vocabulary: collect words >= 3 characters to act as targets
            vocab_set = set()
            for tokens in tokenized_docs:
                for token in tokens:
                    if len(token) >= 3:
                        vocab_set.add(token)
            self._fuzzy_vocab = list(vocab_set)
            self.is_initialized = True
            print(f"✅ BM25 ready with {len(self.documents)} documents")
            print(f"✅ Fuzzy vocabulary built with {len(self._fuzzy_vocab)} unique terms")

        except Exception as e:
            print(f"❌ BM25 / Fuzzy error: {e}")
            self.is_initialized = False

    def _expand_query_with_fuzzy(self, query: str, score_threshold: int = 80) -> str:
        """
        Checks words in the query against the vocabulary.
        If a word matches a known token with high confidence, it appends the correct
        term to the query string so BM25 keyword search can catch it.
        """
        # If vocabulary is empty, return original query unchanged
        if not self._fuzzy_vocab:
            return query

        # Clean query and split into individual word tokens
        query_tokens = re.sub(r'[^\w\s]', ' ', query.lower()).split()
        extra_terms = []

        for token in query_tokens:
            # Only try to fuzzy-expand significant words (e.g., 3+ characters)
            if len(token) < 3:
                continue

            # Compare query token against database vocabulary using Levenshtein distance ratio
            result = process.extractOne(
                token,
                self._fuzzy_vocab,
                scorer=fuzz.ratio,
            )
            if result is None:
                continue

            best_match, score = result

            # Only append if the correction is high-confidence and distinct
            if score >= score_threshold and best_match != token:
                extra_terms.append(best_match)
                print(f"🔍 Fuzzy expansion: '{token}' → '{best_match}' (score: {score})")

        if extra_terms:
            expanded = query + " " + " ".join(extra_terms)
            print(f"📝 Expanded query: '{query}' → '{expanded}'")
            return expanded

        return query

    def search(self, query: str, k: int = 5, vector_weight: float = 0.5, keyword_weight: float = 0.5) -> List[Document]:
        """
        IMPROVED: Retrieves from BOTH Vector and Keyword indices, merges them,
        and then ranks. This fixes the 'Wenchi/Wonchi' miss.
        """
        if not self.is_initialized:
            self.initialize_bm25()

         # 1. Fuzzy query expansion (Handles spelling variations for keyword matching)
        expanded_query = self._expand_query_with_fuzzy(query)

        # 2. Get Top Candidates from BOTH sources
        # Vector search uses original query (embeddings handle semantics naturally)
        vector_results = self.vector_store.similarity_search_with_score(query, k=k * 3)

        # BM25 uses the expanded query text to capture corrected spellings
        tokenized_query = re.sub(r'[^\w\s]', ' ', expanded_query.lower()).split()
        bm25_scores = self.bm25.get_scores(tokenized_query) if self.bm25 else []

        # 3. Merge unique documents into a single candidate dictionary using (content, url) composite key
        doc_pool = {}  # (content, url) -> (doc_object, normalized_vector_score, doc_idx)

        # Step 3a: Seed the pool with Vector search results & normalize distance score to [0.0, 1.0]
        for doc, score in vector_results:
            meta = doc.metadata or {}
            url = meta.get("url", "")
            key = (doc.page_content, url)
            norm_vector_score = max(0.0, min(1.0, 1.0 - score))  # Clamp similarity metric to [0.0, 1.0]
            doc_idx = self.doc_map.get(key, self.doc_map.get(doc.page_content))
            doc_pool[key] = (doc, norm_vector_score, doc_idx)

        # Step 3b: Bring in top BM25 items that vector search might have completely missed
        if self.bm25 and len(bm25_scores) > 0:
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 3]
            for idx in top_bm25_indices:
                if bm25_scores[idx] <= 0:
                    continue
                content = self.documents[idx]
                meta = self.metadatas[idx] if idx < len(self.metadatas) and isinstance(self.metadatas[idx], dict) else {}
                url = meta.get("url", "")
                key = (content, url)
                if key not in doc_pool:
                    doc_pool[key] = (Document(page_content=content, metadata=meta), 0.0, idx)

        # 4. Calculate Global BM25 Max for normalization scaling
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1

        # 5. Score every candidate document in our combined pool
        final_scored_results = []
        for key, (doc, norm_vector_score, doc_idx) in doc_pool.items():
            # Extract normalized BM25 score using exact doc_idx
            norm_bm25 = (bm25_scores[doc_idx] / max_bm25) if (doc_idx is not None and doc_idx < len(bm25_scores) and max_bm25 > 0) else 0

            # Calculate weighted linear combination:
            # Score = w_vector * S_vector + w_keyword * S_keyword
            combined_score = (vector_weight * norm_vector_score) + (keyword_weight * norm_bm25)
            final_scored_results.append((combined_score, doc))

        # 6. Sort and Return Top K (FIXED: Placed safely outside processing loop blocks)
        PRIMARY_DOMAIN = "https://visitethiopia.et/"
        PRIORITY_BOOST = 0.15  # Tune this value (0.1–0.3)

        boosted_results = []
        for score, doc in final_scored_results:
            url = doc.metadata.get("url", "")
            if PRIMARY_DOMAIN in url:
                score += PRIORITY_BOOST
            boosted_results.append((score, doc))

        final_scored_results = boosted_results

        # 7. Sort and Return Top K
        final_scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in final_scored_results[:k]]

    def multi_search(self, queries: List[str], k: int = 5) -> List[Document]:
        """
        Proactive Multi-Query Search:
        Executes hybrid retrieval across multiple query variations,
        merges and deduplicates candidate document chunks while preserving source metadata.
        """
        if not queries:
            return []

        unique_docs: Dict[str, Document] = {}
        for q in queries:
            if not q or not q.strip():
                continue
            retrieved = self.search(query=q.strip(), k=k)
            for doc in retrieved:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc

        return list(unique_docs.values())


    def _find_document_index(self, content: str) -> Optional[int]:
        """Internal helper to find where a specific text chunk lives in the index."""
        try:
            return self.documents.index(content)
        except ValueError:
            content_start = content[:500]
            for i, doc in enumerate(self.documents):
                if doc.startswith(content_start):
                    return i
            return None

    def refresh(self):
        """Forces the system to rebuild the keyword index (used when new data is crawled)."""
        print("🔄 Refreshing BM25...")
        self.is_initialized = False
        self.initialize_bm25()