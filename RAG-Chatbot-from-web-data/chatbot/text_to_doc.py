"""
Text preprocessing and chunking pipeline.

This module cleans crawled raw text, splits it into retrievable chunks,
and converts chunks into LangChain Document objects with safe metadata.
"""

import re
from typing import Dict, List, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =================================================================
# TEXT PRE-PROCESSING
# =================================================================

def clean_text(text: str) -> str:
    """
    Normalize scraped text and remove common website noise.

    Why:
    - Reduces irrelevant footer/header/legal text in embeddings
    - Improves retrieval quality for RAG
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Common junk patterns from headers/footers/legal sections
    junk_patterns = [
        r"cookie[s]?\s*policy.*",
        r"privacy\s*policy.*",
        r"terms\s*of\s*service.*",
        r"all\s*rights\s*reserved.*",
        r"©.*",
        r"follow\s+us\s+on.*",
        r"subscribe.*newsletter.*",
        r"menu.*menu.*",
    ]

    # Remove boilerplate website text that hurts retrieval quality.
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


# =================================================================
# DOCUMENT CHUNKING
# =================================================================

def text_to_docs(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """
    Split long text into overlapping chunks and convert to LangChain Documents.

    Strategy:
    - RecursiveCharacterTextSplitter to preserve semantic boundaries
    - Skip tiny fragments that are usually noise
    - Enrich chunk content with source/bureau attribution
    """
    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = text_splitter.split_text(text)
    docs: List[Document] = []

    bureau = metadata.get("bureau", "National Tourism")
    title = metadata.get("title", "")
    url = metadata.get("url", "")

    for chunk in chunks:
        clean_chunk = chunk.strip()

        # Keep source context inside page_content for better grounded answers.
        is_primary = "visitethiopia.et" in url
        enriched_chunk = (
            f"{'[PRIMARY SOURCE] ' if is_primary else ''}"
            f"SOURCE BUREAU: {bureau}\n"
            f"TITLE: {title}\n"
            f"URL: {url}\n\n"
            f"{clean_chunk}"
        )

        # Chroma metadata cannot contain None values.
        raw_metadata = {
            "url": url,
            "title": title,
            "bureau": bureau,  # keep bureau explicitly
            "source": metadata.get("source"),
            "region": metadata.get("region"),
        }
        # Keep metadata JSON-safe for Chroma (no None values).
        safe_metadata = {k: v for k, v in raw_metadata.items() if v is not None}

        doc = Document(
            page_content=enriched_chunk,
            metadata=safe_metadata,
        )

        docs.append(doc)

    return docs


def get_doc_chunks(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """
    Main entry point:
    1) clean raw text
    2) split into chunked Documents
    """
    cleaned_text = clean_text(text)
    return text_to_docs(cleaned_text, metadata)