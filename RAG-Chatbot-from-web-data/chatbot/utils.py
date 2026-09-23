"""
Visit Ethiopia - Core Utility Functions
---------------------------------------
This is the 'Brain' of the application. It orchestrates:
1. Data Ingestion: Crawling and saving websites into a Vector Database.
2. Hybrid Retrieval: Finding the best information using both AI (Semantic) and Keywords (BM25).
3. Failover Logic: Gemini primary, Ollama fallback, then safe mode.
4. Intelligent Routing: Handling greetings and 'Safe Mode' when the AI is offline.
"""

import os
import re
import time
from pathlib import Path
from typing import List, Tuple
import hashlib
from urllib.parse import urlparse

# LangChain Imports
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# Local Imports
from hybrid_retriever import HybridRetriever
from text_to_doc import get_doc_chunks
from web_crawler import crawl_website
from prompt import get_prompt
from api_key_manager import api_key_manager
# Core Shared Utilities Import
from core_utils import (
    get_chroma_client,
    format_chat_history,
    stringify_chat_history,
    build_memory_context,
    clean_meta_talk,
    extract_clean_text,
    format_live_currency_context,
)

_stringify_chat_history = stringify_chat_history
_build_memory_context = build_memory_context

_hybrid_retriever = None
_current_chain = None
LLM_AVAILABLE = True
# ==================== 2. DATA INGESTION (CRAWL & STORE) ====================
def store_docs(url: str) -> bool:
    """
    Crawl one source URL, conditionally update Chroma chunks if content changed,
    and automatically purge orphaned URLs that no longer exist on the site.
    """
    vector_store = get_chroma_client()
    crawled_data = crawl_website(url)
    if not crawled_data:
        print("❌ Crawl produced no pages. Skipping storage.")
        return False

    all_docs = []
    chunks_modified = False

    # 🎯 Track every single internal URL discovered during THIS specific crawl session
    visited_urls_in_session = set()

    for text, metadata in crawled_data:
        page_url = metadata["url"]
        visited_urls_in_session.add(page_url)  # Log it as active
        clean_text = text.strip()

        # Step 1: Generate a unique hash of the newly crawled text content
        new_hash = hashlib.md5(clean_text.encode('utf-8')).hexdigest()

        # Step 2: Query the vector database for any existing records under this URL
        try:
            existing = vector_store.get(
                where={"url": page_url},
                include=["metadatas"]
            )
        except Exception as e:
            print(f"⚠️ Could not read existing data for {page_url}: {e}")
            existing = None

        # Step 3: Compare hashes if database records exist
        if existing and existing["ids"] and existing["metadatas"]:
            old_hash = existing["metadatas"][0].get("text_hash")

            if old_hash == new_hash:
                print(f"⏩ Content unchanged for: {page_url} (Hashes match). Skipping database write.")
                continue
                # If hashes don't match, content has changed. Purge the old out-of-date chunks.
            print(f"🔄 Content change detected for: {page_url}. Updating knowledge base...")
            vector_store.delete(ids=existing["ids"])
            chunks_modified = True
        else:
            print(f"🆕 New page discovered: {page_url}. Preparing ingestion...")
            chunks_modified = True
        # Step 4: Inject the text hash into the metadata bundle before chunking
        metadata["text_hash"] = new_hash
        docs = get_doc_chunks(clean_text, metadata)
        all_docs.extend(docs)
    # Step 5: Batch-commit documents to Chroma if any additions/modifications occurred
    if all_docs:
        vector_store.add_documents(all_docs)
        print(f"✅ Successfully written {len(all_docs)} updated chunks to the knowledge base.")
        chunks_modified = True

    # =================================================================
    # STEP 6: ORPHANED CONTENT CLEANUP (DELETED PAGES)
    # =================================================================
    print(f"\n🧼 Checking for orphaned pages linked to {url}...")
    try:
        # 1. Fetch all documents currently stored belonging to this domain group/bureau
        # (We filter by domain prefix to avoid accidentally touching other regions' data)
        parsed_base = urlparse(url)
        base_domain = parsed_base.netloc.lower().replace("www.", "")
        # Fetch metadatas of everything in the collection
        all_stored = vector_store.get(include=["metadatas"])

        if all_stored and all_stored["ids"] and all_stored["metadatas"]:
            orphan_ids_to_delete = []
            deleted_urls_logged = set()

            for idx, meta in enumerate(all_stored["metadatas"]):
                stored_url = meta.get("url", "")
                stored_domain = urlparse(stored_url).netloc.lower().replace("www.", "")
                # Only evaluate if it belongs to the domain we are currently indexing
                if stored_domain == base_domain:
                    # If it's in the DB but WAS NOT seen in the fresh crawl session, it's an orphan!
                    if stored_url not in visited_urls_in_session:
                        orphan_ids_to_delete.append(all_stored["ids"][idx])
                        deleted_urls_logged.add(stored_url)

            # 2. Issue a batch deletion call for any identified orphan IDs
            if orphan_ids_to_delete:
                print(f"🗑️ Found {len(deleted_urls_logged)} orphaned pages that no longer exist on the website.")
                for d_url in deleted_urls_logged:
                    print(f"   ❌ Removing: {d_url}")
                vector_store.delete(ids=orphan_ids_to_delete)
                print(f"✅ Successfully purged {len(orphan_ids_to_delete)} stale chunks from the database.")
                chunks_modified = True
            else:
                print("✨ No orphaned pages found for this domain. Database is clean!")

    except Exception as e:
        print(f"⚠️ Warning: Safe-cleanup pass failed: {e}")
    # =================================================================

    if chunks_modified:
        global _hybrid_retriever
        if _hybrid_retriever:
            _hybrid_retriever.initialize_bm25()
        return True

    print("✨ Knowledge base is already perfectly synchronized. No changes made.")
    return False
# ==================== 3. SEARCH & RETRIEVAL LOGIC ====================

def get_hybrid_retriever() -> HybridRetriever:
    """Lazy-init singleton retriever so index build happens once per process."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        print(" Booting Hybrid Search Engine...")
        _hybrid_retriever = HybridRetriever(get_chroma_client())
        _hybrid_retriever.initialize_bm25()
    return _hybrid_retriever

# ==================== 4. AI CONVERSATION CHAIN ====================

def make_chain():
    if ChatOllama is None:
        print("⚠️ ChatOllama is unavailable. Ollama path is disabled.")
        return None

    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.2,
        timeout=30,
    )


    prompt = get_prompt()
    return prompt | model

def get_chain_with_failover():
    global _current_chain
    if _current_chain is None:
        _current_chain = make_chain()
    return _current_chain

def _is_ollama_runtime_error(err: Exception) -> bool:
    msg = str(err).lower()
    markers = [
        "status code: 500",
        "status code 404",
        "model is not found",
        "pull the model",
        "llama runner process has terminated",
        "runner terminated",
        "load failed",
        "connection refused"
        "cuda",
        "out of memory",
    ]
    return any(m in msg for m in markers)

def _ollama_cooldown_seconds() -> int:
    try:
        return int(os.getenv("OLLAMA_COOLDOWN_SECONDS", "300"))
    except Exception:
        return 300

def _ollama_max_fails() -> int:
    try:
        return int(os.getenv("OLLAMA_MAX_FAILS", "1"))
    except Exception:
        return 1

def _is_ollama_healthy_now() -> bool:
    return time.time() >= _ollama_retry_after

def _record_ollama_failure() -> None:
    global _ollama_fail_count, _ollama_retry_after
    _ollama_fail_count += 1
    if _ollama_fail_count >= _ollama_max_fails():
        _ollama_retry_after = time.time() + _ollama_cooldown_seconds()

def _record_ollama_success() -> None:
    global _ollama_fail_count, _ollama_retry_after
    _ollama_fail_count = 0
    _ollama_retry_after = 0.0

# Note: Greeting heuristics, identity queries, and chat history formatting are centralized in core_utils.py

_FILLER_PHRASES = re.compile(
    r'\b(tell me about|tell me more|what is|what are|can you|i want to know about|explain|describe|give me info on|give me information about)\b',
    re.IGNORECASE
)
_CONTEXT_PRONOUNS = {"it", "there","other","others", "that", "they", "them", "this", "those", "these", "its", "here" , "nearby" , "around" , "more",}

def _build_retrieval_query(question: str, formatted_history: List[Tuple[str, str]]) -> str:
    print(f"🛠️ DEBUG - Received History Pairs: {len(formatted_history)}")

    if not formatted_history:
        return question

    question_tokens = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    has_unresolved_pronoun = bool(question_tokens & _CONTEXT_PRONOUNS)

    if has_unresolved_pronoun:
        last_user_query = formatted_history[-1][0]  # always the most recent topic
        expanded = f"{last_user_query} {question}"
        print(f"🔗 Expanded retrieval query: '{expanded}'")
        return expanded

    return question

def _build_generation_question(question: str, formatted_history: List[Tuple[str, str]]) -> str:
    """
    When the question contains a pronoun, prepend the last topic explicitly
    so the LLM never asks for clarification.
    """
    if not formatted_history:
        return question

    question_tokens = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    has_unresolved_pronoun = bool(question_tokens & _CONTEXT_PRONOUNS)

    if has_unresolved_pronoun:
        last_user_query = formatted_history[-1][0]
        clean_context = _FILLER_PHRASES.sub('', last_user_query).strip()
        explicit_question = f"{question} (referring to: {clean_context})"
        print(f"🎯 GENERATION QUESTION: {explicit_question}")
        return explicit_question

    return question

# ==================== 6. MAIN RESPONSE GENERATOR ====================

def _llm_call(
    question: str,
    docs: List[Document],
    org_name: str,
    org_info: str,
    contact: str,
    formatted_history: list,
    memory_context: str,
):
    """Ollama fallback LLM call path with normalized output extraction."""
    max_attempts = 1
    raw_docs_text = "\n\n".join([doc.page_content for doc in docs])
    context_text = f"{format_live_currency_context()}\n\n{raw_docs_text}"
    chat_history_text = _stringify_chat_history(formatted_history)

    last_error = None
    for attempt in range(max_attempts):
        chain = get_chain_with_failover()
        if not chain:
            last_error = RuntimeError("LLM chain is not available")
            break

        try:
            result = chain.invoke({
                "question": question,
                "chat_history": chat_history_text,
                "memory_context": memory_context,
                "organization_name": org_name,
                "organization_info": org_info,
                "contact_info": contact,
                "context": context_text,
            })

            answer = getattr(result, "content", None)
            if answer is None:
                answer = result.get("content") if isinstance(result, dict) else str(result)

            return {"answer": answer}

        except Exception as e:
            last_error = e
            print(f"⚠️ AI Error (attempt {attempt + 1}/{max_attempts}): {e}")
            continue
    raise Exception(f"All attempts failed: {last_error}")

def _gemini_call(
    question: str,
    docs: List[Document],
    org_name: str,
    org_info: str,
    contact: str,
    formatted_history: list,
    memory_context: str,
):
    """Gemini primary path with API-key rotation and retry semantics."""
    raw_docs_text = "\n\n".join([doc.page_content for doc in docs])
    context_text = f"{format_live_currency_context()}\n\n{raw_docs_text}"
    chat_history_text = _stringify_chat_history(formatted_history)

    tried_keys = set()
    max_attempts = max(1, len(api_key_manager.keys)) if hasattr(api_key_manager, "keys") else 1
    last_error = None

    fallback_models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
        "gemini-3.5-flash-lite",
    ]
    for _ in range(max_attempts):
        key = api_key_manager.get_next_available_key()
        if not key or key in tried_keys:
            break

        tried_keys.add(key)
        for current_model in fallback_models:
           try:
                print(f"🤖 Attempting Gemini generation with model: {current_model}")
                model = ChatGoogleGenerativeAI(
                    model=current_model,
                    google_api_key=key,
                    temperature=0.0,
                    max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "0")),
                )
                chain = get_prompt() | model

                result = chain.invoke({
                    "question": question,
                    "chat_history": chat_history_text,
                    "memory_context": memory_context,
                    "organization_name": org_name,
                    "organization_info": org_info,
                    "contact_info": contact,
                    "context": context_text,
                })

                answer = extract_clean_text(result)
                answer = clean_meta_talk(answer)

                api_key_manager.mark_key_success(key)
                return {"answer": answer}

           except Exception as e:
                last_error = e
                msg = str(e).lower()

                # 1. Safeguard: Catch Client / Invalid Input / Safety Errors -> FAIL FAST! Do NOT rotate keys!
                if api_key_manager.is_client_error(e):
                    print(f"🛑 Client/Input Error: '{e}' - Failing fast without key rotation to protect API key quotas.")
                    raise e

                # 2. Catch Model Config / 404 Errors -> Try next model on SAME key!
                if any(x in msg for x in ["not_found", "404", "model"]):
                    print(f"⚠️ Model {current_model} not available on key. Trying next model...")
                    continue

                # 3. Catch Model Overload / Server Errors
                if any(x in msg for x in ["overloaded", "busy", "503", "500", "timeout"]):
                    print(f"⚠️ {current_model} is busy/unavailable. Falling back to the next model...")
                    continue

                # 4. Catch Key/Quota Errors -> Rotate to next key!
                if api_key_manager.is_quota_error(e):
                    print(f"⚠️ API Key Quota/Auth error ({api_key_manager._mask_key(key)}). Rotating to next key...")
                    api_key_manager.mark_key_failed(key, error=e)
                    break

                raise e

    raise Exception(f"Gemini generation failed: {last_error}")

_agentic_engine = None

def get_agentic_rag_engine():
    """
    This function creates and holds a single, shared instance of `LangChainAgenticRAG`.
    """
    global _agentic_engine
    if _agentic_engine is None:
        from langchain_agent import LangChainAgenticRAG
        _agentic_engine = LangChainAgenticRAG(retriever_getter=get_hybrid_retriever)
    return _agentic_engine

def get_agentic_response(
    question: str,
    organization_name: str,
    organization_info: str,
    contact_info: str,
    chat_history: list = None,
) -> dict:
    """
    Agentic RAG pipeline execution returning full result dictionary (answer + step traces).
    """
    global LLM_AVAILABLE
    if not question or not question.strip():
        return {
            "answer": "How can I help you today? 🇪🇹",
            "route": "greeting",
            "trace_logs": ["Empty question received."],
            "docs": [],
            "is_grounded": True,
            "execution_time_sec": 0.0
        }

    formatted_history = format_chat_history(chat_history or [])
    chat_history_str = _stringify_chat_history(formatted_history)
    memory_context = _build_memory_context(formatted_history, memory_pairs=3)

    engine = get_agentic_rag_engine()
    try:
        res = engine.run(
            question=question,
            organization_name=organization_name,
            organization_info=organization_info,
            contact_info=contact_info,
            chat_history_text=chat_history_str,
            memory_context=memory_context,
        )
        LLM_AVAILABLE = True
        return res
    except Exception as e:
        print(f"⚠️ Agentic RAG execution failed: {e}. Falling back to standard pipeline.")
        # Fallback to standard RAG pipeline if agentic loop encounters error
        try:
            hybrid = get_hybrid_retriever()
            docs = hybrid.search(query=question, k=10)
            result_data = _gemini_call(
                question, docs, organization_name, organization_info, contact_info, formatted_history, memory_context
            )
            LLM_AVAILABLE = True
            return {
                "answer": result_data.get("answer", ""),
                "route": "retrieve",
                "trace_logs": [f"Standard RAG fallback executed due to: {e}"],
                "docs": docs,
                "is_grounded": True,
                "execution_time_sec": 0.0
            }
        except Exception as fallback_err:
            print(f"❌ Fallback Error: {fallback_err}")
            LLM_AVAILABLE = False
            return {
                "answer": "I am currently experiencing technical difficulties. Please try again later.",
                "route": "error",
                "trace_logs": [f"Error: {fallback_err}"],
                "docs": [],
                "is_grounded": False,
                "execution_time_sec": 0.0
            }


def get_response(
    question: str,
    organization_name: str,
    organization_info: str,
    contact_info: str,
    chat_history: list = None,
) -> str:
    """
    End-to-end response pipeline returning standard text response string.
    """
    res = get_agentic_response(
        question=question,
        organization_name=organization_name,
        organization_info=organization_info,
        contact_info=contact_info,
        chat_history=chat_history,
    )
    return res.get("answer", "")

