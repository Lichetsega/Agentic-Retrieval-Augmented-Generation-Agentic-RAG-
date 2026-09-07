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
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Local Imports
from hybrid_retriever import HybridRetriever
from text_to_doc import get_doc_chunks
from web_crawler import crawl_website
from prompt import get_prompt
from api_key_manager import api_key_manager
# ==================== GLOBAL CONFIGURATION ====================

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma")
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "website_data"


# ==================== GLOBAL STATE MANAGEMENT ====================

_hybrid_retriever = None
_current_chain = None

# Public flag used by app.py sidebar status
LLM_AVAILABLE = True

# Ollama circuit-breaker state
_ollama_fail_count = 0
_ollama_retry_after = 0.0

# ==================== 1. VECTOR DATABASE ENGINE ====================

def get_chroma_client() -> Chroma:
    """Create or connect to the persisted Chroma collection."""
    settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=settings,
    )
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
    # 🧹 NEW STEP 6: ORPHANED CONTENT CLEANUP (DELETED PAGES)
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

# ==================== 5. SMART ROUTING (GREETINGS & FOLLOW-UPS) ====================

def is_greeting(question: str) -> bool:
    q = question.lower().strip().replace("?", "")
    greetings = {
        "hi",
        "hello",
        "hey",
        "selam",
        "good afternoon",
        "good evening",
        "how are you",
        "how's it going",
        "yo",
        "morning",
        "afternoon",
    }
    return q in greetings

def is_identity_query(question: str) -> bool:
    q = re.sub(r'[^\w\s]', '', question.lower()).strip()
    identity_triggers = {
        # Identity
        "who are you",
        "who is this",
        "what is this",
        "who am i speaking to",
        "who am i talking to",
        "who is behind this",
        "who is on the other side",
        "what is your name",
        "whats your name",
        "introduce yourself",
        "tell me about yourself",
        "tell me who you are",
        # Creator / origin
        "who made you",
        "who built you",
        "who created you",
        "who developed you",
        "who is your creator",
        "who is your developer",
        "who is behind you",
        # Technology / architecture
        "what are you made of",
        "what are you made off",
        "what technology are you",
        "what technology powers you",
        "what engine do you use",
        "what architecture is this",
        "what llm is this",
        "what llm are you",
        "what model are you",
        "what model is this",
        "is this rag",
        "are you a rag",
        "are you rag",
        "how do you work",
        "how does this work",
        "where do you get your info",
        "where do you get your information",
        "where do you get your data",
        # System / instructions
        "what is your system prompt",
        "show me your system prompt",
        "what are your instructions",
        "show me your instructions",
        "what are your rules",
        "show me your code",
        "what is your code",
        # LLM identity checks
        "are you google",
        "are you gemini",
        "are you chatgpt",
        "are you gpt",
        "are you openai",
        "are you deepseek",
        "are you meta",
        "are you llama",
        "are you ollama",
        "are you claude",
        "are you an ai",
        "are you a bot",
        "are you a robot",
        "are you human",
        "are you a machine",
        # Parameters / internals
        "what is your temperature",
        "what is your context window",
        "what are your parameters",
        "what version are you",
    }
    if any(trigger in q for trigger in identity_triggers):
        return True
    tech_keywords = {"google", "gemini", "openai", "chatgpt", "llama", "claude"}
    words = q.split()
    if any(word in words for word in tech_keywords):
        return True

    return False

def get_greeting_response(question: str) -> str:
    return """🇪🇹 **Selam! Welcome to the Visit Ethiopia Travel Assistant!** I'm here to help you explore the Land of Origins. I can assist with:
- 🏛️ **Historical Sites** (Lalibela, Axum, Gondar)
- 🏔️ **Nature** (Simien Mountains, Bale Mountains)
- 🍲 **Culture** (Cuisine, Festivals, Coffee)

What would you like to explore today? ✨"""

def get_identity_response() -> str:
    return """I am the Visit Ethiopia Travel Assistant, a specialized digital guide designed specifically to showcase the wonders of the Land of Origins. 🇪🇹

My knowledge is built from the official records of Ethiopia's national and regional tourism bureaus to ensure you get the most accurate travel information. Rather than discussing my technical background, I’d love to tell you more about Ethiopia!

Are you interested in exploring our historical landmarks, national parks, or our vibrant cultural festivals?"""


def format_chat_history(raw_history: list) -> List[Tuple[str, str]]:
    formatted: List[Tuple[str, str]] = []
    if not raw_history:
        return formatted

    last_user_msg = None
    for msg in raw_history:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "").strip()
        role = msg.get("role")

        # Skip the initial bot greeting message safely without exact string matching
        if role == "assistant" and ("Welcome to the Visit Ethiopia" in content or "Selam!" in content):
            continue

        if role == "user":
            last_user_msg = content
        elif role == "assistant" and last_user_msg is not None:
            formatted.append((last_user_msg, content))
            last_user_msg = None

    return formatted

def _stringify_chat_history(pairs: List[Tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines = []
    for user_msg, ai_msg in pairs:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {ai_msg}")
    return "\n".join(lines)

def _build_memory_context(pairs: List[Tuple[str, str]], memory_pairs: int = 3) -> str:
    """
    Build compact memory context from the last N conversation pairs.
    This is injected into generation prompt, not used as primary retrieval query.
    """
    if not pairs:
        return ""
    recent_pairs = pairs[-memory_pairs:]
    lines = []
    for user_msg, ai_msg in recent_pairs:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {ai_msg}")
    return "\n".join(lines)

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
    context_text = "\n\n".join([doc.page_content for doc in docs])
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
    context_text = "\n\n".join([doc.page_content for doc in docs])
    chat_history_text = _stringify_chat_history(formatted_history)

    tried_keys = set()
    max_attempts = max(1, len(api_key_manager.keys)) if hasattr(api_key_manager, "keys") else 1
    last_error = None

    fallback_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
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
                    temperature=0.2,
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

                answer = getattr(result, "content", None)
                if answer is None:
                    answer = result.get("content") if isinstance(result, dict) else str(result)

                api_key_manager.mark_key_success(key)
                return {"answer": answer}

           except Exception as e:
                last_error = e
                msg = str(e).lower()

                # 1. Catch Model Overload / Server Errors
                # If the specific model is busy (503), overloaded, or timed out, fallback to the Lite model.
                if any(x in msg for x in ["overloaded", "busy", "503", "500", "timeout"]):
                    print(f"⚠️ {current_model} is busy/unavailable. Falling back to the next model...")
                    continue  # Try the next model in the fallback_models list

                # 2. Catch Key/Quota Errors
                # If it's a rate limit or auth issue, the model isn't the problem; the key is.
                if any(x in msg for x in
                       ["quota", "rate", "429", "permission", "api key", "invalid", "unauthenticated"]):
                    print(f"⚠️ API Key Quota/Auth error. Rotating to the next key...")
                    api_key_manager.mark_key_failed(key)
                    break  # Break out of the inner model loop to grab the next API key

                # 3. Catch Model Config Errors
                if "not_found" in msg or "model" in msg:
                    print(f"⚠️ Model {current_model} not found. Skipping to next model...")
                    continue

                raise

    raise Exception(f"Gemini generation failed: {last_error}")

def get_response(
    question: str,
    organization_name: str,
    organization_info: str,
    contact_info: str,
    chat_history: list = None,
) -> str:
    """
        End-to-end response pipeline with input interception.
        1. Identity Interception
        2. Greeting Interception
        3. RAG (Retrieve -> Generate -> Fallback)
        """
    global LLM_AVAILABLE
    if not question:
        return "How can I help you today? 🇪🇹"

    if is_identity_query(question):
        return get_identity_response()

    if is_greeting(question):
        return get_greeting_response(question)
    #  PREPARE FOR RAG
    formatted_history = format_chat_history(chat_history or [])
    memory_context = _build_memory_context(formatted_history, memory_pairs=3)

    # Expand query for better retrieval
    retrieval_query = _build_retrieval_query(question, formatted_history)
    generation_question = _build_generation_question(question, formatted_history)

    print("\n" + "#"*70)
    print("🔍 DETAILED DEBUG TRACE START")
    print("#"*70)
    print(f"👤 RAW QUESTION: {question}")
    print(f"🔄 RETRIEVAL QUERY: {retrieval_query}")
    print(f"📝 GENERATION QUESTION: {generation_question}")
    print("#"*70 + "\n")

    try:
        hybrid = get_hybrid_retriever()
        docs = hybrid.search(query=retrieval_query, k=10)

        print("\n" + "="*70)
        print(f"🧠 CONTEXT SENT TO LLM ({len(docs)} chunks, truncated to 500 chars):")
        for i, d in enumerate(docs):
            print(f"\n--- Chunk {i+1} [Source: {d.metadata.get('url', 'Unknown')}] ---")
            print(d.page_content[:500] + "..." if len(d.page_content) > 500 else d.page_content)
        print("="*70 + "\n")

    except Exception as e:
        print(f"❌ Search Error: {e}")
        return "I am currently unable to access my knowledge base. Please try again later."

    # 1) Try Gemini first (primary)
    print("🤖 ATTEMPTING GEMINI GENERATION...")
    try:
        result_data = _gemini_call(
            generation_question,
            docs,
            organization_name,
            organization_info,
            contact_info,
            formatted_history,
            memory_context,
        )
        answer = result_data.get("answer", "")
        print(f"\n✅ GEMINI RAW ANSWER:\n{answer}")
        print("-" * 70)
        print("="*70 + "\n")
        LLM_AVAILABLE = True
        return answer

    except Exception as e:
        print(f"⚠️ Gemini Error (primary): {e}")

    # 2) Fallback to Ollama
    if _is_ollama_healthy_now():
        print("\n🤖 ATTEMPTING OLLAMA FALLBACK GENERATION...")
        chain = get_chain_with_failover()
        if chain:
            try:
                result_data = _llm_call(
                    generation_question,
                    docs,
                    organization_name,
                    organization_info,
                    contact_info,
                    formatted_history,
                    memory_context,
                )
                answer = result_data.get("answer", "")
                print(f"\n✅ OLLAMA RAW ANSWER:\n{answer}")
                print("-" * 70)
                print("="*70 + "\n")
                _record_ollama_success()
                LLM_AVAILABLE = True
                return answer
            except Exception as e:
                print(f"⚠️ Ollama Fallback Error: {e}")
                if _is_ollama_runtime_error(e):
                    _record_ollama_failure()
    else:
        print("⚠️ OLLAMA IS UNAVAILABLE OR IN COOLDOWN.")

    # 3) Final safe mode fallback
    print("\n🛡️ ENTERING SAFE MODE FALLBACK")
    print("="*70 + "\n")
    LLM_AVAILABLE = False
    return "I am currently experiencing technical difficulties and cannot connect to my language models. Please try again later or visit the official websites for information."
