"""
API entrypoint for website chatbot integration.

This file exposes REST endpoints used by external UIs:
- /ask: main question-answer endpoint
- /health and /ready: service checks
- /admin/reindex: protected manual ingestion trigger
It also handles API auth, rate limiting, caching, and timeout guards.
"""

import os
import re
import sys
import time
import uuid
import json
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# ==================== SYSTEM CONFIGURATION ====================

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from utils import get_chroma_client, get_response, store_docs  # noqa: E402

# ==================== API SETUP ====================

app = Flask(__name__)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    CORS(app, resources={r"/*": {"origins": allowed_origins}})
else:
    CORS(app)
    print("⚠️ ALLOWED_ORIGINS is empty. CORS is currently open for development.")

# ==================== CONFIG ====================

ORGANIZATION_NAME = "Visit Ethiopia"
ORGANIZATION_INFO = (
    "Visit Ethiopia is the official tourism platform that showcases Ethiopia's rich cultural "
    "heritage, historical landmarks, natural attractions, and diverse travel experiences. "
    "It provides valuable information about destinations, cultural activities, travel guides, "
    "and tourism opportunities across the country. The platform aims to promote Ethiopia as a "
    "global tourist destination by highlighting its ancient history, breathtaking natural beauty, "
    "and vibrant cultural diversity."
)
CONTACT_INFO = "https://visitethiopia.et/contact"

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1000"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
API_KEY = os.getenv("CHATBOT_API_KEY", "")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() in {"1", "true", "yes"}

DEFAULT_INGEST_URLS = [
    "https://visitethiopia.et/",
    "https://visitoromia.org/",
    "https://visitamhara.travel/",
    "https://tourismtigrai.com/",
    "https://visitsidama.travel/",
    "https://visitsouthethiopia.et/",
    "https://www.pmo.gov.et/",
    "https://mot.gov.et/",
    "https://mfa.gov.et/",
    "https://www.mor.gov.et/",
    "https://motri.gov.et/en",
    "https://www.motl.gov.et/en",
    "https://www.mofed.gov.et/",
    "https://www.moi.gov.et/",
    "http://www.mint.gov.et/",
    "https://mopd.gov.et/en/",
    "https://mui.gov.et/",
    "https://www.mowe.gov.et/en/",
    "https://www.mowsa.gov.et/",
    "https://www.moh.gov.et/",
    "https://www.ethiopianairlines.com/",
    "https://www.ethiopianholidays.com/",
    "https://ics.gov.et/",
    "https://ecc.gov.et/",
    "https://www.moh.gov.et/",
    "https://combanketh.et/",
    "https://nbe.gov.et/",
    "https://www.etoa.travel/",
    "https://www.stoa-ethiopia.org/",
    "https://www.aha97.com/",
    "https://www.ethiopianrun.org/",
]

_rate_lock = Lock()
_request_buckets = defaultdict(deque)
_cache_lock = Lock()
_response_cache = {}
_history_lock = Lock()
_session_histories = {}
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("API_WORKERS", "4")))


def _json_error(code: str, message: str, status_code: int, request_id: str):
    friendly = {
        "BAD_REQUEST":    "Your message is too long. Please shorten it and try again. 😊",
        "RATE_LIMITED":   "You're sending messages too fast. Please wait a moment. 😊",
        "TIMEOUT":        "This is taking longer than expected. Please try again. 😊",
        "INTERNAL_ERROR": "I'm having trouble right now. Please try again. 😊",
        "UNAUTHORIZED":   "Access denied. Please contact support.",
        "NOT_READY":      "The assistant is still starting up. Please try again shortly. 😊",
    }

    return jsonify({
        "status":     "success",
        "answer":     friendly.get(code, "Something went wrong. Please try again. 😊"),
        "sources":    [],
        "latency_ms": 0,
        "cached":     False,
        "request_id": request_id,
        "_debug":     {"code": code, "message": message},
    }), 200


def _extract_sources(answer: str):
    """Extract source URLs from model output for UI/source attribution."""
    urls = re.findall(r"https?://[^\s)]+", answer or "")
    return sorted(set(urls))


def _is_authorized() -> bool:
    if not REQUIRE_API_KEY:
        return True
    if not API_KEY:
        return False
    provided = request.headers.get("X-API-Key", "")

    return provided == API_KEY


def _is_rate_limited(client_id: str) -> bool:
    """Sliding-window rate limit using per-client timestamp buckets."""
    now = time.time()
    with _rate_lock:
        bucket = _request_buckets[client_id]
        while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            return True
        bucket.append(now)
        return False


def _get_cached_answer(cache_key: str):
    """Read cached response if TTL has not expired."""
    ttl = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "180"))
    now = time.time()
    with _cache_lock:
        cached = _response_cache.get(cache_key)
        if not cached:
            return None
        if now - cached["ts"] > ttl:
            del _response_cache[cache_key]
            return None
        return cached["answer"]


def _set_cached_answer(cache_key: str, answer: str):
    """Write response cache entry with timestamp."""
    with _cache_lock:
        _response_cache[cache_key] = {"answer": answer, "ts": time.time()}


def _sanitize_history(history: Any):
    """
    Validate/normalize chat history payload to [{role, content}, ...].
    Ignores malformed entries instead of failing the whole request.
    """
    if not isinstance(history, list):
        return []

    sanitized = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            sanitized.append({"role": role, "content": content.strip()})
    return sanitized


def _get_effective_history(session_id: str, request_history: Any):
    """
    Use client-provided history when present; otherwise fallback to server session memory.
    """
    sanitized_request_history = _sanitize_history(request_history)
    if sanitized_request_history:
        return sanitized_request_history

    if not session_id:
        return []

    with _history_lock:
        return list(_session_histories.get(session_id, []))


def _append_session_history(session_id: str, user_query: str, answer: str):
    """Persist latest turn for follow-up questions in same session."""
    if not session_id:
        return

    max_messages = int(os.getenv("SESSION_HISTORY_MAX_MESSAGES", "30"))
    with _history_lock:
        history = _session_histories.setdefault(session_id, [])
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > max_messages:
            _session_histories[session_id] = history[-max_messages:]


def _history_fingerprint(chat_history):
    """
    Build a short stable key segment so cache respects conversation state.
    """
    if not chat_history:
        return "no-history"
    tail = chat_history[-6:]
    raw = json.dumps(tail, sort_keys=True, ensure_ascii=True)
    return str(abs(hash(raw)))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check: API process is up and Chroma is queryable."""
    request_id = str(uuid.uuid4())
    try:
        db = get_chroma_client()
        count = db._collection.count()
        return jsonify({
            "status": "ready",
            "request_id": request_id,
            "kb_document_count": count,
        }), 200
    except Exception:
        return _json_error("NOT_READY", "Service dependencies are not ready.", 503, request_id)


@app.route("/admin/reindex", methods=["POST"])
def admin_reindex():
    """Protected endpoint to trigger full source re-indexing on demand."""
    request_id = str(uuid.uuid4())
    if not _is_authorized():
        return _json_error("UNAUTHORIZED", "Invalid or missing API key.", 401, request_id)

    success = 0
    failed = 0
    for url in DEFAULT_INGEST_URLS:
        try:
            if store_docs(url):
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    return jsonify({
        "status": "success",
        "request_id": request_id,
        "indexed_sources": success,
        "failed_sources": failed,
    }), 200


@app.route("/ask", methods=["POST"])
def ask_chatbot():
    """Main chatbot endpoint consumed by external website UI."""
    started = time.time()
    request_id = str(uuid.uuid4())
    client_id = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    print(f'request_id: {request_id}')
    print(f'client_id: {client_id}')
    print(f'request: {request}')

    if not _is_authorized():
        return _json_error("UNAUTHORIZED", "Invalid or missing API key.", 401, request_id)

    if _is_rate_limited(client_id):
        return _json_error("RATE_LIMITED", "Too many requests. Please retry later.", 429, request_id)

    data = request.get_json(silent=False)
    if not isinstance(data, dict):
        return _json_error("BAD_REQUEST", "Request body must be valid JSON object.", 400, request_id)

    user_query = data.get("question")  # data.get("question")
    session_id = data.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        return _json_error("BAD_REQUEST", "'session_id' must be a string when provided.", 400, request_id)

    session_id = (session_id or "").strip()
    if not session_id:
        session_id = f"ip-{client_id}"

    chat_history = _get_effective_history(session_id, data.get("chat_history"))
    print(f"📌 session_id: {session_id} | history length: {len(chat_history)}")
    print(f'the data is: {data}')
    if not isinstance(user_query, str):
        return _json_error("BAD_REQUEST", "'question' must be a string.", 400, request_id)

    user_query = user_query.strip()
    if not user_query:
        return _json_error("BAD_REQUEST", "'question' cannot be empty.", 400, request_id)
    if len(user_query) > MAX_QUESTION_CHARS:
        return _json_error(
            "BAD_REQUEST",
            f"'question' exceeds max length ({MAX_QUESTION_CHARS}).",
            400,
            request_id,
        )

    history_key = _history_fingerprint(chat_history)
    cache_key = f"{session_id or 'no-session'}::{history_key}::{user_query.lower()}"
    cached_answer = _get_cached_answer(cache_key)
    if cached_answer:
        _append_session_history(session_id, user_query, cached_answer)

        latency_ms = int((time.time() - started) * 1000)
        return jsonify({
            "status": "success",
            "request_id": request_id,
            "answer": cached_answer,
            "sources": _extract_sources(cached_answer),
            "latency_ms": latency_ms,
            "cached": True,
        }), 200

    try:
        future = _executor.submit(
            get_response,
            user_query,
            ORGANIZATION_NAME,
            ORGANIZATION_INFO,
            CONTACT_INFO,
            chat_history,
        )
        answer = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
        _append_session_history(session_id, user_query, answer)
        _set_cached_answer(cache_key, answer)
        latency_ms = int((time.time() - started) * 1000)
        return jsonify({
            "status": "success",
            "request_id": request_id,
            "answer": answer,
            "sources": _extract_sources(answer),
            "latency_ms": latency_ms,
            "cached": False,
        }), 200
    except FuturesTimeoutError:
        return _json_error(
            "TIMEOUT",
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds.",
            504,
            request_id,
        )
    except Exception:
        return _json_error(
            "INTERNAL_ERROR",
            "An internal error occurred while processing your request.",
            500,
            request_id,
        )


if __name__ == "__main__":
    print("📡 Visit Ethiopia API is live and listening...")
    app.run(host="0.0.0.0", port=5000, debug=False)