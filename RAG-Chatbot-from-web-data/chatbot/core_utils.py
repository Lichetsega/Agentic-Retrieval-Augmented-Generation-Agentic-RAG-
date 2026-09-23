"""
Visit Ethiopia - Core Shared Utilities & System Constants
---------------------------------------------------------
Centralized utility module holding shared organization constants, vector database
singletons, text cleaners, intent heuristics, and chat history processors.
This module eliminates code duplication across api_server.py, app.py, utils.py,
langchain_agent.py, and query_cache.py.
"""

import os
import re
import time
import json
import datetime
import urllib.request
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# LangChain Vector DB & Embeddings Imports
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings


# ==================== GLOBAL ORGANIZATION CONFIGURATION ====================

ORGANIZATION_NAME = "Visit Ethiopia"

ORGANIZATION_INFO = """
Visit Ethiopia is the official tourism platform that showcases Ethiopia's rich cultural heritage, 
historical landmarks, natural attractions, and diverse travel experiences. It provides valuable 
information about destinations, cultural activities, travel guides, and tourism opportunities 
across the country.

The platform aims to promote Ethiopia as a global tourist destination by highlighting its 
ancient history, breathtaking natural beauty, and vibrant cultural diversity. It also supports 
travelers in planning their visits by offering insights into various destinations, traditions, 
and travel-related information.

The platform is a unified travel assistant providing data from federal and regional 
tourism bureaus including Oromia, Amhara, Tigray, Sidama, and South Ethiopia.
"""

CONTACT_INFO = """
Official Website: https://visitethiopia.et/
Contact & Inquiries: https://visitethiopia.et/contact

Visit Ethiopia primarily provides informational content to help travelers explore the country. 
For detailed travel arrangements, bookings, or inquiries, users are advised to consult official 
tourism channels or accredited local operators.
"""

# Default Target Tourism Website URLs for Data Ingestion & Scheduled Crawling
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
    "https://combanketh.et/",
    "https://nbe.gov.et/",
    "https://www.etoa.travel/",
    "https://www.stoa-ethiopia.org/",
    "https://www.aha97.com/",
    "https://www.ethiopianrun.org/",
]

DEFAULT_URLS = DEFAULT_INGEST_URLS


# ==================== VECTOR DATABASE & EMBEDDINGS SINGLETON ====================

EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma")
Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "website_data"

def get_chroma_client() -> Chroma:
    """Create or connect to the persisted Chroma collection (Singleton pattern)."""
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


# ==================== TEXT NORMALIZATION & CLEANERS ====================

def normalize_text(text: str) -> str:
    """Standardizes string by stripping punctuation, extra spaces, and lowercasing."""
    if not text:
        return ""
    clean = re.sub(r'[^\w\s]', '', text.strip().lower())
    return " ".join(clean.split())

def clean_json_markdown(raw_content: str) -> str:
    """Strips markdown ```json code fences from LLM outputs for clean JSON parsing."""
    if not raw_content:
        return ""
    cleaned = raw_content.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0].strip()
    return cleaned

def extract_clean_text(val: Any) -> str:
    """Recursively extracts actual text from string, list, nested list, or dict containing extras/signature."""
    if val is None:
        return ""

    def _rec(obj: Any) -> List[str]:
        res = []
        if isinstance(obj, str):
            res.append(obj)
        elif isinstance(obj, dict):
            if "text" in obj and isinstance(obj["text"], str):
                res.append(obj["text"])
            else:
                for v in obj.values():
                    res.extend(_rec(v))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                res.extend(_rec(item))
        elif hasattr(obj, "content"):
            res.extend(_rec(getattr(obj, "content")))
        return res

    extracted = _rec(val)
    if extracted:
        return "\n".join(extracted).strip()
    return str(val).strip()

def clean_meta_talk(text: str) -> str:
    """Strips internal system meta-talk leakage, parenthetical annotations, meta-phrases, and raw base64 data blobs from responses."""
    if not text:
        return ""
    # Strip parenthetical meta annotations
    text = re.sub(r'\s*\((?:Enrichment from|From|Retrieved from|Knowledge from|Context enrichment|While not explicitly|Discrepancy)[^)]*\)', '', text, flags=re.IGNORECASE)
    # Strip meta phrases like "While not explicitly in the context...", "According to the provided text...", etc.
    text = re.sub(r'(?:While not explicitly in the context|This detail might be a slight discrepancy in the context|Based on the provided context|According to the retrieved documents)[^.\n]*[.\n]?', '', text, flags=re.IGNORECASE)
    # Strip raw base64 data blobs (long contiguous base64 strings > 100 chars)
    text = re.sub(r'[A-Za-z0-9+/=]{100,}', '', text)
    return text.strip()

def extract_sources(answer: Any) -> List[str]:
    """Extracts source URLs from model answer strings or lists for UI attribution."""
    if not answer:
        return []
    if isinstance(answer, list):
        answer = "\n".join([str(x) for x in answer])
    elif not isinstance(answer, str):
        answer = str(answer)
    urls = re.findall(r"https?://[^\s)]+", answer)
    return sorted(set(urls))


# ==================== CHAT HISTORY PROCESSORS ====================

def format_chat_history(raw_history: list) -> List[Tuple[str, str]]:
    """
    Normalizes raw chat history list of dicts into (User, Assistant) tuples.
    Filters out welcome greetings and invalid payloads.
    """
    formatted: List[Tuple[str, str]] = []
    if not raw_history or not isinstance(raw_history, list):
        return formatted

    last_user_msg = None
    for msg in raw_history:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "").strip() if isinstance(msg.get("content"), str) else ""
        role = msg.get("role")

        if role == "assistant" and ("Welcome to the Visit Ethiopia" in content or "Selam!" in content):
            continue

        if role == "user" and content:
            last_user_msg = content
        elif role == "assistant" and last_user_msg is not None and content:
            formatted.append((last_user_msg, content))
            last_user_msg = None

    return formatted

def stringify_chat_history(pairs: List[Tuple[str, str]]) -> str:
    """Converts (User, Assistant) tuples into clean formatted text for LLM prompts."""
    if not pairs:
        return ""
    lines = []
    for user_msg, ai_msg in pairs:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {ai_msg}")
    return "\n".join(lines)

def build_memory_context(pairs: List[Tuple[str, str]], memory_pairs: int = 3) -> str:
    """Builds compact memory context string from the last N conversation pairs."""
    if not pairs:
        return ""
    recent = pairs[-memory_pairs:]
    return stringify_chat_history(recent)


# ==================== FAST HEURISTIC ROUTER & GREETINGS ====================

def is_greeting(question: str) -> bool:
    """Fast check for standard greeting phrases (0 LLM calls)."""
    q_norm = normalize_text(question)
    greetings = {
        "hi", "hello", "hey", "selam", "good morning", "good afternoon",
        "good evening", "how are you", "hows it going", "yo", "morning", "afternoon"
    }
    return q_norm in greetings

def is_identity_query(question: str) -> bool:
    """Fast check for assistant identity / origin inquiries (0 LLM calls)."""
    q_norm = normalize_text(question)
    identity_triggers = {
        "who are you", "who is this", "what is this", "who am i speaking to",
        "who am i talking to", "who is behind this", "what is your name",
        "whats your name", "introduce yourself", "tell me about yourself",
        "tell me who you are", "who made you", "who built you", "who created you",
        "who developed you", "who is your creator", "what technology are you",
        "what llm are you", "what model are you", "are you google", "are you gemini",
        "are you chatgpt", "are you an ai", "are you a bot", "are you human"
    }
    if any(trigger in q_norm for trigger in identity_triggers):
        return True
    tech_keywords = {"google", "gemini", "openai", "chatgpt", "llama", "claude"}
    words = set(q_norm.split())
    return bool(words & tech_keywords)

def get_greeting_response(question: str = "") -> str:
    """Returns a warm, natural conversational greeting."""
    q_norm = normalize_text(question or "")
    if "selam" in q_norm:
        return "Selam! Welcome to Visit Ethiopia. 🇪🇹 I'm your local travel guide. Are you planning a trip, looking for destination tips, or curious about Ethiopian culture?"
    elif "morning" in q_norm:
        return "Good morning! Welcome! How can I help you explore Ethiopia today?"
    elif "evening" in q_norm:
        return "Good evening! Welcome! What would you like to discover about Ethiopia today?"
    else:
        return "Hello and welcome to Visit Ethiopia! 🇪🇹 I'm your local travel guide. How can I help you plan your journey or answer your travel questions today?"

def get_identity_response() -> str:
    """Returns official identity response in a warm, conversational tone."""
    return "I'm your Visit Ethiopia travel guide! I'm dedicated to helping travelers explore Ethiopia—from ancient historical sites and mountain trekking to traditional culture, food, and travel logistics. What would you like to explore today?"

def is_out_of_scope_query(question: str) -> bool:
    """Fast check for obvious non-Ethiopian coding, stock market, or out-of-scope technical queries (0 LLM calls)."""
    q_norm = normalize_text(question)
    out_triggers = {
        "django", "python code", "write code", "fix my code", "debug code",
        "react app", "javascript code", "stock price", "tesla stock", "bitcoin",
        "crypto price", "fix app", "programming", "code for me", "write a script",
        "fix my django application", "write python code"
    }
    return any(trig in q_norm for trig in out_triggers)

def get_out_of_scope_response() -> str:
    """Returns official concise response for out-of-scope queries without extra footers."""
    return "I am the Visit Ethiopia Travel Assistant, an official digital guide designed specifically to showcase the tourism, culture, and government infrastructure of Ethiopia. 🇪🇹 I cannot assist with software development, coding, or topics outside of Ethiopia."

def check_fast_heuristics(question: str) -> Optional[Dict[str, Any]]:
    """
    Evaluates if query is a greeting, identity, or out-of-scope inquiry.
    Returns response dictionary instantly (0 LLM calls) if matched, otherwise None.
    """
    if is_greeting(question):
        return {
            "answer": get_greeting_response(question),
            "route": "greeting",
            "trace_logs": ["⚡ Fast-routed via greeting heuristic (0 LLM calls)."],
            "docs": [],
            "is_grounded": True,
            "execution_time_sec": 0.0,
        }
    if is_identity_query(question):
        return {
            "answer": get_identity_response(),
            "route": "identity",
            "trace_logs": ["⚡ Fast-routed via identity heuristic (0 LLM calls)."],
            "docs": [],
            "is_grounded": True,
            "execution_time_sec": 0.0,
        }
    if is_out_of_scope_query(question):
        return {
            "answer": get_out_of_scope_response(),
            "route": "out_of_scope",
            "trace_logs": ["⚡ Fast-routed via out-of-scope heuristic (0 LLM calls)."],
            "docs": [],
            "is_grounded": True,
            "execution_time_sec": 0.0,
        }
    return None


# ==================== REAL-TIME CURRENCY EXCHANGE ENGINE ====================

_EXCHANGE_RATE_CACHE = {
    "timestamp": 0.0,
    "rates": {
        "USD": 125.40,
        "EUR": 136.20,
        "GBP": 160.10,
        "CAD": 91.50,
        "AED": 34.14,
    }
}
_CACHE_TTL_SECONDS = 3600  # 1-hour cache TTL to keep requests fast and prevent rate limiting

def get_live_exchange_rates() -> Dict[str, float]:
    """
    Fetches real-time market exchange rates (USD, EUR, GBP, etc. to ETB).
    Uses 1-hour in-memory caching with graceful fallback defaults if API is offline.
    """
    global _EXCHANGE_RATE_CACHE
    now = time.time()

    # Return cached rates if fresh (under 1 hour old)
    if now - _EXCHANGE_RATE_CACHE["timestamp"] < _CACHE_TTL_SECONDS and _EXCHANGE_RATE_CACHE["rates"]:
        return _EXCHANGE_RATE_CACHE["rates"]

    try:
        req = urllib.request.Request(
            "https://open.er-api.com/v6/latest/USD",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                rates = data.get("rates", {})
                etb_rate = rates.get("ETB")
                if etb_rate:
                    # Calculate rates relative to ETB
                    eur_in_usd = rates.get("EUR", 0.92)
                    gbp_in_usd = rates.get("GBP", 0.78)
                    cad_in_usd = rates.get("CAD", 1.37)
                    aed_in_usd = rates.get("AED", 3.67)

                    updated_rates = {
                        "USD": round(float(etb_rate), 2),
                        "EUR": round(float(etb_rate) / float(eur_in_usd), 2) if eur_in_usd else 136.20,
                        "GBP": round(float(etb_rate) / float(gbp_in_usd), 2) if gbp_in_usd else 160.10,
                        "CAD": round(float(etb_rate) / float(cad_in_usd), 2) if cad_in_usd else 91.50,
                        "AED": round(float(etb_rate) / float(aed_in_usd), 2) if aed_in_usd else 34.14,
                    }
                    _EXCHANGE_RATE_CACHE["rates"] = updated_rates
                    _EXCHANGE_RATE_CACHE["timestamp"] = now
                    print(f"💰 [CURRENCY SERVICE] Refreshed live exchange rates: 1 USD = {updated_rates['USD']} ETB", flush=True)
                    return updated_rates
    except Exception as e:
        print(f"⚠️ [CURRENCY SERVICE] Live exchange rate fetch fallback: {e}", flush=True)

    return _EXCHANGE_RATE_CACHE["rates"]

def format_live_currency_context() -> str:
    """
    Generates a live exchange rate context string with current date and rate variation disclaimer.
    """
    rates = get_live_exchange_rates()
    usd = rates.get("USD", 125.40)
    eur = rates.get("EUR", 136.20)
    gbp = rates.get("GBP", 160.10)
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    return (
        f"=== REAL-TIME CURRENCY EXCHANGE RATES (Date: {today_str}) ===\n"
        f"Live Rates: 1 USD = {usd:.2f} ETB | 1 EUR = {eur:.2f} ETB | 1 GBP = {gbp:.2f} ETB\n"
        "(Note: Exchange rates and prices vary daily by bank and official market sources.)\n"
        "INSTRUCTION: Use these exact live exchange rates whenever converting foreign currency figures (USD, EUR, GBP) to Ethiopian Birr (ETB / ብር). Always state the date or note that rates vary daily by bank."
    )

def convert_currency(amount: float, from_curr: str = "USD", to_curr: str = "ETB") -> str:
    """
    Performs dynamic numeric currency conversion using live market exchange rates.
    """
    rates = get_live_exchange_rates()
    curr_clean = from_curr.strip().upper()
    rate = rates.get(curr_clean, rates.get("USD", 125.40))
    converted = amount * rate
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    return f"{amount:,.2f} {curr_clean} is equivalent to {converted:,.2f} ETB (As of {today_str}, Rate: 1 {curr_clean} = {rate:.2f} ETB; note that rates vary daily by bank)."

