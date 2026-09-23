# 🇪🇹 Agentic RAG Assistant

A multi-agent, Retrieval-Augmented Generation (RAG) assistant designed for Ethiopian tourism, cultural heritage, and travel logistics. Built with LangChain, ChromaDB, Google Gemini, and ChatOllama, the system features autonomous supervisor routing, domain-specialized sub-agents, multi-query hybrid retrieval, real-time tool augmentation, and empirical faithfulness verification.

---

## 🏛️ System Architecture

```text
                                  User Query
                                       │
                                       ▼
                       [Streamlit UI / Flask REST API]
                                       │
                         [Exact & Semantic Query Cache]
                          ├── Hit (Similarity >= 0.82) ──► Instant Response (0 API calls)
                          │
                          └── Miss
                                │
                                ▼
                   [Fused Supervisor Planner]
        (Intent Classification + Pronoun Resolution + Sentiment + Multi-Query)
                                │
               ┌────────────────┼────────────────┐
               │                │                │
          [Greeting]       [Identity]      [Substantive RAG]
          (0 LLM calls)   (0 LLM calls)          │
                                                 ▼
                                     [Sub-Agent Domain Router]
                          ┌──────────────┬───────┴──────┬──────────────┐
                          │              │              │              │
                     [Itinerary]     [Culture]     [Logistics]     [General]
                          └──────────────┬───────┬──────┴──────────────┘
                                         │
                                         ▼
                             [Hybrid Retrieval Engine]
                           ├── Dense Semantic (ChromaDB)
                           └── Sparse Keyword (BM25 + Levenshtein Fuzzy)
                                         │
                                         ▼
                            [Batch Document Evaluator]
                                         │
                                         ▼
                            [Live External Tool Injection]
                                         │
                                         ▼
                       [Domain Sub-Agent Context Synthesis]
                         (Gemini Primary ──► Ollama Fallback)
                                         │
                                         ▼
                            [Faithfulness Verification]
                                         │
                                         ▼
                                  Final Answer

```
# ⚡ Core Capabilities

```
Fused Supervisor Planner: Combines intent detection, conversation memory rephrasing, user sentiment analysis, and search query expansion into a single low-latency call.

Specialized Expert Sub-Agents:

Itinerary & Route Planning Expert: Separates logistics from activities and accounts for realistic overland travel across Ethiopian terrain.

Culture & Heritage Expert: Covers UNESCO heritage sites, festivals (Timkat, Meskel), and Ethiopian cuisine (Injera, coffee ceremonies).

Logistics & Visa Expert: Handles e-Visa rules, domestic flight logistics (Ethiopian Airlines), safety advisories, and currency conversions.

Hybrid Search with Fuzzy Vocabulary Expansion: Merges dense vector retrieval with BM25 keyword matching and Levenshtein distance expansion to reliably match localized phonetic spellings (e.g., Wenchi / Wonchi).

Live Tool Augmentation: Connects to real-time currency exchange APIs to convert foreign monetary figures ($ USD, € EUR) to Ethiopian Birr (ETB) with daily market disclaimers.

Self-Healing Key Manager: Round-Robin Gemini key rotation that distinguishes client input errors (bad requests, safety blocks) from true 429 quota exhaustion to prevent unnecessary key cooldowns.

Two-Tier Caching: Exact hash and sequence-matching semantic cache yielding instant responses with 0 API calls for repeated or semantically equivalent questions.

Automated Crawler Scheduler: Background daemon running on East Africa Time (EAT / UTC+3) that re-indexes official national and regional tourism portals with content hash diffing and orphan chunk purging.

```

# 📂 Repository Layout

```text
RAG-Chatbot/
├── RAG-Chatbot-from-web-data/
│   ├── chatbot/
│   │   ├── api_key_manager.py     # Gemini key rotation, error classification & cooldowns
│   │   ├── api_server.py          # Production Flask REST API (/ask, /ready, /admin)
│   │   ├── app.py                 # Streamlit UI with agent execution traces
│   │   ├── core_utils.py          # Shared singletons, currency conversion, text cleaners
│   │   ├── demo.ipynb             # Interactive testing and query demo notebook
│   │   ├── hybrid_retriever.py    # Hybrid BM25 + Vector search with fuzzy expansion
│   │   ├── ingest.py              # CLI batch ingestion runner with crawl metadata logging
│   │   ├── langchain_agent.py     # Supervisor planner, document grading & faithfulness check
│   │   ├── prompt.py              # System prompts and supervisory instructions
│   │   ├── query_cache.py         # Exact and semantic similarity cache
│   │   ├── scheduler.py           # Background web crawler scheduler (00:00:00 EAT)
│   │   ├── specialized_agents.py  # Domain sub-agents (Itinerary, Culture, Logistics)
│   │   ├── text_to_doc.py         # Scraped text normalizer and recursive chunker
│   │   ├── utils.py               # Core pipeline bridge and model fallback orchestrator
│   │   └── web_crawler.py         # Headless Selenium crawler for tourism bureaus
│   ├── generate_pdf_manual.py     # Architecture documentation generator
│   ├── requirements.txt           # Python package dependencies
│   └── SETUP_GUIDE.md             # Complete deployment documentation
├── generate_pptx.py               # Project slide deck generator
├── Visit_Ethiopia_Agentic_RAG_Presentation.pptx
├── .gitignore                     # Git tracking exclusions
└── README.md                      # Project documentation
```
# 🚀 Quickstart & Setup

1. Clone the Repository
   
```
        Bash
        
        git clone https://github.com/Lichetsega/Agentic-Retrieval-Augmented-Generation-Agentic-RAG-.git
        
        cd RAG-CHATBOT/RAG-Chatbot-from-web-data
```

2. Configure Environment Variables
   
```
        Create a .env file inside RAG-Chatbot-from-web-data/:
         and inside the .env file add your API keys and the other mentioned files
         
        Google Gemini API Keys (Supports multiple keys for automatic failover)
        
        GOOGLE_API_KEY_1="your_gemini_api_key_1"
        
        GOOGLE_API_KEY_2="your_gemini_api_key_2"
        
        Ollama Fallback (Optional local fallback)
        
        OLLAMA_MODEL="llama3.1:8b-instruct-q4_K_M"
        
        OLLAMA_BASE_URL="http://localhost:11434"
        
        API Server Security
        
        CHATBOT_API_KEY="your_api_auth_key"
        
        REQUIRE_API_KEY="false"
```

3. Install Dependencies
   
```
        Bash
        cd RAG-Chatbot-from-web-data
        pip install -r requirements.txt
```
4. Ingest Official Data Sources
   
```
        Bash
        cd chatbot
        python ingest.py
```
5. Launch the Interfaces
   
```
        Bash
        python api_server.py
        Standalone Midnight Web Crawler Scheduler:
        
        Interactive Streamlit Web UI:
        
        Bash
        streamlit run app.py
        Production Flask REST API:
        
        Bash
        python scheduler.py --mode midnight
```  
---

