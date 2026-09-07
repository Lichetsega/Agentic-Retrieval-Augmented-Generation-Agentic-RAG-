Markdown

\# Visit Ethiopia: RAG Chatbot



A production-grade Retrieval-Augmented Generation (RAG) assistant designed to provide comprehensive, grounded information regarding Ethiopian destinations, cultural landmarks, national services, and travel logistics\[cite: 4, 6]. The platform indexes official federal and regional tourism portals into a persistent Chroma vector collection and leverages hybrid retrieval to serve accurate responses\[cite: 1, 2, 6].



\---



\## Core Capabilities



\* \*\*Hybrid Retrieval (Dense + Sparse):\*\* Combines semantic vector similarity with BM25 keyword scoring and fuzzy query expansion to catch regional terminology and exact landmark names\[cite: 2].

\* \*\*Automated Web Crawler:\*\* Headless Selenium and BeautifulSoup crawler tuned to extract domain-specific content across Ethiopian tourism platforms\[cite: 7].

\* \*\*Intelligent Ingestion Pipeline:\*\* MD5 hash validation avoids duplicate indexing, while an automated orphan-purging routine removes outdated chunks\[cite: 6].

\* \*\*Failover \& Multi-Key Management:\*\* Round-robin Google Gemini key rotation with fallback handling to local Ollama execution and safe-mode failover\[cite: 6, 8].

\* \*\*Multi-Channel Serving:\*\* Provides both a Streamlit interactive chat application and a RESTful Flask API server with rate-limiting, session memory, and response caching\[cite: 9, 10].



\---



\## Architecture Flow



```text

User Query

&#x20;   │

&#x20;   ▼

\[Streamlit App (app.py) / Flask API (api\_server.py)]

&#x20;   │

&#x20;   ├── In-Memory Query Cache Check (Hit -> Instant Return)

&#x20;   │

&#x20;   └── Query Cache Miss

&#x20;            │

&#x20;            ├── \[Query Expansion \& Intent Parsing]

&#x20;            │

&#x20;            ├── \[Hybrid Search Engine]

&#x20;            │        ├── Vector Semantic Retrieval (ChromaDB)

&#x20;            │        └── Keyword Search (BM25 + Fuzzy Match)

&#x20;            │

&#x20;            ▼

&#x20;     \[Context Assembly \& Grounding]

&#x20;            │

&#x20;            ▼

&#x20;      \[LLM Generation (Gemini Primary / Ollama Fallback)]

&#x20;            │

&#x20;            ▼

&#x20;       Verified Response

Repository Structure

Plaintext

RAG-Chatbot/

├── RAG-Chatbot-from-web-data/

│   ├── chatbot/

│   │   ├── api\_key\_manager.py     # Gemini key rotation and cooldown management

│   │   ├── api\_server.py          # Flask REST API backend (/ask, /ready, /admin)

│   │   ├── app.py                 # Streamlit interactive UI application

│   │   ├── demo.ipynb             # Interactive testing and indexing notebook

│   │   ├── hybrid\_retriever.py    # Vector + BM25 keyword search engine

│   │   ├── ingest.py              # Batch ingestion CLI runner

│   │   ├── prompt.py              # Grounding prompts and personality templates

│   │   ├── text\_to\_doc.py         # Text cleaner, chunker, and LangChain Document creator

│   │   ├── utils.py               # Core pipeline orchestrator and Chroma connection

│   │   └── web\_crawler.py         # Selenium web scraping utility

│   ├── generate\_pdf\_manual.py     # User manual PDF generation script

│   ├── requirements.txt           # Python dependencies

│   └── SETUP\_GUIDE.md             # Detailed deployment documentation

├── generate\_pptx.py               # Presentation deck generator

├── Visit\_Ethiopia\_Agentic\_RAG\_Presentation.pptx

├── .gitignore

└── README.md

Setup \& Installation

1\. Clone the Repository

Bash

git clone \[https://github.com/Lichetsega/VISIT-ETHIOPIA-RAG-CHATBOT-.git](https://github.com/Lichetsega/VISIT-ETHIOPIA-RAG-CHATBOT-.git)

cd VISIT-ETHIOPIA-RAG-CHATBOT-

2\. Configure Environment Variables

Create a .env file inside RAG-Chatbot-from-web-data/:



Code snippet

GOOGLE\_API\_KEY\_1="your\_gemini\_api\_key\_1"

GOOGLE\_API\_KEY\_2="your\_gemini\_api\_key\_2"

CHATBOT\_API\_KEY="your\_optional\_service\_auth\_key"

3\. Run the Services

From the RAG-Chatbot-from-web-data/chatbot directory:



Start the Flask REST API Server:



Bash

python api\_server.py





Start the Streamlit User Interface:



Bash

streamlit run app.py





Ingest Data Sources Manually:



Bash

python ingest.py



\---



\### Step 3: Commit and Push to GitHub



Run these commands in your Command Prompt (`C:\\Users\\liche\\Desktop\\RAG-Chatbot>`):



```cmd

git add .

git status

Verify that no .env files appear in the staged list. Then commit and push:



DOS

git commit -m "docs: restructure repository layout and update comprehensive README"

git push origin main

