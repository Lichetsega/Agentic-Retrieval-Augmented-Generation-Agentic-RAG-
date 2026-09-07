# Step-by-Step Setup, Ingestion, Re-indexing & Execution Manual
## Visit Ethiopia Agentic RAG Chatbot System (Zero-to-Hero Guide)

This manual provides **complete, step-by-step instructions** to install, configure, ingest data into, re-index, and run the **Visit Ethiopia RAG Chatbot System**.

---

## 1. Project Overview & Architecture (What is this project?)

The **Visit Ethiopia RAG Chatbot** is an AI travel assistant designed to answer questions about Ethiopian tourism, culture, visa rules, and history using data crawled directly from official government and regional tourism portals.

### How it works under the hood:
1. **Web Crawler & Chunker**: Selenium (headless Chrome) and BeautifulSoup crawl websites like `visitethiopia.et`, clean the text, and split it into readable chunks.
2. **Vector Store & Hybrid Search**: HuggingFace sentence transformer embeddings (`all-MiniLM-L6-v2`) turn text chunks into numbers and store them in **ChromaDB**. Search uses a hybrid algorithm combining **BM25 keyword search** and **vector similarity**.
3. **LLM Engine**: Google Gemini (with automatic API key rotation) generates answers based *only* on retrieved website context. If offline, it falls back to a local **Ollama** model (`mistral:7b`).
4. **Services**:
   - **Flask API Server** (`api_server.py`): Serves REST requests on port `5000` with authentication, rate limiting, and an administrative `/admin/reindex` endpoint.
   - **Streamlit Web UI** (`app.py`): Interactive chat interface running on port `8501`.

```
[ Official Tourism Websites ] ──> [ Web Crawler ] ──> [ Text Chunker ] ──> [ HuggingFace Embeddings ]
                                                                                   │
                                                                                   ▼
[ Streamlit UI / Flask API ] <── [ Gemini LLM / Ollama ] <── [ Chroma Vector DB + Hybrid BM25 ]
```

---

## 2. System Prerequisites

Before starting, make sure the following software is installed on your computer:

- **Python**: Python 3.10.x (Recommended: Python 3.10.11)
- **Google Chrome Browser**: Required by Selenium for crawling JavaScript-enabled web pages
- **Git**: For code version control
- **Google Gemini API Key(s)**: Free from Google AI Studio
- *(Optional)* **Ollama**: Required only if you want to run local offline LLM models (e.g. `mistral:7b`)

---

## 3. Step-by-Step Installation Guide

### Step 1: Open Terminal & Navigate to Project Folder

Open PowerShell or Command Prompt (Windows) or Terminal (Linux/macOS):

```bash
cd "c:\Users\liche\Desktop\RAG-Chatbot\RAG-Chatbot-from-web-data"
```

---

### Step 2: Create a Python Virtual Environment

Create an isolated environment to avoid library conflicts:

**On Windows (PowerShell / CMD):**
```powershell
python -m venv venv
```

**On Linux / macOS:**
```bash
python3 -m venv venv
```

---

### Step 3: Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```
*(Note: If PowerShell shows a script error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
source venv/bin/activate
```

*(You will see `(venv)` at the beginning of your terminal prompt).*

---

### Step 4: Install All Required Libraries (Direct Commands)

Instead of using `requirements.txt`, run the two standard `pip install` commands below to install all project dependencies:

#### Command 1: Install Core Data Science & Processing Libraries
```bash
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn tqdm
```

#### Command 2: Install LangChain, Gemini, ChromaDB, Web Crawling & Web Server Libraries
```bash
pip install -U langchain langchain-core langchain-community langchain-google-genai langchain-chroma google-generativeai rank-bm25 flask flask-cors gunicorn chromadb sentence-transformers beautifulsoup4 bs4 html2text requests selenium webdriver-manager python-dotenv tqdm markdown streamlit pypdf typing-extensions pydantic protobuf reportlab
```

#### Why these libraries are needed:
* `langchain`, `langchain-community`, `langchain-google-genai`: RAG orchestration & Gemini LLM integration.
* `google-generativeai`: Official Google Gemini API SDK.
* `chromadb`, `langchain-chroma`, `sentence-transformers`: Local vector database & semantic embeddings (`all-MiniLM-L6-v2`).
* `rank-bm25`: Keyword relevance ranking for hybrid retriever.
* `selenium`, `webdriver-manager`, `beautifulsoup4`, `html2text`: Headless browser web crawling and HTML parsing.
* `flask`, `flask-cors`, `gunicorn`: REST API web server and CORS handling.
* `streamlit`: User-friendly graphical web chat application.
* `python-dotenv`: Loads API keys from `.env` file automatically.
* `reportlab`: Automates generation of downloadable PDF manuals.

---

## 4. Environment Configuration (`.env` Setup)

Create a configuration file named `.env` inside the `RAG-Chatbot-from-web-data/chatbot` directory (and root directory):

```ini
# =====================================================================
# Google Gemini API Keys (Multi-Key Automatic Failover & Key Rotation)
# =====================================================================
GOOGLE_API_KEY_1=AIzaSyACRPPvdOptcazzjUmVCJV2Sd2x88mLBWs
GOOGLE_API_KEY_2=AIzaSyChm5oonrV6RU3_ZqoOtL4JpKteLeN3w_Q

# =====================================================================
# Local Ollama LLM Fallback Configuration
# =====================================================================
OLLAMA_MODEL=mistral:7b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MAX_FAILS=1
OLLAMA_COOLDOWN_SECONDS=300

# =====================================================================
# API Security & Server Settings
# =====================================================================
CHATBOT_API_KEY=LYOqxG-9KM8CcfOtZF9iWr2h0lOvui0OYPimW_0g9OUfHvdQMsh8dbRqbbmzWQCe
REQUIRE_API_KEY=false
ALLOWED_ORIGINS=https://visitethiopia.et,http://localhost:8501
REQUEST_TIMEOUT_SECONDS=60
RATE_LIMIT_REQUESTS=30
RATE_LIMIT_WINDOW_SECONDS=60
RESPONSE_CACHE_TTL_SECONDS=180

# Disable Telemetry & Set Automated Crawl Interval
ANONYMIZED_TELEMETRY=False
CRAWL_INTERVAL_HOURS=24
```

---

## 5. Data Ingestion & Vector Indexing (3 Methods)

Before asking questions, website data must be crawled and stored in ChromaDB.

First navigate to the `chatbot` folder:
```bash
cd chatbot
```

### Method 1: One-Off Manual CLI Ingestion (`ingest.py`)
Run this command to immediately crawl default tourism portals and populate `chatbot/data/chroma`:
```bash
python ingest.py
```
To crawl specific websites:
```bash
python ingest.py --url https://visitethiopia.et/ --url https://visitoromia.org/
```

---

### Method 2: Automated Ethiopian Midnight Background Scheduler (`scheduler.py`)
The Flask REST API server (`api_server.py`) automatically launches the background scheduler daemon thread on startup, targeting **00:00:00 Ethiopian Midnight (`Africa/Addis_Ababa` / UTC+3)** every night.

You can also run the scheduler manually in standalone mode:
```bash
python scheduler.py                       # Midnight schedule, runs initial crawl on startup
python scheduler.py --skip-initial         # Midnight schedule, sleeps until first midnight
python scheduler.py --mode interval        # Uses fixed interval hours (e.g. 24h)
```

---

### Method 3: On-Demand API Re-indexing Endpoint (`/admin/reindex`)
When the Flask REST API server (`api_server.py`) is running, you can trigger data re-indexing remotely over HTTP **without restarting the server**.

#### Endpoint Specification:
* **URL**: `POST http://localhost:5000/admin/reindex`
* **Headers**: `Content-Type: application/json` and `X-API-Key: <CHATBOT_API_KEY>` (if API key requirement is enabled).
* **Payload (JSON)**:
  ```json
  {
    "urls": [
      "https://visitethiopia.et/",
      "https://visitoromia.org/"
    ]
  }
  ```

#### Example cURL Command:
```bash
curl -X POST http://localhost:5000/admin/reindex \
  -H "Content-Type: application/json" \
  -H "X-API-Key: LYOqxG-9KM8CcfOtZF9iWr2h0lOvui0OYPimW_0g9OUfHvdQMsh8dbRqbbmzWQCe" \
  -d "{\"urls\": [\"https://visitethiopia.et/\"]}"
```

#### Expected Success Response:
```json
{
  "status": "success",
  "message": "Reindexing finished",
  "success_sources": 1,
  "failed_sources": 0
}
```

---

## 6. Starting & Running the Chatbot Services

Ensure your virtual environment is active and you are inside `RAG-Chatbot-from-web-data/chatbot`.

### Service 1: Flask REST API Backend Server (`api_server.py`)
Provides REST API endpoints for frontend apps, web widgets, and administrative triggers.

Launch the server:
```bash
python api_server.py
```
* **Server Address**: `http://localhost:5000`
* **Health Check**: `http://localhost:5000/health`
* **Readiness Check**: `http://localhost:5000/ready`
* **Chat Endpoint**: `POST http://localhost:5000/ask`
* **Re-index Endpoint**: `POST http://localhost:5000/admin/reindex`

---

### Service 2: Streamlit Interactive Web Interface (`app.py`)
Provides a user-friendly chat interface in the browser.

Launch Streamlit:
```bash
streamlit run app.py
```
* **Browser Access**: Opens automatically at `http://localhost:8501`

---

## 7. Generating the Downloadable PDF Manual

To generate a PDF file (`Visit_Ethiopia_RAG_Chatbot_Setup_Manual.pdf`):

Run the Python PDF generator script:
```bash
python generate_pdf_manual.py
```

The PDF will be created directly in your folder at:
`c:\Users\liche\Desktop\RAG-Chatbot\RAG-Chatbot-from-web-data\Visit_Ethiopia_RAG_Chatbot_Setup_Manual.pdf`

---

## 8. Verification & Summary Checklist

| Step | Action | Execution Command | Expected Output |
| :--- | :--- | :--- | :--- |
| **1** | Activate Virtual Env | `.\venv\Scripts\Activate.ps1` | `(venv)` prompt visible |
| **2** | Install Packages | `pip install -U ...` | Packages installed successfully |
| **3** | Create `.env` | Copy template into `chatbot/.env` | `.env` file populated |
| **4** | Index Data | `python ingest.py` | `chatbot/data/chroma/` created |
| **5** | API Re-index | `POST /admin/reindex` | Returns status success JSON |
| **6** | Launch API Server | `python api_server.py` | HTTP 200 on `/health` endpoint |
| **7** | Launch Streamlit | `streamlit run app.py` | Browser opens at `localhost:8501` |
| **8** | Generate PDF | `python generate_pdf_manual.py` | PDF manual file generated |
