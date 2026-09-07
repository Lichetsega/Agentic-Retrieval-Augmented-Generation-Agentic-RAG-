"""
Visit Ethiopia Agentic RAG Chatbot - PDF Setup Manual Generator
This script automatically generates `Visit_Ethiopia_RAG_Chatbot_Setup_Manual.pdf`.
"""

import sys
import os

# Auto-install reportlab if missing
try:
    import reportlab
except ImportError:
    print("Installing reportlab for PDF generation...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    import reportlab

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_pdf_manual(output_filename="Visit_Ethiopia_RAG_Chatbot_Setup_Manual.pdf"):
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F172A")    # Dark Slate
    SECONDARY = colors.HexColor("#0284C7")  # Sky Blue
    ACCENT = colors.HexColor("#D97706")     # Amber Gold
    TEXT_DARK = colors.HexColor("#1E293B")  # Text Main
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Light Gray
    CODE_BG = colors.HexColor("#1E293B")    # Code block background
    CODE_TEXT = colors.HexColor("#38BDF8")  # Code text

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        fontName='Courier',
        fontSize=9,
        leading=12,
        textColor=CODE_TEXT,
        spaceBefore=4,
        spaceAfter=4
    )

    story = []

    # Title Banner
    story.append(Paragraph("Visit Ethiopia Agentic RAG Chatbot", title_style))
    story.append(Paragraph("Complete Beginner-to-Pro Manual: Setup, Dependencies, Ingestion, Re-indexing & Serving", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceAfter=15))

    # Section 1: Overview
    story.append(Paragraph("1. Project Overview & Architecture", h1_style))
    overview_text = (
        "The <b>Visit Ethiopia RAG Chatbot</b> is an autonomous artificial intelligence assistant designed to answer "
        "travel queries using data crawled directly from official Ethiopian national and regional tourism portals. "
        "It uses a Retrieval-Augmented Generation (RAG) pipeline combining web crawlers, HuggingFace sentence embeddings, "
        "a Chroma vector database, hybrid BM25 search, Google Gemini LLMs with automatic key rotation, and local Ollama fallback models."
    )
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 8))

    # Section 2: Prerequisites
    story.append(Paragraph("2. System Prerequisites", h1_style))
    prereqs = [
        "<b>Python 3.10+</b> (Recommended: Python 3.10.11)",
        "<b>Google Chrome Browser</b> (Required for Selenium dynamic web crawler)",
        "<b>Git</b> (For project version control)",
        "<b>Google Gemini API Key(s)</b> (Free from Google AI Studio)",
        "<b>(Optional) Ollama</b> (If using local offline LLM like <i>mistral:7b</i>)"
    ]
    for p in prereqs:
        story.append(Paragraph(f"• {p}", bullet_style))
    story.append(Spacer(1, 8))

    # Section 3: Virtual Environment Setup
    story.append(Paragraph("3. Step-by-Step Environment Setup", h1_style))
    story.append(Paragraph("Open terminal and navigate to the project directory:", body_style))
    
    def make_code_box(code_text):
        p = Paragraph(code_text.replace('\n', '<br/>').replace(' ', '&nbsp;'), code_style)
        t = Table([[p]], colWidths=[520])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), CODE_BG),
            ('PADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('TOPPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    story.append(make_code_box('cd "c:\\Users\\liche\\Desktop\\RAG-Chatbot\\RAG-Chatbot-from-web-data"'))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Create Virtual Environment:</b>", body_style))
    story.append(make_code_box('python -m venv venv'))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Activate Virtual Environment:</b>", body_style))
    story.append(make_code_box(
        '# Windows (PowerShell)\n.\\venv\\Scripts\\Activate.ps1\n\n'
        '# Windows (Command Prompt)\nvenv\\Scripts\\activate.bat\n\n'
        '# Linux / macOS\nsource venv/bin/activate'
    ))
    story.append(Spacer(1, 10))

    # Section 4: Direct Package Installation (NO requirements.txt)
    story.append(Paragraph("4. Installing Required Libraries (Direct Commands)", h1_style))
    story.append(Paragraph(
        "Instead of using <code>requirements.txt</code>, install all necessary libraries directly using the two standard <code>pip install</code> commands below:",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Command 1: Core Data Science & Utility Packages</b>", h2_style))
    story.append(make_code_box('pip install numpy pandas scikit-learn scikit-image matplotlib seaborn tqdm'))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Command 2: Core LangChain, Gemini, RAG, Web & Serving Stack</b>", h2_style))
    story.append(make_code_box(
        'pip install -U langchain langchain-core langchain-community langchain-google-genai '
        'langchain-chroma google-generativeai rank-bm25 flask flask-cors gunicorn chromadb '
        'sentence-transformers beautifulsoup4 bs4 html2text requests selenium webdriver-manager '
        'python-dotenv tqdm markdown streamlit pypdf typing-extensions pydantic protobuf'
    ))
    story.append(Spacer(1, 10))

    # Section 5: Configuration (.env)
    story.append(Paragraph("5. Environment Configuration (.env Setup)", h1_style))
    story.append(Paragraph(
        "Create a file named <code>.env</code> inside the <code>RAG-Chatbot-from-web-data/chatbot</code> directory with the following contents:",
        body_style
    ))
    env_content = (
        "# Google Gemini API Keys (Supports multi-key round-robin rotation)\n"
        "GOOGLE_API_KEY_1=AIzaSyACRPPvdOptcazzjUmVCJV2Sd2x88mLBWs\n"
        "GOOGLE_API_KEY_2=AIzaSyChm5oonrV6RU3_ZqoOtL4JpKteLeN3w_Q\n\n"
        "# Local Ollama LLM Fallback\n"
        "OLLAMA_MODEL=mistral:7b\n"
        "OLLAMA_BASE_URL=http://localhost:11434\n\n"
        "# API Security & Limits\n"
        "CHATBOT_API_KEY=LYOqxG-9KM8CcfOtZF9iWr2h0lOvui0OYPimW_0g9OUfHvdQMsh8dbRqbbmzWQCe\n"
        "REQUIRE_API_KEY=false\n"
        "ALLOWED_ORIGINS=https://visitethiopia.et,http://localhost:8501\n"
        "REQUEST_TIMEOUT_SECONDS=60\n"
        "ANONYMIZED_TELEMETRY=False\n"
        "CRAWL_INTERVAL_HOURS=24"
    )
    story.append(make_code_box(env_content))
    story.append(Spacer(1, 10))

    # Section 6: Ingestion & Re-indexing (INCLUDING /admin/reindex endpoint)
    story.append(Paragraph("6. Data Ingestion & Vector Indexing (3 Methods)", h1_style))
    story.append(Paragraph(
        "First navigate into the <code>chatbot</code> module directory: <code>cd chatbot</code>",
        body_style
    ))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Method A: One-Off Manual CLI Ingestion</b>", h2_style))
    story.append(Paragraph("Crawls configured official websites and builds the ChromaDB index in <code>chatbot/data/chroma</code>:", body_style))
    story.append(make_code_box('python ingest.py'))
    story.append(Spacer(1, 4))
    story.append(Paragraph("To target specific websites:", body_style))
    story.append(make_code_box('python ingest.py --url https://visitethiopia.et/ --url https://visitoromia.org/'))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Method B: 24-Hour Background Automated Scheduler</b>", h2_style))
    story.append(Paragraph("Runs a background daemon that updates vector embeddings automatically every 24 hours:", body_style))
    story.append(make_code_box('python scheduler.py'))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Method C: On-Demand API Re-indexing Endpoint (/admin/reindex)</b>", h2_style))
    story.append(Paragraph(
        "When the Flask API server is running, administrators can trigger website crawling and vector re-indexing on demand "
        "via HTTP POST request to the <code>/admin/reindex</code> endpoint:",
        body_style
    ))
    story.append(make_code_box(
        '# Trigger Re-indexing via cURL command:\n'
        'curl -X POST http://localhost:5000/admin/reindex \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -H "X-API-Key: LYOqxG-9KM8CcfOtZF9iWr2h0lOvui0OYPimW_0g9OUfHvdQMsh8dbRqbbmzWQCe" \\\n'
        '  -d \'{"urls": ["https://visitethiopia.et/"]}\'\n\n'
        '# Expected HTTP 200 JSON Response:\n'
        '{\n'
        '  "status": "success",\n'
        '  "message": "Reindexing finished",\n'
        '  "success_sources": 1,\n'
        '  "failed_sources": 0\n'
        '}'
    ))
    story.append(Spacer(1, 10))

    # Section 7: Starting and Running the Services
    story.append(Paragraph("7. Starting and Running the Services", h1_style))
    story.append(Paragraph("Make sure you are inside <code>chatbot/</code> directory with virtual environment active.", body_style))

    story.append(Paragraph("<b>Start Service 1: Flask REST API Backend Server</b>", h2_style))
    story.append(make_code_box('python api_server.py'))
    story.append(Paragraph("• Server URL: <code>http://localhost:5000</code><br/>• Health Check: <code>http://localhost:5000/health</code><br/>• Question Answering Endpoint: <code>POST http://localhost:5000/ask</code>", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Start Service 2: Streamlit Interactive Web Application</b>", h2_style))
    story.append(make_code_box('streamlit run app.py'))
    story.append(Paragraph("• Browser Access: Opens automatically at <code>http://localhost:8501</code>", body_style))
    story.append(Spacer(1, 10))

    # Section 8: Testing & Verification Checklist Table
    story.append(Paragraph("8. Testing & Verification Summary", h1_style))
    
    table_data = [
        [Paragraph("<b>Step</b>", body_style), Paragraph("<b>Action / Component</b>", body_style), Paragraph("<b>Execution Command</b>", body_style), Paragraph("<b>Success Output</b>", body_style)],
        [Paragraph("1", body_style), Paragraph("Activate Virtual Env", body_style), Paragraph("<code>.\\venv\\Scripts\\Activate.ps1</code>", body_style), Paragraph("<code>(venv)</code> prompt visible", body_style)],
        [Paragraph("2", body_style), Paragraph("Install Libraries", body_style), Paragraph("<code>pip install -U ...</code>", body_style), Paragraph("Packages installed cleanly", body_style)],
        [Paragraph("3", body_style), Paragraph("Data Ingestion", body_style), Paragraph("<code>python ingest.py</code>", body_style), Paragraph("Chroma database created", body_style)],
        [Paragraph("4", body_style), Paragraph("API Re-indexing", body_style), Paragraph("<code>POST /admin/reindex</code>", body_style), Paragraph("Status success JSON", body_style)],
        [Paragraph("5", body_style), Paragraph("Flask API Backend", body_style), Paragraph("<code>python api_server.py</code>", body_style), Paragraph("HTTP 200 on /health", body_style)],
        [Paragraph("6", body_style), Paragraph("Streamlit Web UI", body_style), Paragraph("<code>streamlit run app.py</code>", body_style), Paragraph("Opens at localhost:8501", body_style)]
    ]

    t_summary = Table(table_data, colWidths=[40, 130, 170, 180])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SECONDARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_summary)

    # Build Document
    doc.build(story)
    print(f"✅ Successfully generated PDF manual at: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    create_pdf_manual()
