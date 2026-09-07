"""
Visit Ethiopia - AI Travel Assistant
Main Streamlit Application
"""

# ==================== IMPORTS ====================
import streamlit as st # Streamlit framework for building the web interface
import requests
import sys  # System-specific parameters and functions
import os  # Operating system interfaces (file paths, environment variables)
from datetime import datetime  # For timestamping chat messages
from typing import Optional  # Type hinting for better code clarity

# Add the current directory to Python's module search path
# This allows imports from local modules like 'utils'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file (contains API keys)
from dotenv import load_dotenv
load_dotenv()

# Import the core chatbot functionality from our custom modules
# get_response: Main function that processes user questions and returns answers
# get_chroma_client: Returns connection to our vector database (where tourism data is stored)
from utils import get_response, get_chroma_client, store_docs

SERVER_URL = "http://172.21.22.33:5000/ask"

import requests

def get_remote_response(prompt):
    """Calls the Flask API server to get RAG-based answers"""
    API_URL = "http://172.21.22.33:5000/ask"

    clean_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    payload = {
        "question": prompt,
        "session_id": st.session_state.get("session_id", "default-user"),
        "chat_history": clean_history
    }

    try:
        # Use a timeout so the UI doesn't hang forever if the server is down
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()  # Returns the dict with 'answer', 'sources', etc.
        else:
            return None
    except Exception as e:
        print(f"Connection Error: {e}")
        return None
# ==================== PAGE CONFIGURATION ====================
# THIS MUST BE THE FIRST STREAMLIT COMMAND
# Sets up the browser tab title, icon, layout width, and sidebar default state
st.set_page_config(
    page_title="Visit Ethiopia AI Travel Assistant",  # Shows in browser tab
    page_icon="🤖",  # Emoji icon in browser tab
    layout="wide",  # Uses full screen width instead of centered narrow layout
    initial_sidebar_state="expanded"  # Sidebar starts open by default
)

# ==================== CUSTOM CSS STYLING ====================
# This block injects custom CSS to make the app look professional and branded
# All styling is applied globally to the entire application
st.markdown("""
<style>
    /* Main header - green gradient background matching Ethiopian flag colors */
    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Header text styling */
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Chat message bubble styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* Sidebar background color */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* White cards in sidebar with subtle shadow */
    .sidebar-section {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Section headers in sidebar */
    .sidebar-section h3 {
        color: #1a472a;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Primary button styling - green rounded buttons */
    .stButton > button {
        background-color: #2d6a4f;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    /* Button hover effect - darker green and slight lift */
    .stButton > button:hover {
        background-color: #1a472a;
        color: white;
        transform: translateY(-1px);
    }
    
    /* Suggestion button styling - light green background */
    .suggestion-btn {
        background-color: #e8f5e9 !important;
        color: #1a472a !important;
        border: 1px solid #2d6a4f !important;
        margin: 0.25rem 0 !important;
    }
    
    .suggestion-btn:hover {
        background-color: #2d6a4f !important;
        color: white !important;
    }
    
    /* Footer styling - centered with border on top */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        color: #666;
        font-size: 0.85rem;
        border-top: 1px solid #ddd;
    }
    
    /* Status indicator colors */
    .status-online {
        color: #10b981;  /* Green for online */
        font-weight: bold;
    }
    
    .status-limited {
        color: #f59e0b;  /* Orange for limited mode */
        font-weight: bold;
    }
    
    /* Welcome message card - purple gradient */
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True allows raw HTML/CSS injection

# ==================== CONSTANTS ====================
# These are pre-defined texts that are passed to the AI for context
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

The platform is also a unified travel assistant providing data from federal and regional 
tourism bureaus including Oromia, Amhara, Tigray, Sidama, and South Ethiopia.

Overall, Visit Ethiopia serves as a comprehensive guide for exploring the country's destinations, 
culture, heritage, and travel experiences.
"""

CONTACT_INFO = """
Official Website: https://visitethiopia.et/

Visit Ethiopia primarily provides informational content to help travelers explore the country. 
For detailed travel arrangements, bookings, or inquiries, users are advised to consult local 
tour operators, travel agencies, or accommodation providers. The official website serves as 
the main source for up-to-date tourism information.
"""

# ==================== HELPER FUNCTIONS ====================
def is_llm_available() -> bool:
    """Check if the AI language model (Gemini) is available"""
    try:
        from utils import LLM_AVAILABLE  # Try to import the global flag
        return LLM_AVAILABLE
    except:
        return True  # Default to True if can't check

def get_document_count() -> tuple[int | None, str | None]:
    """Return (count, error). count is None when DB is not accessible."""
    try:
        db = get_chroma_client()  # Get database connection
        return db._collection.count(), None  # Count documents in the collection
    except Exception as e:
        return None, str(e)

def ingest_default_sources() -> tuple[int, int]:
    """Index default tourism sources into Chroma and return (success, failed)."""
    source_urls = [
        "https://visitethiopia.et/",
        "https://visitoromia.org/",
        "https://visitamhara.travel/",
        "https://tourismtigrai.com/",
        "https://visitsidama.travel/",
        "https://visitsouthethiopia.et/",
    ]

    success = 0
    failed = 0
    for url in source_urls:
        try:
            if store_docs(url):
                success += 1
            else:
                failed += 1
        except Exception:
            failed += 1
    return success, failed

def init_session_state():
    """Initialize Streamlit's session state variables
    Session state persists data across reruns of the app"""

    # Initialize chat message history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",  # Who sent the message (user or assistant)
                "content": """🇪🇹 **Selam! Welcome to the Visit Ethiopia Travel Assistant!**

I'm here to help you explore the beautiful Land of Origins. I can assist you with:

**What would you like to know about Ethiopia today?** ✨""",
                "timestamp": datetime.now()  # When the message was sent
            }
        ]

    # Initialize conversation counter to track number of exchanges
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

def process_user_input(user_input: str):
    """Process what the user typed and generate a response

    Args:
        user_input: The text the user typed in the chat input
    """

    # STEP 1: Add the user's message to the chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    st.session_state.conversation_count += 1  # Increment message counter

    # STEP 2: Generate the assistant's response
    # Create a chat message container for the assistant's response
    with st.chat_message("assistant"):
        # Show a loading spinner while generating response
        with st.spinner("🌍 Thinking..."):
            try:
                # CALL THE CORE FUNCTION: This sends the question to utils.get_response()
                # which searches the database and queries the AI model
                response = get_response(
                    user_input,  # The user's question
                    ORGANIZATION_NAME,  # Context about the organization
                    ORGANIZATION_INFO,  # Detailed organization information
                    CONTACT_INFO , # Contact details
                    st.session_state.messages  # pass the history
                )

                # STEP 3: Add the assistant's response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })

                # STEP 4: Rerun the app to refresh the display with new messages
                st.rerun()

            except Exception as e:
                # If anything goes wrong, show an error message
                error_msg = f"""⚠️ **I encountered an error**

Sorry, something went wrong: `{str(e)}`

Please try asking your question differently or refresh the page.

💡 **Tip:** If this keeps happening, you can visit our [official website](https://visitethiopia.et) for travel information."""

                # Add error message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
                st.rerun()  # Refresh the page

# ==================== SIDEBAR RENDERING ====================
def render_sidebar():
    """Create and populate the sidebar with information and controls"""

    # Everything in this block appears in the sidebar
    with st.sidebar:

        # Logo and title at the top of sidebar
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <h2 style="color: #1a472a; margin: 0;">🇪🇹 Visit Ethiopia</h2>
            <p style="color: #666; margin: 0;">AI Travel Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")  # Horizontal divider line

        # ===== STATUS CARD =====
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 System Status")

        # Two columns for metrics
        col1, col2 = st.columns(2)
        with col1:
            doc_count, doc_error = get_document_count()

            if doc_error is not None:
                st.metric("📚 Knowledge Base", "DB Error")
                # Optional debug line:
                # st.caption(f"KB error: {doc_error}")
            elif doc_count == 0:
                st.metric("📚 Knowledge Base", "0")
            else:
                st.metric("📚 Knowledge Base", f"{doc_count:,}")

        with col2:
            # Show AI status (Online or Limited Mode)
            status = "🟢 Online" if is_llm_available() else "🟡 Limited Mode"
            st.metric("🤖 AI Status", status)

        # Show how many messages exchanged in this session
        st.markdown(f"**💬 Session Messages:** {st.session_state.conversation_count}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ===== KNOWLEDGE BASE INDEXING =====
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🧠 Knowledge Base")
        st.caption("Index official tourism websites into Chroma.")

        if st.button("🔄 Index/Update Knowledge Base", use_container_width=True):
            with st.spinner("Indexing sources... this may take several minutes."):
                success, failed = ingest_default_sources()
            if success > 0:
                st.success(f"Indexing finished: {success} source(s) indexed, {failed} failed.")
            else:
                st.error(f"Indexing failed. Success: {success}, Failed: {failed}")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ===== REGIONS COVERED CARD =====
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Regions Covered")

        # Dictionary of Ethiopian regions and their tourism websites
        regions = {
            "🇪🇹 Federal": "visitethiopia.et",
            "🇪🇹 Oromia": "visitoromia.org",
            "⛪ Amhara": "visitamhara.travel",
            "🏔️ Tigray": "tourismtigrai.com",
            "☕ Sidama": "visitsidama.travel",
            " South Ethiopia": "visitsouthethiopia.et"
        }

        # Display each region with its website
        for region, site in regions.items():
            st.markdown(f"**{region}** \n<small>{site}</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ===== SUGGESTED QUESTIONS CARD =====
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 💡 Quick Questions")

        # Pre-written questions users can click on
        suggestions = [
            "🏛️ What are the UNESCO World Heritage sites in Ethiopia?",
            "🏔️ Tell me about the Simien Mountains National Park",
            "⛪ When is the best time to visit Lalibela?",
            "🍛 What traditional Ethiopian foods should I try?",
            "🎉 What are the major festivals in Ethiopia?",
            "🧭 How do I get to the Danakil Depression?",
            "☕ Tell me about the Ethiopian coffee ceremony",
            "🗺️ What are the top attractions in Oromia?"
        ]

        # Create a button for each suggestion
        for suggestion in suggestions:
            if st.button(suggestion, key=suggestion, use_container_width=True):
                # When clicked, process the question (remove emoji prefix first)
                # This calls the same function as typing in chat
                process_user_input(suggestion.replace("🏛️ ", "").replace("🏔️ ", "").replace("⛪ ", "").replace("🍛 ", "").replace("🎉 ", "").replace("🧭 ", "").replace("☕ ", "").replace("🗺️ ", ""))

        st.markdown('</div>', unsafe_allow_html=True)

        # ===== TRAVEL TIPS CARD =====
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 💡 Travel Tips")
        # Info box with essential travel information
        st.info("""
        **Best Time to Visit:** October to March (Dry season)
        
        **Visa:** E-visa available at evisa.gov.et
        
        **Currency:** Ethiopian Birr (ETB)
        
        **Language:** Amharic is widely spoken, English in tourist areas
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # ===== CLEAR CHAT BUTTON =====
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            # Reset messages to just the welcome message
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "✨ Chat history cleared! How can I help you with your Ethiopian adventure today? 🇪🇹",
                    "timestamp": datetime.now()
                }
            ]
            st.session_state.conversation_count = 0  # Reset counter
            st.rerun()  # Refresh the page

        # ===== SIDEBAR FOOTER =====
        st.markdown("---")
        st.caption("© 2024 Visit Ethiopia")
        st.caption("Powered by RAG Technology")  # RAG = Retrieval Augmented Generation

# ==================== MAIN CHAT INTERFACE ====================
def render_chat():
    """Display the chat conversation and handle input"""

    # 1. Display existing history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(f"🕐 {message['timestamp'].strftime('%I:%M %p')}")

    # 2. Handle New User Input
    if prompt := st.chat_input("Ask me anything about traveling to Ethiopia..."):
        # Add user message to state & display immediately
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕐 {user_message['timestamp'].strftime('%I:%M %p')}")

        # 3. Get and display Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching Ethiopia's treasures..."):
                response_data = get_remote_response(prompt)

                if response_data:
                    full_response = response_data.get("answer", "I'm sorry, I couldn't process that.")
                    st.markdown(full_response)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.now()
                    })
                else:
                    st.error("Failed to connect to the travel assistant server.")
# ==================== FOOTER ====================
def render_footer():
    """Display footer with links at the bottom of the page"""

    st.markdown("---")  # Horizontal divider

    # Create columns for footer links
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("[🏠 Official Website](https://visitethiopia.et)")
    with col2:
        st.markdown("[📧 Contact](mailto:info@visitethiopia.et)")

    # Centered footer text
    st.markdown(
        '<div class="footer">'
        '🇪🇹 Visit Ethiopia AI Travel Assistant | Your Intelligent Guide to the Land of Origins'
        '</div>',
        unsafe_allow_html=True
    )

# ==================== MAIN APP ENTRY POINT ====================
def main():
    """Main application function - orchestrates the entire app"""

    # STEP 1: Initialize session state (chat history, counters)
    init_session_state()

    # STEP 2: Render the sidebar with all its content
    render_sidebar()

    # STEP 3: Create main content area with centered layout
    # Using columns to center content: left(1) | center(8) | right(1)
    col1, col2, col3 = st.columns([1, 8, 1])

    # Everything in col2 will be centered
    with col2:

        # Display the main header with Ethiopian flag colors
        st.markdown("""
        <div class="main-header">
            <h1>🇪🇹 Visit Ethiopia AI Assistant</h1>
            <p>Your Intelligent Guide to the Land of Origins</p>
        </div>
        """, unsafe_allow_html=True)

        # Show welcome/guide message for new users (only if no messages yet)
        if len(st.session_state.messages) == 1:  # Only the initial welcome message
            with st.expander("🎯 Quick Guide", expanded=False):  # Collapsible section
                st.markdown("""
                **How to use this assistant:**
                
                1. **Ask any travel-related question** about Ethiopia
                2. **Get detailed, accurate information** from official tourism sources
                3. **Follow up with more questions** - I remember our conversation!
                4. **Use the sidebar** for quick questions and travel tips
                
                **Example questions:**
                - "What are the must-see attractions in Lalibela?"
                - "Tell me about the food in Ethiopia"
                - "How do I get from Addis Ababa to Gondar?"
                - "What is the weather like in Simien Mountains in October?"
                """)

        # STEP 4: Render the chat interface (messages and input)
        render_chat()

        # STEP 5: Render the footer
        render_footer()

# ==================== APP START ====================
# This checks if the script is being run directly (not imported as a module)
# If so, call the main() function to start the app
if __name__ == "__main__":
    main()