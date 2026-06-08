import streamlit as st
import streamlit.components.v1 as components
import os
import time
import sqlite3
import uuid

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    load_vector_store
)
from utils.rag_chain import ask_gemini
from utils.pdf_export import generate_chat_pdf
from utils.tts import text_to_audio

# =====================================================
# DATABASE UTILITIES (SQLite Persistence)
# =====================================================
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

def db_create_session(session_id, title):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)", (session_id, title))
    conn.commit()
    conn.close()

def db_save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def db_get_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title FROM sessions ORDER BY created_at DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

def db_get_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    messages = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return messages

def db_delete_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

# Initialize Schema
init_db()

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI PDF Assistant Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

if "voice_query" not in st.session_state:
    st.session_state.voice_query = ""

if "audio_reply" not in st.session_state:
    st.session_state.audio_reply = None

# =====================================================
# SYSTEM AUTOMATION: INSTANT VOICE REPLY
# =====================================================
if "voice_text" in st.query_params:
    voice_transcript = st.query_params["voice_text"]
    st.query_params.clear()
    
    if voice_transcript and voice_transcript.strip():
        # Session is registered in the database only when the first voice question is sent
        db_create_session(st.session_state.current_session_id, voice_transcript[:25])
        
        db_save_message(st.session_state.current_session_id, "user", voice_transcript)
        st.session_state.messages = db_get_messages(st.session_state.current_session_id)
        st.session_state.processing = True
        st.rerun()

# Sync message timeline from database (handles empty state dynamically)
st.session_state.messages = db_get_messages(st.session_state.current_session_id)

# ==========================================
# VOICE QUERY AUTO PROCESS
# ==========================================
if st.session_state.voice_query:
    user_question = st.session_state.voice_query
    st.session_state.voice_query = ""

    # Session is registered in the database only when the first query is sent
    db_create_session(st.session_state.current_session_id, user_question[:25])

    db_save_message(st.session_state.current_session_id, "user", user_question)
    st.session_state.processing = True
    st.rerun()

# =====================================================
# THEME INJECTION UTILITY
# =====================================================
def load_css_and_theme():
    os.makedirs("assets", exist_ok=True)
    if os.path.exists("assets/style.css"):
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            
    if st.session_state.theme == "dark":
        st.markdown(
            """
            <style>
            :root {
                --bg-gradient: radial-gradient(circle at 50% 25%, #0f172a 0%, #020617 85%) !important;
                --text-color: #f8fafc !important;
                --card-bg: rgba(30, 41, 59, 0.45) !important;
                --card-border: rgba(255, 255, 255, 0.08) !important;
                --sidebar-bg: #090d16 !important;
                --input-bg: #1e293b !important;
            }
            .stApp { background: var(--bg-gradient) !important; color: var(--text-color) !important; }
            [data-testid="stSidebar"] { background-color: var(--sidebar-bg) !important; color: var(--text-color) !important; }
            [data-testid="stSidebar"] * { color: var(--text-color) !important; }
            [data-testid="stChatInputContainer"] { background-color: var(--input-bg) !important; border: 1px solid var(--card-border) !important; }
            [data-testid="stChatInputContainer"] textarea { color: var(--text-color) !important; }
            .chat-bubble.user-bubble { background-color: #1e293b !important; color: #f8fafc !important; border: 1px solid #334155 !important; }
            .chat-bubble.assistant-bubble { background-color: #0f172a !important; color: #e2e8f0 !important; border: 1px solid #334155 !important; }
            .sources-wrapper-box { background-color: var(--card-bg) !important; border-color: var(--card-border) !important; }
            [data-testid="stSidebar"] [data-testid="stFileUploader"] { background-color: var(--card-bg) !important; border: 1px dashed var(--card-border) !important; }
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background-color: var(--card-bg) !important; border: 1px dashed var(--card-border) !important; color: var(--text-color) !important; }
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background-color: var(--sidebar-bg) !important; color: var(--text-color) !important; border: 1px solid var(--card-border) !important; }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            :root {
                --bg-gradient: radial-gradient(circle at 50% 25%, #f4f8fd 0%, #ffffff 85%) !important;
                --text-color: #0f172a !important;
                --card-bg: rgba(255, 255, 255, 0.45) !important;
                --card-border: rgba(0, 0, 0, 0.05) !important;
                --sidebar-bg: #fafbfc !important;
                --input-bg: #ffffff !important;
            }
            .stApp { background: var(--bg-gradient) !important; color: var(--text-color) !important; }
            [data-testid="stSidebar"] { background-color: var(--sidebar-bg) !important; }
            [data-testid="stChatInputContainer"] { background-color: var(--input-bg) !important; border: 1px solid var(--card-border) !important; }
            [data-testid="stChatInputContainer"] textarea { color: var(--text-color) !important; }
            .chat-bubble.user-bubble { background-color: #f1f5f9 !important; color: #1e293b !important; }
            .chat-bubble.assistant-bubble { background-color: #ffffff !important; color: #0f172a !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

load_css_and_theme()

# ==========================================
# AUTO PLAY AI VOICE RESPONSE
# ==========================================
if st.session_state.audio_reply:
    st.audio(
        st.session_state.audio_reply,
        format="audio/mp3"
    )

# =====================================================
# SIDEBAR (GEMINI MODE WITH SYSTEM CONTROLS)
# =====================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header-card">
        <div class="sidebar-logo">✨ PDF AI</div>
        <div class="sidebar-subtitle">AI Powered PDF Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    col_theme, col_new_chat = st.columns(2)
    with col_theme:
        theme_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
        if st.button(theme_label, use_container_width=True):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    with col_new_chat:
        if st.button("＋ New Chat", use_container_width=True, type="secondary"):
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Active Document</p>', unsafe_allow_html=True)
    uploaded_file_sidebar = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="sidebar_uploader",
        label_visibility="collapsed"
    )

    st.markdown('<p class="sidebar-label" style="margin-top:15px;">Search Depth (Chunks)</p>', unsafe_allow_html=True)
    top_k = st.slider(
        "Top K Chunks",
        1,
        10,
        4,
        label_visibility="collapsed"
    )

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    process_btn = st.button(
        "⚡ Process Document",
        use_container_width=True,
        type="primary"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Conversations Threads</p>', unsafe_allow_html=True)
    active_threads = db_get_sessions()
    
    for tid, title in active_threads:
        col_t, col_d = st.columns([8, 2])
        with col_t:
            if st.button(f"💬 {title[:20]}...", key=f"t_{tid}", use_container_width=True):
                st.session_state.current_session_id = tid
                st.session_state.messages = db_get_messages(tid)
                st.rerun()
        with col_d:
            if st.button("🗑", key=f"d_{tid}", use_container_width=True):
                db_delete_session(tid)
                if st.session_state.current_session_id == tid:
                    st.session_state.current_session_id = str(uuid.uuid4())
                st.rerun()

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Voice & Utilities</p>', unsafe_allow_html=True)
    
    col_mic, col_exp = st.columns(2)
    with col_mic:
        mic_status = "🎙️ Listening" if st.session_state.voice_active else "🎙️ Start Voice"
        if st.button(mic_status, use_container_width=True):
            st.session_state.voice_active = not st.session_state.voice_active
            st.rerun()
            
    with col_exp:
        try:
            pdf_path = generate_chat_pdf(st.session_state.messages)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Export PDF",
                    data=f,
                    file_name="AI_PDF_Chat_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        except Exception:
            # printable HTML template backup if PDF module fails
            html_export_template = """
            <html>
            <head>
                <style>
                    body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #0f172a; padding: 40px; margin: 0; }
                    .container { max-width: 800px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 16px; border: 1px solid #e2e8f0; }
                    h1 { border-bottom: 2px solid #3b82f6; padding-bottom: 15px; font-size: 26px; color: #1e3a8a; font-weight: 700; margin-top: 0; }
                    .msg-row { margin-bottom: 25px; padding: 16px 20px; border-radius: 12px; font-size: 14px; line-height: 1.6; }
                    .user { background-color: #f1f5f9; border-left: 4px solid #475569; }
                    .assistant { background-color: #eff6ff; border-left: 4px solid #2563eb; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>✨ PDF AI Chat Transcript</h1>
            """
            for m in st.session_state.messages:
                role_class = "user" if m['role'] == "user" else "assistant"
                html_export_template += f"""
                    <div class="msg-row {role_class}">
                        <div class="role" style="font-weight:700; font-size:11px; text-transform:uppercase; color:#64748b; margin-bottom:6px;">{m['role']}</div>
                        <div class="content">{m['content']}</div>
                    </div>
                """
            html_export_template += """
                </div>
                <script>window.onload = function() { window.print(); }</script>
            </body>
            </html>
            """
            st.download_button(
                label="📄 Export PDF",
                data=html_export_template,
                file_name="chat_transcript.html",
                mime="text/html",
                use_container_width=True
            )

# =====================================================
# SPEECH RECOGNITION POPUP FRAME
# =====================================================
if st.session_state.voice_active:
    st.markdown('<div class="voice-widget-card">', unsafe_allow_html=True)
    components.html("""
        <div style="font-family: 'Inter', sans-serif; text-align: center; color: #475569; font-size: 13px; padding: 10px; background: rgba(255,255,255,0.7); border-radius: 12px;">
            <p id="status" style="margin: 0 0 8px 0; font-weight: 500; color: #2563eb;">🎙️ Ready to record. Click button below to start.</p>
            <button id="btn" style="background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); color: white; border: none; padding: 8px 18px; border-radius: 20px; font-weight: 600; cursor: pointer; box-shadow: 0 4px 12px rgba(37,99,235,0.2); transition: all 0.2s;">🎙️ Click and Speak</button>
        </div>
        <script>
            const btn = document.getElementById('btn');
            const status = document.getElementById('status');
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            
            if (!SpeechRecognition) {
                status.innerText = "❌ Browser does not support speech recognition. Please use Chrome/Edge.";
                btn.style.display = "none";
            } else {
                const recognition = new SpeechRecognition();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                
                btn.addEventListener('click', () => {
                    recognition.start();
                    status.innerText = "⏳ Listening carefully...";
                    btn.style.background = "#ef4444";
                    btn.innerText = "Recording...";
                });
                
                recognition.onresult = (event) => {
                    const transcript = event.results[0][0].transcript;
                    status.innerText = "✅ Captured: " + transcript;
                    
                    const url = window.location.href.split("?")[0] + "?voice_text=" + encodeURIComponent(transcript);
                    window.parent.location.href = url;
                };
                
                recognition.onerror = (e) => {
                    status.innerText = "❌ Mic error: Make sure permissions are granted.";
                    btn.style.background = "#2563eb";
                    btn.innerText = "🎙️ Try Again";
                };
            }
        </script>
    """, height=100)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# DYNAMIC HERO INTRO (CENTER FLOAT)
# =====================================================
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-badge">
                ✨ AI Powered PDF Assistant
            </div>
            <div class="hero-title">
                Chat with your PDFs
            </div>
            <div class="hero-subtitle">
                Upload documents and ask anything using AI semantic search.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# CONVERSATION RENDER (MESSAGES TIMELINE)
# =====================================================
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">
                        {msg["content"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row assistant-row">
                    <div class="assistant-avatar">✨</div>
                    <div class="chat-bubble assistant-bubble">
                        <div class="assistant-badge-tag">Flash Output</div>
                        <div class="assistant-response-content">{msg["content"]}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# MINIMALIST CAPSULE INPUT BAR
# =====================================================
user_question = st.chat_input(
    "Ask anything from your PDF..."
)

# =====================================================
# FILE UPLOAD PIPELINE
# =====================================================
uploaded_file = uploaded_file_sidebar

if process_btn and uploaded_file:
    with st.spinner("Processing document corpus..."):
        os.makedirs("data/uploaded_pdfs", exist_ok=True)
        save_path = os.path.join("data/uploaded_pdfs", uploaded_file.name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        documents = load_pdf(save_path)
        chunks = split_text(documents)
        embeddings = get_embedding_model()

        create_vector_store(chunks, embeddings)
        st.success(f"Indexed successfully: {uploaded_file.name}")

# =====================================================
# MESSAGE WORKFLOW CONTROL & ANIMATED TYPING EFFECT
# =====================================================
if user_question and not st.session_state.processing:
    st.session_state.processing = True
    
    # Session is created in database on-demand on first query submission
    db_create_session(st.session_state.current_session_id, user_question[:25])
    
    # Save user query to DB and State
    db_save_message(st.session_state.current_session_id, "user", user_question)
    st.session_state.messages = db_get_messages(st.session_state.current_session_id)
    st.rerun()

if (
    st.session_state.processing
    and len(st.session_state.messages) > 0
    and st.session_state.messages[-1]["role"] == "user"
):
    last_question = st.session_state.messages[-1]["content"]
    embeddings = get_embedding_model()
    vectorstore = load_vector_store(embeddings)

    if vectorstore is None:
        st.warning("Please upload and process a PDF document first.")
        st.session_state.processing = False
    else:
        with st.spinner("Analyzing semantic mapping..."):
            docs = vectorstore.similarity_search(last_question, k=top_k)
            context = "\n\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer ONLY from the PDF context. Keep it accurate and readable.

            Context:
            {context}

            Question:
            {last_question}

            Answer:
            """
            response = ask_gemini(prompt)
            
            # Generate audio reply
            audio_file = text_to_audio(response)
            st.session_state.audio_reply = audio_file

        # ANIMATED TYPING STREAM (Simulated delivery flow character-by-character)
        st.markdown(
            f"""
            <div class="chat-row assistant-row">
                <div class="assistant-avatar">✨</div>
                <div class="chat-bubble assistant-bubble">
                    <div class="assistant-badge-tag">Flash Output</div>
                    <div class="assistant-response-content" id="live-typing-container">
            """,
            unsafe_allow_html=True
        )
        
        typing_placeholder = st.empty()
        streamed_text = ""
        for word in response.split(" "):
            streamed_text += word + " "
            typing_placeholder.markdown(
                f"""
                <div class="chat-row assistant-row" style="margin-top: -24px;">
                    <div class="assistant-avatar">✨</div>
                    <div class="chat-bubble assistant-bubble">
                        <div class="assistant-badge-tag">Generating response...</div>
                        <div class="assistant-response-content">{streamed_text}▒</div>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            time.sleep(0.04)

        # Commit final structured text to SQL database and State
        db_save_message(st.session_state.current_session_id, "assistant", response)
        
        # Move state indicators INSIDE the execution check block to resolve the infinite loops
        st.session_state.processing = False
        st.rerun()

# =====================================================
# PDF SOURCE METADATA PANEL (GRID SYSTEM)
# =====================================================
if len(st.session_state.messages) > 0:
    st.markdown('<div class="sources-wrapper-box">', unsafe_allow_html=True)
    with st.expander("🔍 View Retrieved Document Chunks", expanded=False):
        try:
            embeddings = get_embedding_model()
            vectorstore = load_vector_store(embeddings)
            if vectorstore:
                docs = vectorstore.similarity_search(
                    st.session_state.messages[-2]["content"] if len(st.session_state.messages) > 1 else st.session_state.messages[-1]["content"],
                    k=top_k
                )
                
                st.markdown('<div class="chunks-cards-grid">', unsafe_allow_html=True)
                for i, doc in enumerate(docs):
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <div class="chunk-card-header">
                                <span class="chunk-index-tag">CHUNK {i+1}</span>
                                <span class="chunk-score-metric">Relevance Verified</span>
                            </div>
                            <div class="chunk-card-body">
                                {doc.page_content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception:
            pass
    st.markdown('</div>', unsafe_allow_html=True)