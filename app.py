import streamlit as st
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    load_vector_store
)
from utils.rag_chain import ask_gemini

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="✨",
    layout="wide"
)

# =====================================================
# LOAD CSS
# =====================================================
with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# =====================================================
# SESSION STATE
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
    ✨ PDF AI
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sidebar-subtitle">
    AI Powered PDF Assistant
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    uploaded_file_sidebar = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="sidebar_uploader"
    )

    top_k = st.slider(
        "Top K Chunks",
        1,
        10,
        4
    )

    process_btn = st.button(
        "Process PDF"
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
<div class="feature-box">
    <h3>🚀 Features</h3>
    <ul>
        <li>✅ AI PDF Chat</li>
        <li>✅ Vector Search</li>
        <li>✅ Semantic Retrieval</li>
        <li>✅ Gemini AI</li>
        <li>✅ FAISS Database</li>
        <li>✅ Smart Chunk Search</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# DYNAMIC HERO SECTION
# =====================================================
if len(st.session_state.messages) == 0:
    # Marker used in CSS to align the floating "+" button dynamically
    st.markdown('<div class="hero-active-marker"></div>', unsafe_allow_html=True)

    hero_html = """<div class="hero-wrapper">
<div class="hero-badge">
    ✨ AI Powered PDF Assistant
</div>
<div class="hero-title">
    Where should we start?
</div>
<div class="hero-subtitle">
    Upload documents and ask anything using AI semantic search.
</div>
<div class="hero-input-spacer"></div>
</div>"""

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )

    # Injecting precise centering CSS only when empty
    st.markdown("""
<style>
div[data-testid="stChatInput"] {
    bottom: 36vh !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
div[data-testid="stChatInput"] > div {
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# FLOATING "+" POPUP MENU
# =====================================================
with st.container():
    st.markdown('<div class="plus-menu-container">', unsafe_allow_html=True)
    with st.popover("＋", use_container_width=False):
        st.markdown("<b style='color:#111827;'>Upload Context</b>", unsafe_allow_html=True)
        uploaded_file_pop = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            key="pop_uploader"
        )
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(226,232,240,0.8);'>", unsafe_allow_html=True)
        st.button("📂 Add from Drive", disabled=True)
        st.button("🖼️ Create image", disabled=True)
        st.button("📝 Canvas", disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Sync both uploaders
uploaded_file = uploaded_file_sidebar or uploaded_file_pop

# =====================================================
# PROCESS PDF
# =====================================================
if process_btn and uploaded_file:
    with st.spinner("Processing PDF..."):
        os.makedirs(
            "data/uploaded_pdfs",
            exist_ok=True
        )

        save_path = os.path.join(
            "data/uploaded_pdfs",
            uploaded_file.name
        )

        with open(save_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )

        documents = load_pdf(save_path)
        chunks = split_text(documents)
        embeddings = get_embedding_model()
        create_vector_store(
            chunks,
            embeddings
        )

        st.success(
            f"'{uploaded_file.name}' processed successfully"
        )

# =====================================================
# CHAT HISTORY
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Inject Gemini-style action button bar beneath Assistant replies
        if msg["role"] == "assistant":
            st.markdown("""
            <div class="assistant-action-bar">
                <span title="Good Response">👍</span>
                <span title="Bad Response">👎</span>
                <span title="Modify/Regenerate">🔄</span>
                <span title="Share Context">🔗</span>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# CHAT INPUT
# =====================================================
user_question = st.chat_input(
    "Ask anything from your PDF..."
)

# =====================================================
# AI RESPONSE
# =====================================================
if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    embeddings = get_embedding_model()
    vectorstore = load_vector_store(
        embeddings
    )

    if vectorstore is None:
        st.warning(
            "Please upload and process PDF first"
        )
    else:
        docs = vectorstore.similarity_search(
            user_question,
            k=top_k
        )

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
        Answer ONLY from the provided PDF context.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        with st.spinner("AI Thinking..."):
            response = ask_gemini(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown("""
            <div class="assistant-action-bar">
                <span title="Good Response">👍</span>
                <span title="Bad Response">👎</span>
                <span title="Modify/Regenerate">🔄</span>
                <span title="Share Context">🔗</span>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("View Source Chunks"):
            for i, doc in enumerate(docs):
                st.markdown(f"### Chunk {i+1}")
                st.write(doc.page_content)
                st.divider()