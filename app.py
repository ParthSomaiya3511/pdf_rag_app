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

if "processing" not in st.session_state:
    st.session_state.processing = False

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

    st.markdown(
        "<div class='sidebar-divider'></div>",
        unsafe_allow_html=True
    )

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
        "Process PDF",
        use_container_width=True
    )

    st.markdown(
        "<div class='sidebar-divider'></div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="feature-box">
        <h3>🚀 Features</h3>
        <ul>
            <li>✅ AI PDF Chat</li>
            <li>✅ Semantic Search</li>
            <li>✅ Gemini AI</li>
            <li>✅ FAISS Vector DB</li>
            <li>✅ Smart Retrieval</li>
            <li>✅ Production UI</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DYNAMIC HERO SECTION
# =====================================================
if len(st.session_state.messages) == 0:
    # Marker used in CSS to align the floating custom input dynamically
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


# =====================================================
# CHAT HISTORY
# =====================================================

chat_container = st.container()

with chat_container:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg["role"] == "assistant":

                st.markdown("""
                <div class="assistant-action-bar">
                    <span>👍</span>
                    <span>👎</span>
                    <span>🔄</span>
                    <span>📋</span>
                </div>
                """, unsafe_allow_html=True)

# =====================================================
# FLOATING INPUT AREA
# =====================================================

st.markdown(
    '<div class="chat-input-fixed">',
    unsafe_allow_html=True
)

with st.container():

    col1, col2, col3 = st.columns(
        [1, 10, 1],
        vertical_alignment="center"
    )

    # =================================================
    # PLUS MENU
    # =================================================

    with col1:

        with st.popover("＋", use_container_width=True):

            st.markdown(
                "### Upload Context"
            )

            uploaded_file_pop = st.file_uploader(
                "Choose PDF",
                type=["pdf"],
                key="pop_uploader"
            )

            st.button(
                "📂 Google Drive",
                disabled=True,
                use_container_width=True
            )

            st.button(
                "🖼️ Image",
                disabled=True,
                use_container_width=True
            )

            st.button(
                "📝 Canvas",
                disabled=True,
                use_container_width=True
            )

    # =================================================
    # CHAT INPUT
    # =================================================

    with col2:

        user_question = st.chat_input(
            "Ask anything from your PDF..."
        )

    # =================================================
    # RIGHT BUTTON
    # =================================================

    with col3:

        st.markdown("""
        <div class="mic-button">
            🎙️
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# UPLOAD SYNC
# =====================================================

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
            f.write(uploaded_file.getbuffer())

        documents = load_pdf(save_path)

        chunks = split_text(documents)

        embeddings = get_embedding_model()

        create_vector_store(
            chunks,
            embeddings
        )

        st.success(
            f"{uploaded_file.name} processed successfully."
        )

# =====================================================
# USER MESSAGE
# =====================================================

if user_question and not st.session_state.processing:

    st.session_state.processing = True

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    st.rerun()

# =====================================================
# AI RESPONSE
# =====================================================

if (
    st.session_state.processing
    and len(st.session_state.messages) > 0
    and st.session_state.messages[-1]["role"] == "user"
):

    last_question = st.session_state.messages[-1]["content"]

    embeddings = get_embedding_model()

    vectorstore = load_vector_store(
        embeddings
    )

    if vectorstore is None:

        st.warning(
            "Please upload and process PDF first."
        )

        st.session_state.processing = False

    else:

        with st.spinner("AI Thinking..."):

            docs = vectorstore.similarity_search(
                last_question,
                k=top_k
            )

            context = "\n\n".join([
                doc.page_content
                for doc in docs
            ])

            prompt = f"""
            Answer ONLY from the PDF context.

            Context:
            {context}

            Question:
            {last_question}

            Answer:
            """

            response = ask_gemini(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        st.session_state.processing = False

        st.rerun()

# =====================================================
# SOURCE CHUNKS
# =====================================================

if len(st.session_state.messages) > 0:

    with st.expander("View Retrieved Chunks"):

        try:

            embeddings = get_embedding_model()

            vectorstore = load_vector_store(
                embeddings
            )

            if vectorstore:

                docs = vectorstore.similarity_search(
                    st.session_state.messages[-1]["content"],
                    k=top_k
                )

                for i, doc in enumerate(docs):

                    st.markdown(
                        f"### Chunk {i+1}"
                    )

                    st.write(doc.page_content)

                    st.divider()

        except:
            pass