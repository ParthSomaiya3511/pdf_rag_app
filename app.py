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

# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

# ===================================
# LOAD CSS
# ===================================

with open("assets/style.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ===================================
# SESSION STATE
# ===================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# ===================================
# SIDEBAR
# ===================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
        🤖 PDF AI
    </div>

    <div class="sidebar-subtitle">
        Intelligent PDF assistant
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-section">
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
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

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### Features

    - AI PDF Chat
    - Semantic Search
    - Vector Retrieval
    - Context Aware Answers
    """)

# ===================================
# EMPTY STATE
# ===================================

if len(st.session_state.messages) == 0:

    st.markdown("""
    <div class="empty-state">

        <div class="empty-icon">
            🤖
        </div>

        <div class="empty-title">
            AI PDF Assistant
        </div>

        <div class="empty-subtitle">
            Upload your PDF and start chatting with AI
        </div>

    </div>
    """, unsafe_allow_html=True)

# ===================================
# PROCESS PDF
# ===================================

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
            "PDF processed successfully"
        )

# ===================================
# CHAT HISTORY
# ===================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ===================================
# CHAT INPUT
# ===================================

user_question = st.chat_input(
    "Ask anything from your PDF..."
)

# ===================================
# QUESTION ANSWERING
# ===================================

if user_question:

    st.session_state.messages.append({

        "role":"user",

        "content":user_question

    })

    with st.chat_message("user"):

        st.markdown(user_question)

    embeddings = get_embedding_model()

    vectorstore = load_vector_store(
        embeddings
    )

    if vectorstore is None:

        st.warning(
            "Please upload PDF first"
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
        Answer only from provided PDF context.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        with st.spinner("AI is thinking..."):

            response = ask_gemini(prompt)

        st.session_state.messages.append({

            "role":"assistant",

            "content":response

        })

        with st.chat_message("assistant"):

            st.markdown(response)

        with st.expander(
            "View Source Chunks"
        ):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(
                    doc.page_content
                )

                st.divider()