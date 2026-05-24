import streamlit as st
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import get_embedding_model
from utils.vector_store import (
    create_vector_store,
    load_vector_store
)
from utils.rag_chain import get_llm

st.set_page_config(
    page_title="PDF AI Chat",
    layout="wide"
)

st.title("📚 PDF AI Chat App")

# Sidebar
with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    process_btn = st.button("Process PDF")

# Process PDF
if process_btn and uploaded_file:

    with st.spinner("Processing PDF..."):

        save_path = f"data/uploaded_pdfs/{uploaded_file.name}"

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF
        documents = load_pdf(save_path)

        # Split Text
        chunks = split_text(documents)

        # Embeddings
        embeddings = get_embedding_model()

        # Vector Store
        create_vector_store(chunks, embeddings)

        st.success("PDF Processed Successfully!")

# Chat UI
st.header("Ask Questions")

user_question = st.text_input(
    "Ask question from PDF"
)

if user_question:

    embeddings = get_embedding_model()

    vectorstore = load_vector_store(embeddings)

    if vectorstore is None:

        st.warning("Please upload PDF first")

    else:

        docs = vectorstore.similarity_search(
            user_question,
            k=4
        )

        llm = get_llm()

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = f"""
        Answer the question using the context below.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response = llm.invoke(prompt)

        st.write("### Answer")
        st.success(response.content)