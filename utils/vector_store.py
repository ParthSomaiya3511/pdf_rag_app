from langchain_community.vectorstores import FAISS
import os

VECTOR_DB_PATH = "vectorstore/faiss_index"

def create_vector_store(chunks, embeddings):

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore


def load_vector_store(embeddings):

    if os.path.exists(VECTOR_DB_PATH):

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    return None