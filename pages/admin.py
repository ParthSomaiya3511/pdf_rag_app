import streamlit as st
import shutil
import os

st.title("🛠 Admin Panel")

if st.button("Delete Vector Database"):

    if os.path.exists("vectorstore"):

        shutil.rmtree("vectorstore")

        os.mkdir("vectorstore")

        st.success("Vector Database Deleted")