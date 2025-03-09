import streamlit as st

st.write("This is an Agentic RAG application powered by Wikipedia, ArXiv, Google Scholar, and custom-added websites for enhanced information retrieval.")

with st.sidebar:
    st.markdown("Customization")
    model = st.selectbox("Select a model", ["llama-3.3-70b-versatile", "llama-3.3-70b-versatile-qa", "llama-3.3-70b-versatile-qa-cc"])