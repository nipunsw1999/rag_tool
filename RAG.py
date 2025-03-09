import streamlit as st

st.write("This is an Agentic RAG application powered by Wikipedia, ArXiv, Google Scholar, and custom-added websites for enhanced information retrieval.")


if "base_model" not in st.session_state:
    st.session_state.base_model = "gemma2-9b-it"
    
with st.sidebar:
    st.title("Agentic RAG")
    tab1, tab2 = st.tabs(["View", "Change"])
    with tab1:
        with st.form(key="model_form_refresh"):
            col1,col2,col3 = st.columns([1,1,2])
            with col1:
                st.write("Model ")
                st.write("Tool 1")
                st.write("Tool 2")
        
            refresh_btn = st.form_submit_button("Refresh")
        with col2:
            st.write(":")
            st.write(":")
            st.write(":")
        
        with col3:
            st.write(st.session_state.base_model)
            st.write()
        st.write()
    
    with tab2:
        st.markdown("Customization")
        with st.form(key="model_form"):
            model = st.selectbox("Select a model", ["llama-3.3-70b-versatile", "llama3-70b-8192", "gemini-2.0-flash","gemma2-9b-it"])
            set_model = st.form_submit_button("Change model")
            
            if set_model:
                st.session_state.base_model = model