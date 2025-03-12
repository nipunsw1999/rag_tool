import streamlit as st

st.write("This is an Agentic RAG application powered by Wikipedia, ArXiv, Google Scholar, and custom-added websites for enhanced information retrieval.")


if "base_model" not in st.session_state:
    st.session_state.base_model = "gemma2-9b-it"

if "tools" not in st.session_state:
    st.session_state.tools = []

if "url" not in st.session_state:
    st.session_state.url = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

with st.sidebar:
    st.title("Agentic RAG")
    
    st.markdown("Customization")
    with st.form(key="model_form"):
        model = st.selectbox("Select a model", ["llama-3.3-70b-versatile", "llama3-70b-8192", "gemini-2.0-flash","gemma2-9b-it"])
        set_model = st.form_submit_button("Change model")
        if set_model:
            st.session_state.base_model = model
            st.success("Model updated")
        
        wiki = st.toggle("Wikipedia")
        if wiki:
            st.session_state.tools.append("wiki")
        elif "wiki" in st.session_state.tools:
            st.session_state.tools.remove("wiki")
            
        arx = st.toggle("arxiv")
        if arx:
            st.session_state.tools.append("arxiv")
        elif "arxiv" in st.session_state.tools:
            st.session_state.tools.remove("arxiv")
  
        scholar = st.toggle("Google Scholar")
        if scholar:
            st.session_state.tools.append("Google Scholar")
        elif "Google Scholar" in st.session_state.tools:
            st.session_state.tools.remove("Google Scholar")
          
        link1 =st.text_input("Enter the URL: ")
        add_link = st.form_submit_button("Add Link")
        if add_link and link1 != "":
            st.session_state.url = link1
            st.success("Link added")
        
        st.write(f"URL: {st.session_state.url}")
        
        clear_chat = st.form_submit_button("Clear Chat")
        if clear_chat:
            st.session_state.chat_history = []
            st.success("Chat history cleared")
            
st.session_state.tools = set(st.session_state.tools)


# Configuring the models and tools


user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    response = "I'm here to help! (LLM response will go here)"
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.write(response)