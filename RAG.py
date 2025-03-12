import streamlit as st
from tools import arxiv, wiki_tool, llm
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import  FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools.retriever import create_retriever_tool 
import streamlit as st 
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_google_genai import ChatGoogleGenerativeAI

st.write("This is an Agentic RAG application powered by Wikipedia, ArXiv, Google Scholar, and custom-added websites for enhanced information retrieval. Updateding in progress")

# Initialize session state variables
if "base_model" not in st.session_state:
    st.session_state.base_model = "gemma2-9b-it"

if "tools" not in st.session_state:
    st.session_state.tools = []

if "url" not in st.session_state:
    st.session_state.url = ""

if st.session_state.url != "":
    loader = WebBaseLoader(st.session_state.url)
    docs=loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorDB = FAISS.from_documents(documents, embeddings) 
    retriever = vectorDB.as_retriever()
    retriever_tool = create_retriever_tool(
    retriever,
    "url_search",
    "",
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Customization
with st.sidebar:
    st.title("Agentic RAG")
    st.markdown("Customization")

    with st.form(key="model_form"):
        model = st.selectbox("Select a model", ["llama-3.3-70b-versatile", "llama3-70b-8192", "gemini-2.0-flash", "gemma2-9b-it"])
        set_model = st.form_submit_button("Change model")
        if set_model:
            st.session_state.base_model = model
            st.success("Model updated")

        wiki = st.toggle("Wikipedia")
        arx = st.toggle("ArXiv")
        scholar = st.toggle("Google Scholar")

        link1 = st.text_input("Enter the URL: ")
        add_link = st.form_submit_button("Add Link")
        if add_link and link1!="":
            st.session_state.url = link1
            st.success("Link added")

        clear_chat = st.form_submit_button("Clear Chat")
        if clear_chat:
            st.session_state.chat_history = []
            st.success("Chat history cleared")

# Manage tools dynamically
st.session_state.tools = []
if wiki:
    st.session_state.tools.append(wiki_tool)
if arx:
    st.session_state.tools.append(arxiv)
if scholar:
    from tools import google_scholar_tool  # Import only if needed
    st.session_state.tools.append(google_scholar_tool)
if st.session_state.url:
    from tools import retriever_tool
    st.session_state.tools.append(retriever_tool)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Load the prompt for the agent
prompt = hub.pull("hwchase17/openai-functions-agent")

# Create the agent
agent = create_openai_tools_agent(llm, st.session_state.tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=st.session_state.tools, verbose=False)

# User input handling
user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)

    response = agent_executor.invoke({"input": user_input})["output"]  # Extract response correctly
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.write(response)

