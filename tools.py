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

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if "url" not in st.session_state:
    st.session_state.url = ""

if "base_model" not in st.session_state:
    st.session_state.base_model = "gemma2-9b-it"

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

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
    



from langchain_groq import ChatGroq


llm = ChatGroq(model_name=st.session_state.base_model, temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))