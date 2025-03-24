from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import  FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Arxiv Tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
import streamlit as st

st.set_page_config(page_title="Chat with Me", page_icon="💬")

st.title("💬 Chat with Me")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "url" not in st.session_state:
    st.session_state.url = ""
    
if "dec" not in st.session_state:
    st.session_state.dec = ""

with st.sidebar:
    web_URL = st.text_input("Enter URL")
    desc_URL = st.text_input("Describe URL")
    submit_button = st.button("Try this URL")
    if submit_button:
        if web_URL=="":
            st.error("Please enter a web URL or URL is none specified")
        st.session_state.url  = web_URL
        st.session_state.dec  = desc_URL
        st.success("Done!")



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)



if st.session_state.url !="": 
    st.write(str(st.session_state.url))
    loader = WebBaseLoader(str(st.session_state.url))
    docs=loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorDB = FAISS.from_documents(documents, embeddings) 
    retriever = vectorDB.as_retriever()

    from langchain.tools.retriever import create_retriever_tool 

    retriever_tool = create_retriever_tool(
        retriever,
        "webURl",
        st.session_state.dec,
    )






if st.session_state.url != "":
    tools = [wiki_tool,arxiv,retriever_tool]
else:
    tools = [wiki_tool,arxiv]


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=1)

from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")


# Agents

from langchain.agents import create_openai_tools_agent

agent = create_openai_tools_agent(llm,tools,prompt)


## Agent Executer

from langchain.agents import AgentExecutor

agent_executer = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)


# agent_executer.invoke({"input":"Tell me about few names of university of jaffna computer science students Academic Year: 2019/2020"})







for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    response = agent_executer.invoke({"input": user_prompt})
    bot_response = response["output"]

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})

