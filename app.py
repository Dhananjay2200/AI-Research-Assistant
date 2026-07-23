import streamlit as st
import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from tools import Tools
from langchain_classic.tools.retriever import create_retriever_tool
from langsmith import Client
import streamlit as st
from langchain_classic.agents import create_tool_calling_agent 
from langchain_classic.agents import AgentExecutor
import time
import shutil

load_dotenv()
api_key = os.getenv("GROQ")
if not api_key:
    api_key = st.secrets.get("GROQ")

PDF_FOLDER = './documents'
os.makedirs(PDF_FOLDER,exist_ok=True)
st.title("AI Research Assistant")
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)   
# save pdf 
if uploaded_files:
    for u_f in uploaded_files:
        file_path = os.path.join(
            PDF_FOLDER,
            u_f.name
        )

        with open(file_path,"wb") as f:
            f.write(u_f.getbuffer())
@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
def vector_embeddings():
    
        
    if "vectors " not in st.session_state:
        st.session_state.embedding = load_embedding_model()
                                                           
        st.session_state.loader = PyPDFDirectoryLoader(PDF_FOLDER)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) 
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embedding)

tools = list(Tools)
if uploaded_files:
    with st.spinner("Processing PDFs and creating embeddings... Please wait."):
            vector_embeddings()

            st.success("✅ Documents are ready! You can now ask questions.")
    
    pdf_retriever = st.session_state.vectors.as_retriever(search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20})

    desc = """
    Search all uploaded PDF documents for information relevant to the user's question.

Use this tool whenever:
- The user asks about information contained in an uploaded PDF.
- The user asks about a course name, course title, syllabus, topic, person,
  project, skill, experience, date, company, or any other detail that may
  appear in an uploaded document.
- The user refers to "my PDF", "this PDF", "uploaded document", "my resume",
  "my course", "the document", or similar phrases.
- Information may need to be combined from different parts or pages of
  one or more PDFs.

Search the PDFs before assuming that information is missing.
If the first retrieved chunks are insufficient, use the available context
carefully and do not invent facts that are not present in the documents.

    """
    pdf_tool = create_retriever_tool(pdf_retriever,'pdf_search',desc)

    tools.append(pdf_tool)

# prompt
client = Client()
prompt = client.pull_prompt("hwchase17/openai-functions-agent",
                  dangerously_pull_public_prompt=True)

## llm
llm = ChatGroq(groq_api_key=api_key,
               model_name = "openai/gpt-oss-20b",
               temperature=0)

## Agents
agent = create_tool_calling_agent(llm = llm,
                                  tools = tools,
                                  prompt = prompt)
agent_executor = AgentExecutor(agent = agent,
                               tools = tools,
                               verbose=True)

prompt = st.text_input("what's in your mind")

if prompt:
    start = time.process_time()
    response = agent_executor.invoke({'input':prompt})
    print(f"Response Time:{time.process_time() - start}")
    st.write(response['output'])

     # Delete PDFs button
    if st.button("Delete Uploaded PDFs"):

        # Delete the whole documents folder
        if os.path.exists(PDF_FOLDER):
            shutil.rmtree(PDF_FOLDER)

        # Recreate empty folder for next upload
        os.makedirs(PDF_FOLDER, exist_ok=True)

        # Delete old FAISS/vector-related session data
        keys_to_delete = [
            "vectors",
            "embedding",
            "loader",
            "docs",
            "text_splitter",
            "final_docs"
        ]

        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]

        st.success("Uploaded PDFs and vector database deleted.")

        st.rerun()

    










