# 🤖 AI Research Assistant

An intelligent **AI-powered Research and Document Assistant** built with LangChain, Streamlit, Groq, FAISS, and Hugging Face embeddings.

The application combines **Agentic AI with Retrieval-Augmented Generation (RAG)**, allowing users to chat normally, upload multiple PDF documents, search academic research, and retrieve external information using AI-powered tools.

The AI agent automatically decides which tool to use based on the user's question.

---

## ✨ Features

### 💬 AI-Powered Chat

Ask general questions and interact with an AI assistant powered by an LLM through Groq.

The agent can answer directly when no external information is required.

### 📄 Multi-PDF Question Answering

Upload one or multiple PDF documents and ask questions about their contents.

The system:

1. Loads uploaded PDFs
2. Extracts document text
3. Splits the text into chunks
4. Generates semantic embeddings
5. Stores embeddings in FAISS
6. Retrieves relevant information based on the user's question
7. Uses the retrieved context to generate an answer

PDF upload is completely optional. The assistant can still answer normal questions without any uploaded documents.

### 🧠 Agentic Tool Selection

The application uses a LangChain tool-calling agent that automatically determines which source or tool should be used.

Depending on the question, the agent can:

- Answer directly using the LLM
- Search uploaded PDFs
- Search academic papers
- Search external/web information
- Combine information from multiple tools

The user does not need to manually select a mode.

### 🔍 Semantic Document Search

Uploaded documents are converted into vector embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

The embeddings are stored in a **FAISS vector database**, enabling semantic search across document content.

### 📚 Academic Research Search

The agent can use **Arxiv** to retrieve information about:

- Research papers
- Machine Learning
- Deep Learning
- Generative AI
- LLMs
- Computer Science
- Other academic topics

### 🌐 External Information Tools

The AI agent can use external search/research tools when the required information is not available in uploaded documents or when additional information is needed.

### 🗑️ Document Management

Users can remove uploaded PDFs and clear their associated vector data when the documents are no longer needed.

---

## 🏗️ Architecture

```text
                    User
                      │
                      ▼
               Streamlit UI
                      │
            ┌─────────┴─────────┐
            │                   │
       User Question       PDF Upload
            │                   │
            │                   ▼
            │             PDF Loader
            │                   │
            │                   ▼
            │              Text Splitter
            │                   │
            │                   ▼
            │          Hugging Face Embeddings
            │                   │
            │                   ▼
            │                 FAISS
            │                   │
            └─────────┬─────────┘
                      ▼
               LangChain Agent
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
     PDF Search     Arxiv      Other Tools
          │           │           │
          └───────────┼───────────┘
                      ▼
                   Groq LLM
                      │
                      ▼
                 Final Answer
