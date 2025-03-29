from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_playwright.document_loaders import PlaywrightWebBaseLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader

# Load environment variables
load_dotenv()

app = FastAPI(title="Documentation RAG with Open WebUI")

# Configuration
OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://localhost:3000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "secret")

# Security
api_key_header = APIKeyHeader(name="Authorization")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != f"Bearer {OPENWEBUI_API_KEY}":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
vectorstore = None

class IndexRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

class OpenWebUIRequest(BaseModel):
    messages: List[dict]
    stream: bool = False

@app.post("/index")
async def index_documentation(request: IndexRequest):
    try:
        # Load documentation using PlaywrightWebBaseLoader
        # It takes a list of URLs
        loader = PlaywrightWebBaseLoader(urls=[request.url], load_timeout=120) # Increased timeout for potentially slow JS pages
        
        # Use load_and_split to potentially improve efficiency with Playwright
        splits = await loader.aload() # Use async load

        # No need to print docs unless debugging
        # print(splits) 
        
        if not splits:
            raise HTTPException(status_code=400, detail="Could not extract any content from the URL. The page might be empty or protected.")
        
        # Create vector store
        global vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return {"message": "Documentation indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documentation(request: QueryRequest):
    if not vectorstore:
        raise HTTPException(status_code=400, detail="Please index documentation first")
    
    try:
        # Initialize Ollama with Gemma model
        llm = Ollama(
            base_url=OLLAMA_URL,
            model="gemma3:12b",
            temperature=0.7
        )
        
        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        # Get response
        response = qa_chain.run(request.question)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def openwebui_chat(request: OpenWebUIRequest, api_key: str = Depends(verify_api_key)):
    if not vectorstore:
        raise HTTPException(status_code=400, detail="Please index documentation first")
    
    try:
        # Get the last message from the conversation
        last_message = request.messages[-1]["content"]
        
        # Initialize Ollama with Gemma model
        llm = Ollama(
            base_url=OLLAMA_URL,
            model="gemma3:12b",
            temperature=0.7
        )
        
        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )
        
        # Get response
        response = qa_chain.run(last_message)
        
        return {
            "id": "chatcmpl-" + str(hash(response)),
            "object": "chat.completion",
            "created": 0,
            "model": "gemma-rag",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        if response.status_code != 200:
            raise HTTPException(status_code=503, detail="Ollama service is not available")
        
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) 