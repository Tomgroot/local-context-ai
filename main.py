from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import requests
from starlette.middleware.base import BaseHTTPMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from fastapi.responses import StreamingResponse
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ollama import chat
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import glob
import httpx
from datetime import datetime, timezone
import logging
import time
from fastapi import Request
import traceback
import asyncio
import json

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Documentation RAG with Open WebUI")

OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://127.0.0.1:3000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
vectorstore = None

class LogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()

        # Reconstruct request stream
        async def receive():
            return {"type": "http.request", "body": body}

        request = Request(request.scope, receive)

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        logger.info(
            f"data={body.decode('utf-8')} "
            f"client={request.client.host}:{request.client.port} "
            f"method={request.method} path={request.url.path} "
            f"status_code={response.status_code} process_time={process_time:.4f}s"
        )
        return response

# Add this middleware
app.add_middleware(LogRequestMiddleware)


class OpenWebUIRequest(BaseModel):
    messages: List[dict]
    stream: bool
    model: str

    class Config:
        extra = "ignore"

class QueryRequest(BaseModel):
    prompt: str

@app.on_event("startup")
async def startup_event():
    await index_documentation()

@app.post("/index")
async def index_documentation():
    try:
        documents_to_index = []
        
        global vectorstore
        persist_directory = "./chroma_db"

        if os.path.exists(persist_directory):
            logger.info(f"Loading existing vector store from {persist_directory}")
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return {"message": "Documentation already indexed."}

        doc_dir = "./documents"
        if os.path.exists(doc_dir) and os.path.isdir(doc_dir):
            patterns = [os.path.join(doc_dir, '**', '*.md'), os.path.join(doc_dir, '**', '*.mdx')]
            md_files = []
            for pattern in patterns:
                md_files.extend(glob.glob(pattern, recursive=True))

            # Deduplicate in case a file somehow matches both (unlikely but safe)
            md_files = list(set(md_files))

            for md_file in md_files:
                try:
                    loader = UnstructuredMarkdownLoader(md_file)
                    md_docs = loader.load()
                    documents_to_index.extend(md_docs)
                except Exception as e:
                    logger.error(f"Error loading markdown file {md_file}: {e}") # Log error and continue

        if not documents_to_index:
            return {"message": "No documents found in ./documents directory to index."}

        # 3. Split documents
        splits = text_splitter.split_documents(documents_to_index)
        
        if not splits:
             return {"message": "No text content found in the documents to index."}
        
        logger.info(f"Creating new vector store in {persist_directory}")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"Created vector store with {len(splits)} document chunks.")

        return {"message": f"Documentation indexed successfully. Added {len(splits)} chunks."}
    except Exception as e:
        # Log the full exception for debugging
        logger.error("Error during indexing:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during indexing: {str(e)}")

@app.post("/api/chat")
async def api_chat(request: OpenWebUIRequest):
    # Log the Ollama URL being used for this request
    logger.info(f"Attempting to connect to Ollama at: {OLLAMA_URL}")
    logger.info(f"Received /api/chat request: {request.dict(exclude={'messages'}) if request.messages and len(request.messages) > 5 else request.dict()}")

    if not vectorstore:
        raise HTTPException(status_code=400, detail="Vector store not initialized. Please index documentation first via POST /index.")

    # Extract the prompt from the last user message
    user_prompt = None
    if request.messages:
        for message in reversed(request.messages):
            if message.get("role") == "user" and message.get("content"):
                user_prompt = message["content"]
                break

    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user prompt found in messages")

    try:
        # --- Streaming Logic ---
        if request.stream:
            callback = AsyncIteratorCallbackHandler()
            logger.info(f"Creating Ollama instance for streaming model {request.model} at {OLLAMA_URL}")
            llm = Ollama(
                base_url=OLLAMA_URL,
                model=request.model,
                temperature=0.7, # Or use temperature from request if provided
                callbacks=AsyncCallbackManager([callback]),
                # verbose=True # Uncomment for debugging langchain calls
            )

            # Retrieve relevant documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = await retriever.aget_relevant_documents(user_prompt)
            context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents for context.")

            # Define the RAG prompt
            template = """The following piece of context can help to answer the question. Keep the answer concise, helpful and professional.

Context:
{context}

Question: {question}

Answer:"""
            rag_prompt = PromptTemplate.from_template(template)

            # Create LLMChain for streaming
            chain = LLMChain(llm=llm, prompt=rag_prompt)

            # Define the async generator for streaming response
            async def stream_rag_response(context: str, question: str):
                # Log before starting the background task
                logger.info(f"Starting background task chain.arun for question: {question[:50]}...")
                # Start the chain run in the background
                task = asyncio.create_task(
                    chain.arun(context=context, question=question)
                )
                # Yield chunks as they become available
                try:
                    async for token in callback.aiter():
                        chunk = {
                            "model": llm.model,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "message": {
                                "role": "assistant",
                                "content": token
                            },
                            "done": False
                        }
                        yield f"{json.dumps(chunk)}\n"

                    # Wait for the background task to finish
                    await task
                    # Send final done message
                    final_chunk = {
                        "model": llm.model,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {
                            "role": "assistant",
                            "content": "" # No final content needed, already streamed
                        },
                        "done": True
                    }
                    yield f"{json.dumps(final_chunk)}\n"

                except Exception as e:
                    logger.error(f"Error during RAG streaming: {e}", exc_info=True)
                    # You might want to yield an error message chunk here
                    error_chunk = {
                        "error": f"An error occurred during streaming: {str(e)}",
                        "done": True
                    }
                    yield f"{json.dumps(error_chunk)}\n"
                finally:
                     # Ensure the iterator is marked as done
                    callback.done.set()

            return StreamingResponse(
                stream_rag_response(context=context, question=user_prompt),
                media_type="application/x-ndjson"
            )

        # --- Non-Streaming Logic (Existing Fallback) ---
        else:
            logger.info(f"Creating Ollama instance for non-streaming model {request.model} at {OLLAMA_URL}")
            llm = Ollama(
                base_url=OLLAMA_URL,
                model=request.model,
                temperature=0.7
            )

            # Create RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
                # verbose=True # Uncomment for debugging langchain calls
            )

            # Run the chain synchronously for a single response
            logger.info(f"Running qa_chain.run for question: {user_prompt[:50]}...")
            answer = qa_chain.run(user_prompt)
            logger.info(f"Generated non-streamed answer for model {llm.model}")
            return {
                "model": llm.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "done": True
            }

    except Exception as e:
        # Log the full traceback for detailed debugging
        logger.error(f"Error in /api/chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.get("/api/version")
async def get_api_version():
    """Returns the API version."""
    return {"version": "1.0.0"}

@app.get("/api/tags")
async def get_ollama_tags():
    """Proxies the Ollama /api/tags endpoint."""
    ollama_tags_url = f"{OLLAMA_URL}/api/tags"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(ollama_tags_url)
            response.raise_for_status() # Raises exception for 4xx/5xx responses
            return response.json() # Return Ollama's response directly
    except httpx.RequestError as e:
        logger.error(f"Error requesting Ollama /api/tags: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to Ollama at {OLLAMA_URL}")
    except httpx.HTTPStatusError as e:
        # Forward Ollama's error status and detail if possible
        detail = f"Ollama API error: {e.response.status_code}"
        try:
            ollama_detail = e.response.json().get("error", "Unknown error")
            detail = f"Ollama API error: {e.response.status_code} - {ollama_detail}"
        except Exception:
            pass # Keep the basic detail message
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        logger.error("Unexpected error in /api/tags proxy:", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error proxying Ollama /api/tags")

@app.get("/health")
async def health_check():
    """Checks connectivity to the Ollama service using httpx."""
    logger.info(f"Performing health check for Ollama at {OLLAMA_URL}")
    ollama_api_endpoint = f"{OLLAMA_URL}/api/tags" # Use a valid API endpoint
    try:
        # Use httpx.AsyncClient for consistency with async operations
        async with httpx.AsyncClient(timeout=5.0) as client: # Add a timeout
            response = await client.get(ollama_api_endpoint)

        # Check if the response status code indicates success (e.g., 200 OK)
        if response.status_code == 200:
            logger.info(f"Health check successful: Ollama responded from {ollama_api_endpoint}")
            return {"status": "healthy", "ollama_url": OLLAMA_URL, "ollama_response_status": response.status_code}
        else:
            # Log the unexpected status code
            logger.warning(f"Health check warning: Ollama at {ollama_api_endpoint} responded with status {response.status_code}")
            # Still consider it "unhealthy" from the perspective of expecting a 200
            raise HTTPException(status_code=503, detail=f"Ollama service responded with status {response.status_code} from {ollama_api_endpoint}")

    except httpx.RequestError as e:
        # This catches connection errors, DNS errors, timeouts, etc.
        error_message = f"Health check failed: Could not connect to Ollama at {OLLAMA_URL}. Error: {type(e).__name__} - {e}"
        logger.error(error_message)
        # Raise HTTPException with 503 Service Unavailable
        raise HTTPException(status_code=503, detail=error_message)
    except Exception as e:
        # Catch any other unexpected errors during the health check
        error_message = f"Health check failed: An unexpected error occurred. Error: {type(e).__name__} - {e}"
        logger.error(error_message, exc_info=True) # Log full traceback for unexpected errors
        raise HTTPException(status_code=500, detail=error_message) 