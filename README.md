# Documentation RAG with Open WebUI

This project combines Open WebUI (using Ollama with Gemma model) with RAG capabilities to provide intelligent responses based on documentation.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Open WebUI installed and running locally
- Gemma model pulled in Ollama

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your configuration:
```
OPENWEBUI_URL=http://localhost:3000
OLLAMA_URL=http://localhost:11434
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## Open WebUI Integration

1. Start Open WebUI:
```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

2. Access Open WebUI at http://localhost:3000

3. In Open WebUI:
   - Go to Settings > API Keys
   - Create a new API key
   - Copy the API key and add it to your `.env` file:
     ```
     OPENWEBUI_API_KEY=your_api_key_here
     ```

4. Configure the model in Open WebUI:
   - Go to Models
   - Add a new model with the following settings:
     - Name: `gemma-rag`
     - Base URL: `http://localhost:8000`
     - Model Type: Custom
     - Context Length: 4096
     - Temperature: 0.7

5. Use the model in Open WebUI:
   - Create a new chat
   - Select the `gemma-rag` model
   - Start chatting with your documentation-aware model

## Usage

1. First, index your documentation by providing the URL:
```bash
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" -d '{"url": "YOUR_DOCUMENTATION_URL"}'
```

2. Then query the system:
```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "YOUR_QUESTION"}'
```

## Features

- RAG-based documentation search
- Integration with Open WebUI and Ollama
- Uses Gemma model for responses
- FastAPI-based REST API
- ChromaDB for vector storage 