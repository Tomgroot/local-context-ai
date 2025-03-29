# RAG with Open WebUI and ollama

This project combines Open WebUI (using Ollama) with RAG capabilities to provide intelligent responses based on local markdown documentation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
# You might also need unstructured for markdown parsing:
pip install "unstructured[md]"
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
   - Go to Settings > Admin settings
   - Under "Connections" add a Ollama API entry to http://host.docker.internal:8000 and disable the original Ollama connection if there is any.  

## Usage

1. Create a directory named `documents` in the project root.
2. Place your documentation files (in Markdown format, `.md`) inside the `documents` directory.
3. Indexing happens when the application is started and there is no `chrome_db` directory.
This will scan the `documents` directory, process the `.md` files, and store their content in the vector database.