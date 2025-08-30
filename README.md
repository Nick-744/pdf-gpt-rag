# PDF RAG Chatbot

Chat with your PDF documents using RAG (Retrieval-Augmented Generation) powered by LlamaIndex and Hugging Face models.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Chat
```bash
python main.py --pdf_path document.pdf
```

### Single Question
```bash
python main.py --pdf_path document.pdf --query "What is this about?"
```

### Multiple PDFs
```bash
python main.py --pdf_path doc1.pdf doc2.pdf doc3.pdf
```

## Configuration

Set via environment variables or CLI flags:

```bash
# Models
export RAG_LLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export RAG_EMBED_MODEL="sentence-transformers/all-mpnet-base-v2"
export HUGGINGFACE_TOKEN="your_token"

# Settings  
export RAG_CHUNK_SIZE="1024"
export RAG_TOP_K="5"
export RAG_TEMPERATURE="0.7"
export RAG_PERSIST_DIR="./index"
```

## Programmatic API

```python
from api import ChatbotAPI

api = ChatbotAPI("document.pdf")
result = api.ask("What is the main topic?")
print(result["response"])
```

## Key Options

- `--show_sources`: Include page numbers
- `--persist_dir`: Save/load index for faster startup
- `--temperature`: Control response randomness (0.0-1.0)
- `--top_k`: Number of retrieved chunks
- `--json`: JSON output format

## Requirements

- Python 3.8+
- ~4GB RAM minimum
- Optional: CUDA-compatible GPU for larger models
