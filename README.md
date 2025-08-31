# Simple PDF RAG Chatbot

Chat with your PDF documents using RAG (Retrieval-Augmented Generation) powered by LlamaIndex and Hugging Face models.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Simply run the chatbot to start an interactive session:

```bash
python main.py
```

The chatbot will load the default PDF document and start an interactive chat session. Type your questions and get answers based on the document content.

Commands during chat:
- Type your question and press Enter
- Type 'quit', 'exit', or 'q' to exit
- Type 'reset' to clear conversation history

## Configuration

You can modify the default settings by editing the configuration in `PDF_GPT/config.py`:

- Change the default PDF path
- Adjust model settings (LLM and embedding models)
- Modify RAG parameters (chunk size, top-k retrieval, temperature)
- Set persistence directory for faster startup

## Requirements

- Python 3.8+
- ~8GB RAM minimum
- Optional: CUDA-compatible GPU for larger models
