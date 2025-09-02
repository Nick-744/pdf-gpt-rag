# PDF RAG Chatbot

Chat with your PDF documents using RAG (Retrieval-Augmented Generation) powered by LlamaIndex and Hugging Face models.

## Installation

```bash
pip install -r requirements.txt
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

Simply run the chatbot to start an interactive session:

```bash
python main.py
```

The chatbot will load the PDF document and start an interactive chat session. Type your questions and get answers based on the document content.
