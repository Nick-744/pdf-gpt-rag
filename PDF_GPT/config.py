'''
Configuration for PDF RAG Chatbot System
-------------------------------------------------
Simple configuration with sensible defaults.
'''

import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class RAGConfig:
    '''Simple configuration for the RAG chatbot.'''
    
    # Models - can be overridden via environment variables
    model_name:       str           = os.environ.get(
        'RAG_LLM_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'
    )
    embed_model_name: str           = os.environ.get(
        'RAG_EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2'
    )
    hf_token:         Optional[str] = os.environ.get('HUGGINGFACE_TOKEN')

    # Chunking and retrieval
    chunk_size:    int = 1024
    chunk_overlap: int = 200
    top_k:         int = 5

    # Generation
    temperature:    float = 0.7
    max_new_tokens: int   = 512

    # Chroma settings
    persist_dir:     str  = './CHROMA_DB'
    collection_name: str  = 'pdf_rag_collection'
    reset_index:     bool = False

    # Hardware
    device: str = 'auto' # 'auto', 'cuda', 'cpu'

    # Behavior
    show_sources: bool = False
    verbose:      bool = False
