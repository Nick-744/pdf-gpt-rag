'''
Configuration module for PDF RAG Chatbot System
-------------------------------------------------
Contains the RAGConfig dataclass and environment variable helpers.
'''

import os
from typing import Optional
from dataclasses import dataclass, field

def _env(key: str, default: str) -> str:
    '''Helper function to get environment variables with defaults.'''
    return os.environ.get(key, default);

@dataclass
class RAGConfig:
    '''Configuration class for the RAG system with environment variable defaults.'''
    
    # Models
    model_name:       str           = field(default_factory = lambda: _env('RAG_LLM_MODEL',   'Qwen/Qwen2.5-1.5B-Instruct'             ))
    embed_model_name: str           = field(default_factory = lambda: _env('RAG_EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2'))
    hf_token:         Optional[str] = field(default_factory = lambda: os.environ.get('HUGGINGFACE_TOKEN'))

    # Chunking + Retrieval
    chunk_size:    int = field(default_factory = lambda: int(_env('RAG_CHUNK_SIZE',    '1024')))
    chunk_overlap: int = field(default_factory = lambda: int(_env('RAG_CHUNK_OVERLAP', '200' )))
    top_k:         int = field(default_factory = lambda: int(_env('RAG_TOP_K',         '5'   )))

    # Generation
    temperature:    float = field(default_factory = lambda: float(_env('RAG_TEMPERATURE', '0.7')))
    max_new_tokens: int   = field(default_factory = lambda: int(_env('RAG_MAX_NEW_TOKENS', '512')))

    # Index persistence
    persist_dir: Optional[str] = field(default_factory = lambda: _env('RAG_PERSIST_DIR', '')) # empty -> no persistence
    reset_index: bool          = False  # force rebuild even if persist_dir exists

    # Behavior flags
    show_sources: bool = False
    verbose:      bool = False # controls LlamaIndex internal verbosity
