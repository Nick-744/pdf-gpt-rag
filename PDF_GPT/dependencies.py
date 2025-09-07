'''
Dependency checking module
---------------------------------------------
Handles all dependency imports and validation
with proper error handling!
'''

from typing import List

# --- Dependency checks (FAIL SUPER FAST) --- #
_MISSING: List[str] = []

try:
    import torch
except Exception:
    _MISSING.append('torch')

try:
    # Core LlamaIndex
    from llama_index.core import (
        SimpleDirectoryReader, # Loads documents from a directory
        VectorStoreIndex,      # Create a searchable knowledge base
        StorageContext,        # Persist data across sessions (+ ChromaDB)
        PromptTemplate,        # Template for how the LLM should respond!
        Settings               # Configuration settings
    )
    # LlamaIndex is a comprehensive data framework designed
    # to help build applications that can connect
    # LLMs with specific data source!

    # Embedding + LLM
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM

    # Vector store
    from llama_index.vector_stores.chroma import ChromaVectorStore
    # A LlamaIndex adapter that wraps the ChromaDB
    # collection to work with LlamaIndex's ecosystem.
except Exception:
    _MISSING.append('llama-index (+ subpackages)')

try:
    import chromadb
    # Used to create and manage the underlying
    # ChromaDB client and collections.
except Exception:
    _MISSING.append('chromadb')

# Check for missing dependencies and raise error if any are missing!
if _MISSING:
    missing_list = ', '.join(_MISSING)
    msg          = (
        f'Missing required dependencies: {missing_list}\n'
        'Please install the requirements and try again!\n'
    )

    raise RuntimeError(msg);

# Export all the imports for use in other modules...
__all__ = [
    'torch',
    
    'SimpleDirectoryReader',
    'VectorStoreIndex',
    'StorageContext',
    'PromptTemplate',
    'Settings',

    'HuggingFaceEmbedding',
    'HuggingFaceLLM',
    
    'ChromaVectorStore',
    'chromadb'
]
