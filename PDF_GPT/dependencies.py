'''
Dependency checking module for PDF RAG Chatbot System
-------------------------------------------------
Handles all dependency imports and validation with proper error handling.
'''

from typing import List

# ---------------------------
# Dependency checks (fail fast)
# ---------------------------
_MISSING: List[str] = []

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM # noqa: F401
except Exception:
    _MISSING.append('transformers')

try:
    # Core LlamaIndex
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        Settings,
        StorageContext,
        load_index_from_storage,
    )
    # Embeddings + LLMs
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
    # Vector stores
    from llama_index.vector_stores.chroma import ChromaVectorStore
except Exception:
    _MISSING.append('llama-index (+ subpackages)')

try:
    import chromadb
except Exception:
    _MISSING.append('chromadb')

try:
    from huggingface_hub import login as hf_login # noqa: F401
except Exception:
    _MISSING.append('huggingface_hub')

# Check for missing dependencies and raise error if any are missing
if _MISSING:
    missing_list = ', '.join(_MISSING)
    msg          = (
        f'Missing required dependencies: {missing_list}\n'
        'Please install the requirements and try again.\n'
        'Example:\n'
        '  pip install transformers torch llama-index '
        'llama-index-llms-huggingface llama-index-embeddings-huggingface '
        'llama-index-vector-stores-chroma chromadb '
        'sentence-transformers huggingface_hub'
    )

    raise RuntimeError(msg);

# Export all the imports for use in other modules
__all__ = [
    'VectorStoreIndex',
    'SimpleDirectoryReader', 
    'Settings',
    'StorageContext',
    'load_index_from_storage',
    'HuggingFaceEmbedding',
    'HuggingFaceLLM',
    'ChromaVectorStore',
    'chromadb',
    'hf_login'
]
