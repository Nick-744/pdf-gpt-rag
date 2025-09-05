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
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    )
except Exception:
    _MISSING.append('transformers')

try:
    # Core LlamaIndex
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        StorageContext,
        Settings
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

# Check for missing dependencies and raise error if any are missing
if _MISSING:
    missing_list = ', '.join(_MISSING)
    msg          = (
        f'Missing required dependencies: {missing_list}\n'
        'Please install the requirements and try again!\n'
    )

    raise RuntimeError(msg);

# Export all the imports for use in other modules
__all__ = [
    'SimpleDirectoryReader',
    'VectorStoreIndex',
    'StorageContext',
    'Settings',

    'HuggingFaceEmbedding',
    'HuggingFaceLLM',
    
    'ChromaVectorStore',
    'chromadb'
]
