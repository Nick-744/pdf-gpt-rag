''' Configuration '''

from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    ''' Configuration for the RAG. '''
    
    # Models
    model_name:       str           = 'Qwen/Qwen2.5-1.5B-Instruct'
    embed_model_name: str           = 'sentence-transformers/all-mpnet-base-v2'
    hf_token:         Optional[str] = None # or 'your_huggingface_token'

    # Chunking and retrieval
    chunk_size:    int = 1024
    chunk_overlap: int = 200
    top_k:         int = 5

    # Generation
    temperature:    float = 0.1
    max_new_tokens: int   = 512

    # Chroma settings
    persist_dir:     str  = './CHROMA_DB'
    collection_name: str  = 'pdf_rag_collection'
    reset_index:     bool = True

    # Hardware
    device: str = 'auto' # 'auto', 'cuda', 'cpu'

    # Behavior
    show_sources: bool = False
    verbose:      bool = False

'''
model_name:       Hugging Face ID of the chat/generation LLM used to answer
                  questions.
embed_model_name: Hugging Face ID of the sentence embedding model used to
                  vectorize chunks for retrieval.
hf_token:         Optional Hugging Face access token (string or None) used
                  to authenticate for gated/private models.

chunk_size:    Number of characters (or tokens depending on splitter) per
               document chunk fed into embeddings / index.
chunk_overlap: Characters overlapped between consecutive chunks to preserve
               context continuity.
top_k:         Number of most similar chunks (vectors) retrieved per query
               for context passed to the LLM.

temperature:    Sampling randomness for text generation; lower = more deterministic.
max_new_tokens: Maximum number of tokens the LLM may generate for an answer.

persist_dir:     Filesystem path where the Chroma vector database is stored
                 (enables reuse across runs).
collection_name: Name of the Chroma collection inside the persistence directory.
reset_index:     If True, deletes any existing Chroma collection and rebuilds
                 from source PDFs.

device: Execution target for models.

show_sources: If True, append page/source identifiers to generated answers.
verbose:      If True, enables more detailed logging / internal traces (propagated
              to LlamaIndex chat engine).
'''
