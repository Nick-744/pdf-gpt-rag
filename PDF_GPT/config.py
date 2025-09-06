from dataclasses import dataclass

@dataclass
class RAGConfig:
    # --- Models --- #
    model_name:       str = 'Qwen/Qwen2.5-1.5B-Instruct'
    # Hugging Face ID of the LLM used to answer questions.

    embed_model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    # Hugging Face ID of the sentence embedding model used to
    # vectorize chunks for retrieval.



    # --- Chunking and retrieval --- #
    chunk_size:    int = 1024
    # Number of characters/tokens per document chunk fed into embeddings.

    chunk_overlap: int = 200
    # Characters/tokens overlapped between consecutive chunks to preserve
    # context continuity.

    top_k:         int = 5
    # Number of most similar chunks (vectors) retrieved per query
    # for context passed to the LLM.



    # --- Generation --- #
    temperature:    float = 0.1
    # Sampling randomness for text generation - lower = more deterministic.

    max_new_tokens: int   = 512
    # Maximum number of tokens the LLM may generate for an answer.

    context_window: int   = 2048
    # Maximum number of tokens the LLM can process in its context window.
    # - Smaller context = faster processing, less memory
    # - Larger context  = better understanding of long documents



    # --- Chroma settings --- #
    persist_dir:     str  = './CHROMA_DB'
    # Filesystem path where the Chroma vector database is stored
    # (enables reuse across runs).

    collection_name: str  = 'pdf_rag_collection'
    # Name of the Chroma collection inside the persistence directory.

    reset_index:     bool = False
    # If True, deletes any existing Chroma collection and rebuilds
    # from source PDF.



    # --- Behavior --- #
    show_sources: bool = True
    # If True, append page/source identifiers to generated answers.

    verbose:      bool = True
    # If True, enables more detailed logging / internal traces (propagated
    # to LlamaIndex chat engine).
