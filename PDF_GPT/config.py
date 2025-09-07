from dataclasses import dataclass

@dataclass
class RAGConfig:
    # --- Models --- #
    model_name:       str = 'Qwen/Qwen2.5-1.5B-Instruct'
    # Hugging Face ID of the LLM used to answer questions.
    'Qwen/Qwen2.5-1.5B-Instruct'
    'Qwen/Qwen2.5-3B-Instruct'

    embed_model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    # Hugging Face ID of the sentence embedding model used to
    # vectorize chunks for retrieval.
    'sentence-transformers/all-mpnet-base-v2'
    'BAAI/bge-large-en-v1.5'



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
    temperature:    float = 0.01
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
    # IMPORTANT: Set to True after changing embed_model_name (dimension change)
    # so the Chroma collection is rebuilt! Once rebuilt you can toggle back to
    # False for faster startups...



    # --- Behavior --- #
    custom_prompt: str  = '''
You are a helpful assistant that answers questions based ONLY on the provided context from a PDF document.

Instructions:
- Answer questions directly and concisely based only on the provided context
- Do not show your reasoning process or intermediate steps
- Do not repeat or rephrase the user's question
- If you cannot find relevant information in the context, say "I don't have enough information in the provided document to answer that question."
- Provide specific details when available, including page references if mentioned in the context

Context: {context_str}

Question: {query_str}

Answer:'''

    verbose:       bool = False
    # If True, enables more detailed logging / internal traces (propagated
    # to LlamaIndex chat engine).
