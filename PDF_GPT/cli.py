'''
Command-line interface module for PDF RAG Chatbot System
-------------------------------------------------
Handles argument parsing and CLI-specific functionality.
'''

import os
import argparse
from typing import List, Optional

from .config import RAGConfig, _env

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    '''Parse command-line arguments for the PDF RAG chatbot.'''
    p = argparse.ArgumentParser(description='PDF RAG Chatbot (LlamaIndex + HuggingFace)')

    p.add_argument(
        '--pdf_path',
        nargs    = '+',
        required = False,
        default  = [
            _env('RAG_DEFAULT_PDF', 'C:\\Users\\nick1\\Documents\\GitHub\\pdf-gpt-rag\\PDF_SOURCE\\Understanding_Climate_Change.pdf')
        ],
        help     = 'Path(s) to PDF file(s). Provide one or more paths.',
    )
    p.add_argument('--persist_dir', default = _env('RAG_PERSIST_DIR', ''), help = 'Directory to persist/load the index.')
    p.add_argument('--reset_index', action = 'store_true', help = 'Force rebuild the index even if persist_dir exists.')

    # Model + embedding
    p.add_argument('--model', default = _env('RAG_LLM_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct'), help = 'HF model name for LLM.')
    p.add_argument('--embed_model', default = _env('RAG_EMBED_MODEL', 'sentence-transformers/all-mpnet-base-v2'),
                   help = 'HF embedding model for retrieval.')
    p.add_argument('--hf_token', default = os.environ.get('HUGGINGFACE_TOKEN'), help = 'Hugging Face token if needed.')

    # RAG settings
    p.add_argument('--chunk_size', type = int, default = int(_env('RAG_CHUNK_SIZE', '1024')), help = 'Chunk size for ingestion.')
    p.add_argument('--chunk_overlap', type = int, default = int(_env('RAG_CHUNK_OVERLAP', '200')), help = 'Chunk overlap.')
    p.add_argument('--top_k', type = int, default = int(_env('RAG_TOP_K', '5')), help = 'Top-K retrieved nodes.')
    p.add_argument('--temperature', type = float, default = float(_env('RAG_TEMPERATURE', '0.7')), help = 'LLM temperature.')
    p.add_argument('--max_new_tokens', type = int, default = int(_env('RAG_MAX_NEW_TOKENS', '512')),
                   help = 'Max new tokens for generation.')

    # Behavior
    p.add_argument('--show_sources', action = 'store_true', help = 'Append source page numbers to answers.')
    p.add_argument('--verbose', action = 'store_true', help = 'Verbose LlamaIndex inner logs.')
    p.add_argument('-v', '--verbosity', action = 'count', default = 1,
                   help = 'Increase log verbosity (-v=INFO, -vv=DEBUG, default=INFO).')

    # Interaction modes
    p.add_argument('--query', default = None, help = 'Ask a single question and print only the answer.')
    p.add_argument('--json', dest = 'json_out', action = 'store_true', help = 'Output JSON in single-shot mode.')
    p.add_argument('--quiet', action = 'store_true',
                   help = 'Suppress banners and info in single-shot mode (prints only the answer).')

    return p.parse_args(argv);

def build_config_from_args(args: argparse.Namespace) -> RAGConfig:
    '''Build a RAGConfig instance from parsed command-line arguments.'''
    cfg = RAGConfig(
        model_name       = args.model,
        embed_model_name = args.embed_model,
        hf_token         = args.hf_token,
        chunk_size       = args.chunk_size,
        chunk_overlap    = args.chunk_overlap,
        top_k            = args.top_k,
        temperature      = args.temperature,
        max_new_tokens   = args.max_new_tokens,
        persist_dir      = args.persist_dir if args.persist_dir else '',
        reset_index      = args.reset_index,
        show_sources     = args.show_sources,
        verbose          = args.verbose
    )

    return cfg;
