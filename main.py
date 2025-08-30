'''
PDF RAG Chatbot System - Main Entry Point
-------------------------------------------------
A complete chatbot with Retrieval-Augmented Generation (RAG) for PDF documents,
built on top of LlamaIndex and Hugging Face models.

Minimal requirements (pin loosely to avoid frequent API breaks):
    transformers>=4.30.0
    torch>=2.0.0
    llama-index>=0.10.0
    llama-index-llms-huggingface>=0.1.0
    llama-index-embeddings-huggingface>=0.1.0
    sentence-transformers>=2.2.0
    huggingface_hub>=0.20.0
'''

from __future__ import annotations

import sys
import json
import logging
import os
from typing import List, Optional

# Disable ChromaDB telemetry before any imports
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from PDF_GPT.config import RAGConfig
from PDF_GPT.chatbot import PDFChatbot, configure_logging
from PDF_GPT.cli import parse_args, build_config_from_args

# ---------------------------
# Logging setup
# ---------------------------
LOG = logging.getLogger('pdf_rag_chatbot')

def main(argv: Optional[List[str]] = None) -> int:
    '''
    Main entry point for the PDF RAG chatbot.
    
    Args:
        argv: Optional command-line arguments (for testing)
        
    Returns:
        Exit code (0 for success, 1 for error)
    '''
    args = parse_args(argv)
    configure_logging(args.verbosity)

    # Build configuration
    cfg = build_config_from_args(args)

    # Banner (skipped in quiet single-shot mode)
    if not (args.query and args.quiet):
        LOG.info('PDF RAG Chatbot — starting up')
        LOG.info('Model: %s | Embeddings: %s | Vector Store: %s | top_k=%d | temp=%.2f | max_new_tokens=%d',
                 cfg.model_name, cfg.embed_model_name, cfg.vector_store_type, cfg.top_k, cfg.temperature, cfg.max_new_tokens)
        if cfg.persist_dir:
            LOG.info('Persistence directory: %s (reset=%s)', cfg.persist_dir, cfg.reset_index)

    # Initialize chatbot
    try:
        chatbot = PDFChatbot(args.pdf_path, cfg)
    except Exception as e:
        LOG.exception('Failed to initialize the chatbot: %s', e)
        if args.query and args.json_out:
            print(json.dumps({'success': False, 'message': f'init_error: {e}', 'response': ''}, ensure_ascii = False))
        else:
            print('Failed to initialize chatbot. See logs for details.')
        
        return 1;

    if not chatbot.is_initialized:
        if args.query and args.json_out:
            print(json.dumps({'success': False, 'message': 'Chatbot not initialized', 'response': ''}, ensure_ascii = False))
        else:
            print('Failed to initialize chatbot. Please check paths and dependencies.')

        return 1;

    # Single-shot mode: just answer once and exit
    if args.query is not None:
        answer = chatbot.chat(args.query)
        if args.json_out:
            print(json.dumps({'success': True, 'message': 'OK', 'response': answer}, ensure_ascii = False))
        else:
            print(answer)

        return 0;

    # Interactive mode
    if not args.quiet:
        print('=' * 60)
        print('PDF RAG Chatbot (Interactive)')
        print('=' * 60)
        info = chatbot.get_document_info()
        if 'error' not in info:
            print(f"Document loaded. Model: {info.get('model_name')} | Embeddings: {info.get('embed_model')} | Vector Store: {info.get('vector_store')}")
            nn = info.get('num_nodes')
            if nn is not None:
                print(f'Indexed nodes: {nn}')
        print("Type 'quit' to exit, 'reset' to clear conversation history.")
        print('-' * 60)

    try:
        while True:
            user_in = input('\nYou: ').strip()
            if not user_in:
                continue;
            if user_in.lower() in {'q', 'quit', 'exit'}:
                print('Goodbye!')

                break;
            if user_in.lower() == 'reset':
                chatbot.reset_conversation()
                print('Conversation reset.')

                continue;

            print('Assistant: ', end = '', flush = True)
            out = chatbot.chat(user_in)
            print(out)

    except KeyboardInterrupt:
        print('\n\nGoodbye!')
    except Exception as e:
        LOG.exception('Fatal error in interactive loop: %s', e)
        print('\nAn unexpected error occurred. Exiting.')

        return 1;

    return 0;

if __name__ == '__main__':
    sys.exit(main())
