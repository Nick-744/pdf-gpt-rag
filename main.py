'''
PDF RAG Chatbot System - Main Entry Point
-------------------------------------------------
A complete chatbot with Retrieval-Augmented Generation (RAG) for PDF documents,
built on top of LlamaIndex and Hugging Face models.
'''

import os
import sys
import logging

# Disable ChromaDB telemetry before any imports
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from PDF_GPT.config import RAGConfig
from PDF_GPT.chatbot import PDFChatbot, configure_logging

# ---------------------------
# Logging setup
# ---------------------------
LOG = logging.getLogger('pdf_rag_chatbot')

def main():
    configure_logging(1) # INFO level logging
    
    # Use default configuration and PDF path
    pdf_path = r'C:\Users\nick1\Documents\GitHub\pdf-gpt-rag\PDF_SOURCE\Understanding_Climate_Change.pdf'
    cfg      = RAGConfig()
    
    LOG.info('PDF RAG Chatbot — starting up')
    LOG.info('Model: %s | Embeddings: %s', cfg.model_name, cfg.embed_model_name)
    
    # Initialize chatbot
    try:
        chatbot = PDFChatbot([pdf_path], cfg)
    except Exception as e:
        LOG.exception('Failed to initialize the chatbot: %s', e)
        print('Failed to initialize chatbot. See logs for details.')

        return 1;

    if not chatbot.is_initialized:
        print('Failed to initialize chatbot. Please check paths and dependencies.')

        return 1;

    # Interactive mode
    print('=' * 60)
    print('PDF RAG Chatbot (Interactive)')
    print('=' * 60)
    info = chatbot.get_document_info()
    if 'error' not in info:
        print(f"Document loaded. Model: {info.get('model_name')} | Embeddings: {info.get('embed_model')}")
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
