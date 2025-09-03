'''
A complete chatbot with Retrieval-Augmented Generation (RAG)
for PDF documents, built on top of LlamaIndex and Hugging Face models.
'''

import sys
import os

# Disable ChromaDB telemetry before any imports
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from PDF_GPT.chatbot import PDFChatbot
from PDF_GPT.config import RAGConfig

def _setup_pdf_path():
    pdf_source_dir = os.path.join(os.path.dirname(__file__), 'PDF_SOURCE')
    pdf_files      = [f for f in os.listdir(pdf_source_dir) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print('No PDF files found in PDF_SOURCE directory!')
        return None;

    if len(pdf_files) > 1:
        print(f'Multiple PDF files found: {pdf_files}. Using the first one: {pdf_files[0]}')

    return os.path.join(pdf_source_dir, pdf_files[0]);

def main():
    pdf_path = _setup_pdf_path()
    if not pdf_path:
        return 1;

    cfg = RAGConfig()

    print('Starting up...\n')
    print(f'Model:      {cfg.model_name}'      )
    print(f'Embeddings: {cfg.embed_model_name}')

    try: # Initialize chatbot
        chatbot = PDFChatbot([pdf_path], cfg)
    except Exception as e:
        print(f'Failed to initialize chatbot...\nError: {e}')

        return 1;

    if not chatbot.is_initialized:
        print('Failed to initialize chatbot...')

        return 1;

    # Main Chat loop
    print('=' * 60)
    print('PDF RAG Chatbot')
    print('=' * 60)
    print()

    try:
        while True:
            user_in = input('- You: ').strip()
            if not user_in:
                break;
            if user_in.lower() in ('exit', 'quit'):
                break;

            print('- Chat: ', end = '', flush = True)
            out = chatbot.chat(user_in)
            print(out)

    except KeyboardInterrupt:
        pass;
    except Exception as e:
        print('An unexpected error occurred. Exiting...')

        return 1;

    print('\n\nGoodbye!')

    return 0;

if __name__ == '__main__':
    sys.exit(main())
