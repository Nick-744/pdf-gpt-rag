'''
Core chatbot module for PDF RAG Chatbot System
-------------------------------------------------
Contains the main PDFChatbot class that orchestrates the RAG pipeline.
'''

import os
import logging
from typing import List, Optional, Dict, Any

from .config import RAGConfig
from .dependencies import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    HuggingFaceEmbedding,
    HuggingFaceLLM,
    hf_login
)

# ---------------------------
# Logging setup
# ---------------------------
LOG = logging.getLogger('pdf_rag_chatbot')

def configure_logging(verbosity: int) -> None:
    '''
    Configure logging level:
        0   -> WARNING
        1   -> INFO
        >=2 -> DEBUG
    '''
    level = logging.WARNING
    if verbosity   == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level   = level,
        format  = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt = '%H:%M:%S',
    )

def _safe_int(x: str) -> int:
    '''Helper function to safely convert strings to integers for sorting.'''
    try:
        return int(x);
    except Exception:
        # non-numeric pages sort last
        return 10**9;

# ---------------------------
# Core Chatbot
# ---------------------------
class PDFChatbot:
    '''
    Orchestrates the RAG pipeline over one or more PDFs.

    - Builds or loads a persistent index (optional).
    - Creates a chat engine with configured LLM + embeddings.
    - Answers questions with optional source-page reporting.
    '''

    def __init__(self, pdf_paths: List[str], config: RAGConfig):
        if not pdf_paths:
            raise ValueError('At least one PDF path must be provided.');

        self.pdf_paths = pdf_paths
        self.config    = config

        self.index:          Optional[VectorStoreIndex] = None
        self.chat_engine          = None
        self.is_initialized: bool = False

        self._initialize()

        return;

    # -----------------------
    # Initialization helpers
    # -----------------------
    def _login_hf_if_needed(self) -> None:
        '''Login to Hugging Face Hub if token is provided.'''
        if self.config.hf_token:
            try:
                hf_login(token=self.config.hf_token)
                LOG.info('Authenticated with Hugging Face Hub.')
            except Exception as e:
                LOG.warning('Failed to authenticate with Hugging Face Hub: %s', e)
        else:
            LOG.info('No HUGGINGFACE_TOKEN provided. If using gated/private models, set it via env or CLI.')

        return;

    def _validate_inputs(self) -> None:
        '''Validate that all input PDF paths exist and are valid files.'''
        for p in self.pdf_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f'Input file not found: {p}');
            if not os.path.isfile(p):
                raise ValueError(f'Expected a file but got a directory or special path: {p}');
            if not p.lower().endswith('.pdf'):
                LOG.warning('Non-PDF file provided (will still attempt to ingest): %s', p)

        if self.config.persist_dir:
            os.makedirs(self.config.persist_dir, exist_ok = True)

        return;

    def _load_or_build_index(self) -> VectorStoreIndex:
        '''
        If persist_dir exists and reset_index=False, load the index; otherwise build and persist (if requested).
        '''
        persist_dir = self.config.persist_dir

        if persist_dir and os.path.isdir(persist_dir) and not self.config.reset_index:
            try:
                LOG.info('Loading index from: %s', persist_dir)
                storage_ctx = StorageContext.from_defaults(persist_dir = persist_dir)

                return load_index_from_storage(storage_ctx);
            except Exception as e:
                LOG.warning('Failed to load index from %s: %s. Rebuilding…', persist_dir, e)

        # Build new index
        LOG.info('Building new index from PDFs...')
        documents = SimpleDirectoryReader(input_files = self.pdf_paths).load_data()
        LOG.info('Loaded %d documents/chunks.', len(documents))

        index = VectorStoreIndex.from_documents(documents)

        if persist_dir:
            try:
                index.storage_context.persist(persist_dir = persist_dir)
                LOG.info('Index persisted to: %s', persist_dir)
            except Exception as e:
                LOG.warning('Failed to persist index to %s: %s', persist_dir, e)

        return index;

    def _configure_llamaindex_settings(self) -> None:
        '''
        Configure global LlamaIndex Settings for embeddings, LLM, and chunking.
        '''
        LOG.debug('Configuring LlamaIndex settings...')
        Settings.embed_model = HuggingFaceEmbedding(model_name = self.config.embed_model_name)

        # Note: HuggingFaceLLM auto-handles causal vs. seq2seq for many models.
        Settings.llm = HuggingFaceLLM(
            model_name      = self.config.model_name,
            tokenizer_name  = self.config.model_name,
            context_window  = 4096, # Adjust as needed per model
            max_new_tokens  = self.config.max_new_tokens,
            generate_kwargs = {
                'temperature': self.config.temperature,
                'do_sample':   True,
            },
        )
        Settings.chunk_size    = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

        return;

    def _initialize(self) -> None:
        '''Initialize the chatbot by setting up all components.'''
        try:
            self._validate_inputs()
            self._login_hf_if_needed()
            self._configure_llamaindex_settings()

            # Build/load index and chat engine
            self.index       = self._load_or_build_index()
            self.chat_engine = self.index.as_chat_engine(
                similarity_top_k = self.config.top_k,
                verbose          = self.config.verbose,
            )
            self.is_initialized = True
            LOG.info('Chatbot initialized successfully.')
        except Exception as e:
            LOG.exception('Initialization failed: %s', e)
            self.is_initialized = False

        return;

    # -----------------------
    # Public API
    # -----------------------
    def chat(self, user_input: str) -> str:
        '''
        Ask a question about the document(s) and return the answer.
        If show_sources=True, includes source page labels when available.
        '''
        if not self.is_initialized or not self.chat_engine:
            return 'Sorry, the chatbot is not initialized. Check inputs and dependencies.';

        if not user_input.strip():
            return 'Please ask a question about the document.';

        try:
            response = self.chat_engine.chat(user_input)
            text     = str(response)

            if self.config.show_sources:
                pages = self._extract_page_labels(response)
                if pages:
                    text += f"\n\n(Sources: pages {', '.join(sorted(pages, key = _safe_int))})"
            
            return text;
        except Exception as e:
            LOG.exception('Error during chat: %s', e)

            return 'I encountered an error while processing your question.';

    def reset_conversation(self) -> None:
        '''Reset the conversation history.'''
        if self.chat_engine:
            try:
                self.chat_engine.reset()
                LOG.info('Conversation history cleared.')
            except Exception as e:
                LOG.warning('Failed to reset conversation: %s', e)

        return;

    def get_document_info(self) -> Dict[str, Any]:
        '''Get information about the loaded documents and configuration.'''
        if not self.is_initialized or not self.index:
            return {'error': 'Index not loaded'};
        try:
            # Safer doc count (avoids poking at private internals)
            # Some versions expose: index.docstore.docs (dict-like)
            num_nodes = None
            try:
                # Best-effort across versions:
                ds = getattr(self.index, 'docstore', None)
                if ds is not None:
                    docs = getattr(ds, 'docs', None)
                    if isinstance(docs, dict):
                        num_nodes = sum(len(v.nodes) if hasattr(v, 'nodes') else 1 for v in docs.values())
            except Exception:
                num_nodes = None

            return {
                'num_nodes':     num_nodes,
                'model_name':    self.config.model_name,
                'embed_model':   self.config.embed_model_name,
                'top_k':         self.config.top_k,
                'chunk_size':    self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'persist_dir':   self.config.persist_dir or '',
            };
        except Exception as e:
            LOG.warning('Failed to extract document info: %s', e)

            return {'error': 'Failed to query document info.'};

    # -----------------------
    # Internals
    # -----------------------
    @staticmethod
    def _extract_page_labels(response: Any) -> List[str]:
        '''
        Attempt to extract page labels from response.source_nodes metadata.
        This is best-effort and tolerant across LlamaIndex versions.
        '''
        pages: List[str] = []
        try:
            src_nodes = getattr(response, 'source_nodes', None)
            if not src_nodes:
                return pages;
            for node_with_score in src_nodes:
                node  = getattr(node_with_score, 'node', None)
                meta  = getattr(node, 'metadata', {}) if node is not None else {}
                label = meta.get('page_label') or meta.get('page') or meta.get('page_number')
                if label is None:
                    # fallback to 1-based page if available
                    # or any other id-like field
                    label = str(meta.get('id') or meta.get('document_id') or '?')
                pages.append(str(label))
        except Exception:
            pass;
        
        # Unique, preserve order
        return list(dict.fromkeys(pages));
