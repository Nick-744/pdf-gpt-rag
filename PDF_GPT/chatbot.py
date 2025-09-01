'''
Core chatbot module
-----------------------------------
Contains the main PDFChatbot class
that orchestrates the RAG pipeline.
'''

import os
from typing import List, Optional, Dict, Any

from .config import RAGConfig
from .dependencies import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,

    HuggingFaceEmbedding,
    HuggingFaceLLM,

    ChromaVectorStore,
    chromadb,

    hf_login
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
        self.pdf_paths = pdf_paths
        self.config    = config

        self.index:          Optional[VectorStoreIndex] = None
        self.chat_engine          = None
        self.is_initialized: bool = False

        self._initialize()

        return;

    # Initialization helpers
    def _login_hf_if_needed(self) -> None:
        '''Login to Hugging Face Hub if token is provided.'''
        if self.config.hf_token:
            try:
                hf_login(token = self.config.hf_token)
                print('Authenticated with Hugging Face Hub.')
            except Exception as e:
                print('Failed to authenticate with Hugging Face Hub: %s', e)
        else:
            print('No HUGGINGFACE_TOKEN provided. If using gated/private models, set it via env or CLI.')

        return;

    def _create_vector_store(self) -> ChromaVectorStore:
        '''
        Create and configure the Chroma vector store.
        Returns the ChromaVectorStore instance.
        '''
        # Disable ChromaDB telemetry to prevent network errors
        os.environ['ANONYMIZED_TELEMETRY'] = 'False'
        
        # Set up Chroma persistence directory
        chroma_dir = self.config.persist_dir
        os.makedirs(chroma_dir, exist_ok=True)
        
        # Create Chroma client
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        
        # Create or get collection
        collection_name = self.config.collection_name
        try:
            # Try to get existing collection
            chroma_collection = chroma_client.get_collection(name=collection_name)
            if self.config.reset_index:
                # Delete and recreate if reset is requested
                chroma_client.delete_collection(name=collection_name)
                chroma_collection = chroma_client.create_collection(name=collection_name)
                print('Reset Chroma collection: %s', collection_name)
            else:
                print('Using existing Chroma collection: %s', collection_name)
        except Exception:
            # Collection doesn't exist, create it
            chroma_collection = chroma_client.create_collection(name=collection_name)
            print('Created new Chroma collection: %s', collection_name)

        # Create ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print('Chroma vector store initialized at: %s', chroma_dir)

        return vector_store;

    def _load_or_build_index(self) -> VectorStoreIndex:
        '''
        Load existing index from Chroma or build a new one.
        Uses only Chroma vector store backend.
        '''
        vector_store = self._create_vector_store()

        # Try to load existing index from Chroma
        if not self.config.reset_index:
            try:
                # Try to create index from existing Chroma collection
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                # Check if the collection has any data
                collection = getattr(vector_store, 'chroma_collection', None)
                if collection is not None and collection.count() > 0:
                    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
                    print('Loaded existing Chroma index with %d vectors', collection.count())
                    return index;
                else:
                    print('Chroma collection is empty, building new index...')
            except Exception as e:
                print('Failed to load from Chroma: %s. Building new index...', e)

        # Build new index with Chroma
        print('Building new index with Chroma vector store...')
        documents = SimpleDirectoryReader(input_files = self.pdf_paths).load_data()
        print('Loaded %d documents/chunks.', len(documents))
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print('Index built and persisted in Chroma')

        return index;

    def _configure_llamaindex_settings(self) -> None:
        '''
        Configure global LlamaIndex Settings for embeddings, LLM, and chunking.
        '''
        print('Configuring LlamaIndex settings...')
        
        # Determine device
        device = self._get_device()
        print('Using device: %s', device)
        
        # Configure embedding model with device
        embed_kwargs = {}
        if device and device != 'auto':
            embed_kwargs['device'] = device
            
        Settings.embed_model = HuggingFaceEmbedding(
            model_name = self.config.embed_model_name,
            **embed_kwargs
        )

        # Configure LLM with device
        llm_kwargs = {
            'model_name': self.config.model_name,
            'tokenizer_name': self.config.model_name,
            'context_window': 4096,  # Adjust as needed per model
            'max_new_tokens': self.config.max_new_tokens,
            'generate_kwargs': {
                'temperature': self.config.temperature,
                'do_sample': True,
            },
        }
        
        if device and device != 'auto':
            llm_kwargs['device_map'] = device
            
        Settings.llm = HuggingFaceLLM(**llm_kwargs)
        
        Settings.chunk_size    = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

        return;

    def _get_device(self) -> str:
        '''Determine the best device to use for model execution.'''
        import torch
        
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print('CUDA detected: %d GPU(s) available', torch.cuda.device_count())
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print('GPU %d: %s (%.1f GB)', i, gpu_name, gpu_memory)
            else:
                device = 'cpu'
                print('CUDA not available, using CPU')
        else:
            device = self.config.device
            if device.startswith('cuda') and not torch.cuda.is_available():
                print('CUDA device requested but not available, falling back to CPU')
                device = 'cpu'
                
        return device;

    def _initialize(self) -> None:
        '''Initialize the chatbot by setting up all components.'''
        try:
            self._login_hf_if_needed()
            self._configure_llamaindex_settings()

            # Build/load index and chat engine
            self.index       = self._load_or_build_index()
            self.chat_engine = self.index.as_chat_engine(
                similarity_top_k = self.config.top_k,

                verbose          = self.config.verbose
                # verbose -> True: "condensed question" is LlamaIndex's smart
                # preprocessing of your input to improve the RAG performance!
            )
            self.is_initialized = True
            print('Chatbot initialized successfully.')
        except Exception as e:
            print('Initialization failed: %s', e)
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
            print('Error during chat: %s', e)

            return 'I encountered an error while processing your question.';

    def reset_conversation(self) -> None:
        '''Reset the conversation history.'''
        if self.chat_engine:
            try:
                self.chat_engine.reset()
                print('Conversation history cleared.')
            except Exception as e:
                print('Failed to reset conversation: %s', e)

        return;

    def get_document_info(self) -> Dict[str, Any]:
        '''Get information about the loaded documents and configuration.'''
        if not self.is_initialized or not self.index:
            return {'error': 'Index not loaded'};
        try:
            # Get node count from Chroma collection
            num_nodes = None
            try:
                # For Chroma, get count from collection
                storage_context = getattr(self.index, 'storage_context', None)
                if storage_context:
                    vector_store = getattr(storage_context, 'vector_store', None)
                    if vector_store:
                        collection = getattr(vector_store, 'chroma_collection', None)
                        if collection:
                            num_nodes = collection.count()
            except Exception as e:
                print('Could not determine node count: %s', e)
                num_nodes = None

            return {
                'num_nodes':     num_nodes,
                'model_name':    self.config.model_name,
                'embed_model':   self.config.embed_model_name,
                'vector_store':  'chroma',
                'top_k':         self.config.top_k,
                'chunk_size':    self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'persist_dir':   self.config.persist_dir,
            };
        except Exception as e:
            print('Failed to extract document info: %s', e)

            return {'error': 'Failed to query document info.'};

    # Internals
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
