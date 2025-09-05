'''PDF chatbot with RAG using LlamaIndex and Chroma.'''

import os
from typing import List, Optional, Dict, Any
from .config import RAGConfig
from .dependencies import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings,
    HuggingFaceEmbedding, HuggingFaceLLM, ChromaVectorStore, chromadb
)

class PDFChatbot:
    def __init__(self, pdf_paths: List[str], config: RAGConfig):
        self.pdf_paths = pdf_paths
        self.config = config
        self.index = None
        self.chat_engine = None
        self.is_initialized = False
        self._initialize()

    def _initialize(self):
        '''Initialize all components.'''
        try:
            # Configure device
            import torch
            device = 'cuda' if self.config.device == 'auto' and torch.cuda.is_available() else self.config.device
            if device.startswith('cuda') and not torch.cuda.is_available():
                device = 'cpu'

            # Configure LlamaIndex settings
            embed_kwargs = {'device': device} if device != 'auto' else {}
            Settings.embed_model = HuggingFaceEmbedding(model_name=self.config.embed_model_name, **embed_kwargs)
            
            llm_kwargs = {
                'model_name': self.config.model_name,
                'tokenizer_name': self.config.model_name,
                'context_window': 4096,
                'max_new_tokens': self.config.max_new_tokens,
                'generate_kwargs': {'temperature': self.config.temperature, 'do_sample': True},
            }
            if device != 'auto':
                llm_kwargs['device_map'] = device
            Settings.llm = HuggingFaceLLM(**llm_kwargs)
            Settings.chunk_size = self.config.chunk_size
            Settings.chunk_overlap = self.config.chunk_overlap

            # Setup vector store
            os.environ['ANONYMIZED_TELEMETRY'] = 'False'
            os.makedirs(self.config.persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.config.persist_dir)
            
            try:
                collection = client.get_collection(name=self.config.collection_name)
                if self.config.reset_index:
                    client.delete_collection(name=self.config.collection_name)
                    collection = client.create_collection(name=self.config.collection_name)
            except:
                collection = client.create_collection(name=self.config.collection_name)
            
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Load or build index
            if not self.config.reset_index and collection.count() > 0:
                self.index = VectorStoreIndex(nodes=[], storage_context=storage_context)
            else:
                documents = SimpleDirectoryReader(input_files=self.pdf_paths).load_data()
                self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

            # Create chat engine
            self.chat_engine = self.index.as_chat_engine(
                similarity_top_k=self.config.top_k,
                verbose=self.config.verbose
            )
            self.is_initialized = True
        except Exception as e:
            print(f'Initialization failed: {e}')
            self.is_initialized = False

    def chat(self, user_input: str) -> str:
        '''Ask a question and return the answer with optional sources.'''
        if not self.is_initialized or not self.chat_engine:
            return 'Sorry, the chatbot is not initialized.'
        if not user_input.strip():
            return 'Please ask a question about the document.'
        
        try:
            response = self.chat_engine.chat(user_input)
            text = str(response)
            
            if self.config.show_sources:
                pages = self._extract_page_labels(response)
                if pages:
                    # Sort pages numerically, handling non-numeric values
                    sorted_pages = sorted(pages, key=lambda x: int(x) if x.isdigit() else 999999)
                    text += f"\n\n(Sources: pages {', '.join(sorted_pages)})"
            return text
        except Exception as e:
            print(f'Error during chat: {e}')
            return 'I encountered an error while processing your question.'

    def reset_conversation(self):
        '''Reset the conversation history.'''
        if self.chat_engine:
            try:
                self.chat_engine.reset()
            except Exception as e:
                print(f'Failed to reset conversation: {e}')

    def get_document_info(self) -> Dict[str, Any]:
        '''Get information about loaded documents and configuration.'''
        if not self.is_initialized or not self.index:
            return {'error': 'Index not loaded'}
        
        try:
            # Get node count from Chroma collection
            num_nodes = None
            try:
                storage_context = getattr(self.index, 'storage_context', None)
                if storage_context:
                    vector_store = getattr(storage_context, 'vector_store', None)
                    if vector_store:
                        collection = getattr(vector_store, 'chroma_collection', None)
                        if collection:
                            num_nodes = collection.count()
            except Exception:
                pass

            return {
                'num_nodes': num_nodes,
                'model_name': self.config.model_name,
                'embed_model': self.config.embed_model_name,
                'vector_store': 'chroma',
                'top_k': self.config.top_k,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'persist_dir': self.config.persist_dir,
            }
        except Exception as e:
            print(f'Failed to extract document info: {e}')
            return {'error': 'Failed to query document info.'}

    @staticmethod
    def _extract_page_labels(response: Any) -> List[str]:
        '''Extract page labels from response source nodes.'''
        pages = []
        try:
            src_nodes = getattr(response, 'source_nodes', None)
            if not src_nodes:
                return pages
            
            for node_with_score in src_nodes:
                node = getattr(node_with_score, 'node', None)
                meta = getattr(node, 'metadata', {}) if node else {}
                label = (meta.get('page_label') or meta.get('page') or 
                        meta.get('page_number') or str(meta.get('id', '?')))
                pages.append(str(label))
        except Exception:
            pass
        
        # Return unique pages preserving order
        return list(dict.fromkeys(pages))
