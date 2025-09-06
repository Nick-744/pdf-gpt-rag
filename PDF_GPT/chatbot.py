from .dependencies import (
    torch,
    SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings,
    HuggingFaceEmbedding, HuggingFaceLLM,
    ChromaVectorStore, chromadb
)
from .config import RAGConfig
from typing import List, Any
import os

class PDFChatbot:
    def __init__(self, pdf_path: str, config: RAGConfig):
        self.pdf_path       = pdf_path
        self.config         = config
        self.index          = None
        self.chat_engine    = None

        try:
            self._initialize()
            self.is_initialized = True
        except Exception as e:
            print(f'Initialization failed: {e}')
            self.is_initialized = False
        
        return;

    def _initialize(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize embedding model
        embed_kwargs         = {'device': device}
        Settings.embed_model = HuggingFaceEmbedding(
            model_name = self.config.embed_model_name, **embed_kwargs
        )

        # Initialize LLM
        llm_kwargs = {
            'device_map':      device,
            'model_name':      self.config.model_name,
            'tokenizer_name':  self.config.model_name,
            'context_window':  self.config.context_window,
            'max_new_tokens':  self.config.max_new_tokens,
            'generate_kwargs': {
                'temperature': self.config.temperature,
                'do_sample':   True
            }
        }
        Settings.llm           = HuggingFaceLLM(**llm_kwargs)
        Settings.chunk_size    = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

        # Setup vector store
        os.environ['ANONYMIZED_TELEMETRY'] = 'False' # Remove unnecessary prints
        os.makedirs(self.config.persist_dir, exist_ok = True)

        # Creates ChromaDB client that saves vectors - Provides CRUD operations for vectors!
        client = chromadb.PersistentClient(path = self.config.persist_dir)

        if self.config.reset_index:
            client.delete_collection(name = self.config.collection_name)
            collection = client.create_collection(name = self.config.collection_name)
        else:
            collection = client.get_collection(name = self.config.collection_name)

        vector_store    = ChromaVectorStore(chroma_collection = collection)
        # Adapter that bridges the gap between LlamaIndex's vector store interface and
        # ChromaDB's native API. By passing the chroma_collection parameter, the
        # vector store is instructed to use an existing ChromaDB
        # collection rather than creating a new one!

        storage_context = StorageContext.from_defaults(vector_store = vector_store)
        # Configuration that tells LlamaIndex:
        # - use ChromaDB for vectors
        # - use memory for everything else!

        if not self.config.reset_index and collection.count() > 0:
            # Load existing index: Connect to pre-processed document chunks stored in ChromaDB
            # (Fast - no PDF processing needed, just connects to existing vectors)
            self.index = VectorStoreIndex(nodes = [], storage_context = storage_context)
        else:
            # Build new index: Process PDF from scratch, create chunks, and store embeddings
            # (Slow - reads PDF, splits into chunks, generates embeddings, stores in ChromaDB)
            documents  = SimpleDirectoryReader(input_files = self.pdf_path).load_data()
            self.index = VectorStoreIndex.from_documents(
                documents, storage_context = storage_context
            )

        # Create chat engine
        self.chat_engine = self.index.as_chat_engine(
            similarity_top_k = self.config.top_k,
            verbose          = self.config.verbose
        )

        return;

    def chat(self, user_input: str) -> str:
        ''' Ask a question and return the answer with optional sources. '''
        if not self.is_initialized:
            return 'Sorry, the chatbot is not initialized.';
        if not user_input.strip():
            return 'Please ask a question about the document.';

        temp: str = ''
        try:
            response = self.chat_engine.chat(user_input)
            text     = str(response)

            if self.config.show_sources:
                pages = self._extract_page_labels(response)
                if pages:
                    text += f"\n\n(Sources: pages {', '.join(pages)})"

            temp = text
        except Exception as e:
            print(f'Error during chat: {e}')
            temp = 'I encountered an error while processing your question.'

        return temp;

    def reset_conversation(self):
        ''' Clear previous context that could interfere with new queries! '''
        if self.chat_engine:
            try:
                self.chat_engine.reset()
            except Exception as e:
                print(f'Failed to reset conversation: {e}')

        return;

    @staticmethod
    def _extract_page_labels(response: Any) -> List[str]:
        ''' Extract page labels from response source nodes. '''
        pages = []
        try:
            src_nodes = getattr(response, 'source_nodes', None)
            if not src_nodes:
                return pages;

            for node_with_score in src_nodes:
                node  = getattr(node_with_score, 'node', None)
                meta  = getattr(node, 'metadata', {}) if node else {}
                label = (meta.get('page_label') or meta.get('page') or 
                        meta.get('page_number') or str(meta.get('id', '?')))
                pages.append(str(label))
        except Exception:
            pass;

        return list(dict.fromkeys(pages));
