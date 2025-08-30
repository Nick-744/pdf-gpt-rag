"""
PDF RAG Chatbot System
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
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# ---------------------------
# Dependency checks (fail fast)
# ---------------------------
_MISSING: List[str] = []
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM  # noqa: F401
except Exception:
    _MISSING.append("transformers")

try:
    # Core LlamaIndex
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        Settings,
        StorageContext,
        load_index_from_storage,
    )
    # Embeddings + LLMs
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceLLM
except Exception:
    _MISSING.append("llama-index (+ subpackages)")

try:
    from huggingface_hub import login as hf_login  # noqa: F401
except Exception:
    _MISSING.append("huggingface_hub")

if _MISSING:
    missing_list = ", ".join(_MISSING)
    msg = (
        f"Missing required dependencies: {missing_list}\n"
        "Please install the requirements and try again.\n"
        "Example:\n"
        "  pip install transformers torch llama-index "
        "llama-index-llms-huggingface llama-index-embeddings-huggingface "
        "sentence-transformers huggingface_hub"
    )
    raise RuntimeError(msg)

# ---------------------------
# Logging setup
# ---------------------------
LOG = logging.getLogger("pdf_rag_chatbot")


def configure_logging(verbosity: int) -> None:
    """
    Configure logging level:
        0 -> WARNING
        1 -> INFO
        >=2 -> DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------
# Configuration
# ---------------------------
def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class RAGConfig:
    # Models
    model_name: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    embed_model_name: str = field(default_factory=lambda: _env("RAG_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"))
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HUGGINGFACE_TOKEN"))

    # Chunking + Retrieval
    chunk_size: int = field(default_factory=lambda: int(_env("RAG_CHUNK_SIZE", "1024")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "200")))
    top_k: int = field(default_factory=lambda: int(_env("RAG_TOP_K", "5")))

    # Generation
    temperature: float = field(default_factory=lambda: float(_env("RAG_TEMPERATURE", "0.7")))
    max_new_tokens: int = field(default_factory=lambda: int(_env("RAG_MAX_NEW_TOKENS", "512")))

    # Index persistence
    persist_dir: Optional[str] = field(default_factory=lambda: _env("RAG_PERSIST_DIR", ""))  # empty -> no persistence
    reset_index: bool = False  # force rebuild even if persist_dir exists

    # Behavior flags
    show_sources: bool = False
    verbose: bool = False  # controls LlamaIndex internal verbosity


# ---------------------------
# Core Chatbot
# ---------------------------
class PDFChatbot:
    """
    Orchestrates the RAG pipeline over one or more PDFs.

    - Builds or loads a persistent index (optional).
    - Creates a chat engine with configured LLM + embeddings.
    - Answers questions with optional source-page reporting.
    """

    def __init__(self, pdf_paths: List[str], config: RAGConfig):
        if not pdf_paths:
            raise ValueError("At least one PDF path must be provided.")

        self.pdf_paths = pdf_paths
        self.config = config

        self.index: Optional[VectorStoreIndex] = None
        self.chat_engine = None
        self.is_initialized: bool = False

        self._initialize()

    # -----------------------
    # Initialization helpers
    # -----------------------
    def _login_hf_if_needed(self) -> None:
        if self.config.hf_token:
            try:
                hf_login(token=self.config.hf_token)
                LOG.info("Authenticated with Hugging Face Hub.")
            except Exception as e:
                LOG.warning("Failed to authenticate with Hugging Face Hub: %s", e)
        else:
            LOG.info("No HUGGINGFACE_TOKEN provided. If using gated/private models, set it via env or CLI.")

    def _validate_inputs(self) -> None:
        for p in self.pdf_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Input file not found: {p}")
            if not os.path.isfile(p):
                raise ValueError(f"Expected a file but got a directory or special path: {p}")
            if not p.lower().endswith(".pdf"):
                LOG.warning("Non-PDF file provided (will still attempt to ingest): %s", p)

        if self.config.persist_dir:
            os.makedirs(self.config.persist_dir, exist_ok=True)

    def _load_or_build_index(self) -> VectorStoreIndex:
        """
        If persist_dir exists and reset_index=False, load the index; otherwise build and persist (if requested).
        """
        persist_dir = self.config.persist_dir

        if persist_dir and os.path.isdir(persist_dir) and not self.config.reset_index:
            try:
                LOG.info("Loading index from: %s", persist_dir)
                storage_ctx = StorageContext.from_defaults(persist_dir=persist_dir)
                return load_index_from_storage(storage_ctx)
            except Exception as e:
                LOG.warning("Failed to load index from %s: %s. Rebuilding…", persist_dir, e)

        # Build new index
        LOG.info("Building new index from PDFs...")
        documents = SimpleDirectoryReader(input_files=self.pdf_paths).load_data()
        LOG.info("Loaded %d documents/chunks.", len(documents))

        index = VectorStoreIndex.from_documents(documents)

        if persist_dir:
            try:
                index.storage_context.persist(persist_dir=persist_dir)
                LOG.info("Index persisted to: %s", persist_dir)
            except Exception as e:
                LOG.warning("Failed to persist index to %s: %s", persist_dir, e)

        return index

    def _configure_llamaindex_settings(self) -> None:
        """
        Configure global LlamaIndex Settings for embeddings, LLM, and chunking.
        """
        LOG.debug("Configuring LlamaIndex settings...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.config.embed_model_name)

        # Note: HuggingFaceLLM auto-handles causal vs. seq2seq for many models.
        Settings.llm = HuggingFaceLLM(
            model_name=self.config.model_name,
            tokenizer_name=self.config.model_name,
            context_window=4096,  # adjust as needed per model
            max_new_tokens=self.config.max_new_tokens,
            generate_kwargs={
                "temperature": self.config.temperature,
                "do_sample": True,
            },
        )
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap

    def _initialize(self) -> None:
        try:
            self._validate_inputs()
            self._login_hf_if_needed()
            self._configure_llamaindex_settings()

            # Build/load index and chat engine
            self.index = self._load_or_build_index()
            self.chat_engine = self.index.as_chat_engine(
                similarity_top_k=self.config.top_k,
                verbose=self.config.verbose,
            )
            self.is_initialized = True
            LOG.info("Chatbot initialized successfully.")
        except Exception as e:
            LOG.exception("Initialization failed: %s", e)
            self.is_initialized = False

    # -----------------------
    # Public API
    # -----------------------
    def chat(self, user_input: str) -> str:
        """
        Ask a question about the document(s) and return the answer.
        If show_sources=True, includes source page labels when available.
        """
        if not self.is_initialized or not self.chat_engine:
            return "Sorry, the chatbot is not initialized. Check inputs and dependencies."

        if not user_input.strip():
            return "Please ask a question about the document."

        try:
            response = self.chat_engine.chat(user_input)
            text = str(response)

            if self.config.show_sources:
                pages = self._extract_page_labels(response)
                if pages:
                    text += f"\n\n(Sources: pages {', '.join(sorted(pages, key=_safe_int))})"
            return text
        except Exception as e:
            LOG.exception("Error during chat: %s", e)
            return "I encountered an error while processing your question."

    def reset_conversation(self) -> None:
        if self.chat_engine:
            try:
                self.chat_engine.reset()
                LOG.info("Conversation history cleared.")
            except Exception as e:
                LOG.warning("Failed to reset conversation: %s", e)

    def get_document_info(self) -> Dict[str, Any]:
        if not self.is_initialized or not self.index:
            return {"error": "Index not loaded"}
        try:
            # Safer doc count (avoids poking at private internals)
            # Some versions expose: index.docstore.docs (dict-like)
            num_nodes = None
            try:
                # Best-effort across versions:
                ds = getattr(self.index, "docstore", None)
                if ds is not None:
                    docs = getattr(ds, "docs", None)
                    if isinstance(docs, dict):
                        num_nodes = sum(len(v.nodes) if hasattr(v, "nodes") else 1 for v in docs.values())
            except Exception:
                num_nodes = None

            return {
                "num_nodes": num_nodes,
                "model_name": self.config.model_name,
                "embed_model": self.config.embed_model_name,
                "top_k": self.config.top_k,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "persist_dir": self.config.persist_dir or "",
            }
        except Exception as e:
            LOG.warning("Failed to extract document info: %s", e)
            return {"error": "Failed to query document info."}

    # -----------------------
    # Internals
    # -----------------------
    @staticmethod
    def _extract_page_labels(response: Any) -> List[str]:
        """
        Attempt to extract page labels from response.source_nodes metadata.
        This is best-effort and tolerant across LlamaIndex versions.
        """
        pages: List[str] = []
        try:
            src_nodes = getattr(response, "source_nodes", None)
            if not src_nodes:
                return pages
            for node_with_score in src_nodes:
                node = getattr(node_with_score, "node", None)
                meta = getattr(node, "metadata", {}) if node is not None else {}
                label = meta.get("page_label") or meta.get("page") or meta.get("page_number")
                if label is None:
                    # fallback to 1-based page if available
                    # or any other id-like field
                    label = str(meta.get("id") or meta.get("document_id") or "?")
                pages.append(str(label))
        except Exception:
            pass
        return list(dict.fromkeys(pages))  # unique, preserve order


def _safe_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 10**9  # non-numeric pages sort last


# ---------------------------
# Programmatic API
# ---------------------------
class ChatbotAPI:
    """A small API wrapper suitable for integration/tests."""

    def __init__(self, pdf_path: str | List[str], config: Optional[RAGConfig] = None):
        paths = [pdf_path] if isinstance(pdf_path, str) else list(pdf_path)
        self.config = config or RAGConfig()
        self.chatbot = PDFChatbot(paths, self.config)

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.chatbot.is_initialized:
            return {"success": False, "message": "Chatbot not initialized", "response": ""}
        try:
            resp = self.chatbot.chat(question)
            return {"success": True, "message": "OK", "response": resp}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}", "response": ""}

    def reset(self) -> Dict[str, Any]:
        self.chatbot.reset_conversation()
        return {"success": True, "message": "Conversation reset"}

    def get_info(self) -> Dict[str, Any]:
        return self.chatbot.get_document_info()


# ---------------------------
# CLI
# ---------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PDF RAG Chatbot (LlamaIndex + HuggingFace)")

    p.add_argument(
        "--pdf_path",
        nargs="+",
        required=False,
        default=[_env("RAG_DEFAULT_PDF", "C:\\Users\\nick1\\Documents\\GitHub\\pdf-gpt-rag\\PDF_SOURCE\\Understanding_Climate_Change.pdf")],
        help="Path(s) to PDF file(s). Provide one or more paths.",
    )
    p.add_argument("--persist_dir", default=_env("RAG_PERSIST_DIR", ""), help="Directory to persist/load the index.")
    p.add_argument("--reset_index", action="store_true", help="Force rebuild the index even if persist_dir exists.")

    # Model + embedding
    p.add_argument("--model", default=_env("RAG_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), help="HF model name for LLM.")
    p.add_argument("--embed_model", default=_env("RAG_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"),
                   help="HF embedding model for retrieval.")
    p.add_argument("--hf_token", default=os.environ.get("HUGGINGFACE_TOKEN"), help="Hugging Face token if needed.")

    # RAG settings
    p.add_argument("--chunk_size", type=int, default=int(_env("RAG_CHUNK_SIZE", "1024")), help="Chunk size for ingestion.")
    p.add_argument("--chunk_overlap", type=int, default=int(_env("RAG_CHUNK_OVERLAP", "200")), help="Chunk overlap.")
    p.add_argument("--top_k", type=int, default=int(_env("RAG_TOP_K", "5")), help="Top-K retrieved nodes.")
    p.add_argument("--temperature", type=float, default=float(_env("RAG_TEMPERATURE", "0.7")), help="LLM temperature.")
    p.add_argument("--max_new_tokens", type=int, default=int(_env("RAG_MAX_NEW_TOKENS", "512")),
                   help="Max new tokens for generation.")

    # Behavior
    p.add_argument("--show_sources", action="store_true", help="Append source page numbers to answers.")
    p.add_argument("--verbose", action="store_true", help="Verbose LlamaIndex inner logs.")
    p.add_argument("-v", "--verbosity", action="count", default=1,
                   help="Increase log verbosity (-v=INFO, -vv=DEBUG, default=INFO).")

    # Interaction modes
    p.add_argument("--query", default=None, help="Ask a single question and print only the answer.")
    p.add_argument("--json", dest="json_out", action="store_true", help="Output JSON in single-shot mode.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress banners and info in single-shot mode (prints only the answer).")

    return p.parse_args(argv)


def build_config_from_args(args: argparse.Namespace) -> RAGConfig:
    cfg = RAGConfig(
        model_name=args.model,
        embed_model_name=args.embed_model,
        hf_token=args.hf_token,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        persist_dir=args.persist_dir if args.persist_dir else "",
        reset_index=args.reset_index,
        show_sources=args.show_sources,
        verbose=args.verbose,
    )
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbosity)

    # Build configuration
    cfg = build_config_from_args(args)

    # Banner (skipped in quiet single-shot mode)
    if not (args.query and args.quiet):
        LOG.info("PDF RAG Chatbot — starting up")
        LOG.info("Model: %s | Embeddings: %s | top_k=%d | temp=%.2f | max_new_tokens=%d",
                 cfg.model_name, cfg.embed_model_name, cfg.top_k, cfg.temperature, cfg.max_new_tokens)
        if cfg.persist_dir:
            LOG.info("Persistence directory: %s (reset=%s)", cfg.persist_dir, cfg.reset_index)

    # Initialize chatbot
    try:
        chatbot = PDFChatbot(args.pdf_path, cfg)
    except Exception as e:
        LOG.exception("Failed to initialize the chatbot: %s", e)
        if args.query and args.json_out:
            print(json.dumps({"success": False, "message": f"init_error: {e}", "response": ""}, ensure_ascii=False))
        else:
            print("Failed to initialize chatbot. See logs for details.")
        return 1

    if not chatbot.is_initialized:
        if args.query and args.json_out:
            print(json.dumps({"success": False, "message": "Chatbot not initialized", "response": ""}, ensure_ascii=False))
        else:
            print("Failed to initialize chatbot. Please check paths and dependencies.")
        return 1

    # Single-shot mode: just answer once and exit
    if args.query is not None:
        answer = chatbot.chat(args.query)
        if args.json_out:
            print(json.dumps({"success": True, "message": "OK", "response": answer}, ensure_ascii=False))
        else:
            print(answer)
        return 0

    # Interactive mode
    if not args.quiet:
        print("=" * 60)
        print("PDF RAG Chatbot (Interactive)")
        print("=" * 60)
        info = chatbot.get_document_info()
        if "error" not in info:
            print(f"Document loaded. Model: {info.get('model_name')} | Embeddings: {info.get('embed_model')}")
            nn = info.get("num_nodes")
            if nn is not None:
                print(f"Indexed nodes: {nn}")
        print("Type 'quit' to exit, 'reset' to clear conversation history.")
        print("-" * 60)

    try:
        while True:
            user_in = input("\nYou: ").strip()
            if not user_in:
                continue
            if user_in.lower() in {"q", "quit", "exit"}:
                print("Goodbye!")
                break
            if user_in.lower() == "reset":
                chatbot.reset_conversation()
                print("Conversation reset.")
                continue

            print("Assistant: ", end="", flush=True)
            out = chatbot.chat(user_in)
            print(out)

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        LOG.exception("Fatal error in interactive loop: %s", e)
        print("\nAn unexpected error occurred. Exiting.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
