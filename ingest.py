"""
PDF Vectorisation Pipeline - Ingestion Module
Uses LangChain and LangGraph to ingest, split, embed, and persist PDF documents.
Flow: load -> split -> embed -> store
"""

import os
import sys
import argparse
from typing import TypedDict, List, Any
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END


# State definition for LangGraph pipeline
class IngestionState(TypedDict):
    file_path: str
    chunk_size: int
    chunk_overlap: int
    persist_directory: str
    collection_name: str
    embedding_model_name: str
    documents: List[Document]
    splits: List[Document]
    embeddings: Any
    vector_store: Any


# Node 1: load
def load_node(state: IngestionState) -> dict:
    file_path = state["file_path"]
    print(f"\n[Node: load] Loading document from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"[Node: load] Successfully loaded {len(docs)} pages.")
    return {"documents": docs}


# Node 2: split
def split_node(state: IngestionState) -> dict:
    docs = state["documents"]
    chunk_size = state.get("chunk_size", 1000)
    chunk_overlap = state.get("chunk_overlap", 150)

    print(f"\n[Node: split] Splitting {len(docs)} pages (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    raw_splits = text_splitter.split_documents(docs)

    # Ensure page number metadata is preserved and normalized (1-indexed)
    splits = []
    for idx, doc in enumerate(raw_splits):
        # PyPDFLoader assigns 0-indexed 'page' in metadata
        page_idx = doc.metadata.get("page", 0)
        doc.metadata["page_number"] = page_idx + 1
        doc.metadata["chunk_index"] = idx
        splits.append(doc)

    print(f"[Node: split] Created {len(splits)} text chunks with preserved page metadata.")
    return {"splits": splits}


# Node 3: embed
def embed_node(state: IngestionState) -> dict:
    model_name = state.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"\n[Node: embed] Initializing embedding model: {model_name}...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"[Node: embed] Embedding model ready.")
    return {"embeddings": embeddings}


# Node 4: store
def store_node(state: IngestionState) -> dict:
    splits = state["splits"]
    embeddings = state["embeddings"]
    persist_dir = state.get("persist_directory", "./vectorstore")
    collection_name = state.get("collection_name", "pdf_documents")

    print(f"\n[Node: store] Persisting {len(splits)} chunks to Chroma at '{persist_dir}' (collection: '{collection_name}')...")
    
    os.makedirs(persist_dir, exist_ok=True)

    # Create deterministic IDs for idempotency: {filename}_page_{page}_chunk_{index}
    filename = Path(state["file_path"]).stem
    ids = [
        f"{filename}_p{doc.metadata.get('page_number', 1)}_c{idx}"
        for idx, doc in enumerate(splits)
    ]

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    # Add documents with unique IDs to ensure idempotent ingestion
    vector_store.add_documents(documents=splits, ids=ids)
    print(f"[Node: store] Ingestion complete. Vector store persisted at: {persist_dir}")
    return {"vector_store": vector_store}


def build_ingestion_graph():
    """Builds and compiles the LangGraph ingestion workflow."""
    workflow = StateGraph(IngestionState)

    # Add the four distinct nodes
    workflow.add_node("load", load_node)
    workflow.add_node("split", split_node)
    workflow.add_node("embed", embed_node)
    workflow.add_node("store", store_node)

    # Define linear execution flow
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "split")
    workflow.add_edge("split", "embed")
    workflow.add_edge("embed", "store")
    workflow.add_edge("store", END)

    return workflow.compile()


def run_ingestion(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 150,
                  persist_dir: str = "./vectorstore", collection_name: str = "pdf_documents",
                  embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print("=" * 60)
    print("LangGraph ingestion flow:")
    print("load -> split -> embed -> store")
    print("=" * 60)

    app = build_ingestion_graph()

    initial_state: IngestionState = {
        "file_path": file_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "persist_directory": persist_dir,
        "collection_name": collection_name,
        "embedding_model_name": embedding_model,
        "documents": [],
        "splits": [],
        "embeddings": None,
        "vector_store": None
    }

    result = app.invoke(initial_state)
    print("\nIngestion successfully completed!")
    print(f"Total chunks stored: {len(result.get('splits', []))}")
    print("=" * 60)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF document into a Chroma vector store using LangChain & LangGraph.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file to ingest")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for splitting (default: 150)")
    parser.add_argument("--persist-dir", type=str, default="./vectorstore", help="Persistent storage directory (default: ./vectorstore)")
    parser.add_argument("--collection-name", type=str, default="pdf_documents", help="Chroma collection name (default: pdf_documents)")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace embedding model")

    args = parser.parse_args()

    run_ingestion(
        file_path=args.file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model=args.model_name
    )
