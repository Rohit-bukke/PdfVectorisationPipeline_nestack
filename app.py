"""
FastAPI Retrieval Service for PDF Vectorisation Pipeline.
Directly queries the persistent Chroma vector store using similarity_search_with_score().
NO LLMs, NO RetrievalQA chains, NO generated answers.
"""

import os
# Restrict math/tensor thread pools to 1 to stay within Render 512MB RAM limit
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
torch.set_num_threads(1)

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Configuration
PERSIST_DIRECTORY = os.getenv("VECTORSTORE_DIR", "./vectorstore")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Global state for cached vector store and embeddings
resources = {}


def get_vector_store():
    """Initializes and returns the persistent Chroma vector store instance."""
    if "vector_store" not in resources:
        torch.set_num_threads(1)
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        resources["embeddings"] = embeddings
        resources["vector_store"] = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        gc.collect()
    return resources["vector_store"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fast startup lifespan: uvicorn binds and opens the port immediately.
    Prevents Render 'Port scan timeout reached'. Vector store is loaded lazily on demand.
    """
    port = os.getenv("PORT", "8000")
    print(f"Uvicorn server ready and listening on port {port}. Port scan check will pass immediately.")
    yield
    resources.clear()
    gc.collect()


app = FastAPI(
    title="PDF Vectorisation Pipeline API",
    description="Vector retrieval service using LangChain Chroma vector store and HuggingFace embeddings",
    version="1.0.0",
    lifespan=lifespan
)


# Request and Response schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query text (non-empty)", example="What are the key concepts?")
    top_k: Optional[int] = Field(default=3, description="Number of top results to retrieve (1-20)", example=3)

    @field_validator("query")
    @classmethod
    def validate_query_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query string cannot be empty or only whitespace.")
        return trimmed

    @field_validator("top_k")
    @classmethod
    def validate_top_k_range(cls, v: Optional[int]) -> int:
        if v is None:
            return 3
        if v <= 0:
            raise ValueError("top_k must be a positive integer greater than 0.")
        if v > 20:
            raise ValueError("top_k must not exceed 20.")
        return v


class ChunkResult(BaseModel):
    chunk_text: str
    page_number: int
    score: float


@app.get("/", tags=["Health"])
async def root():
    """Root health and status endpoint."""
    return {
        "status": "online",
        "service": "PDF Vectorisation Retrieval API",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection": COLLECTION_NAME,
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post(
    "/query",
    response_model=List[ChunkResult],
    status_code=status.HTTP_200_OK,
    summary="Query Vector Store",
    description="Directly executes similarity_search_with_score() against the persistent Chroma store and returns raw chunk matches."
)
async def query_vectorstore(request: QueryRequest) -> List[ChunkResult]:
    """
    Executes a similarity search directly against Chroma vector store.
    No LLMs, no conversational chains, no QA wrappers.
    Returns:
        List of objects containing:
        - chunk_text (str): Content of the chunk
        - page_number (int): Original PDF page number (1-indexed)
        - score (float): Similarity distance score from vector store
    """
    try:
        vector_store = get_vector_store()
        
        # Directly call similarity_search_with_score on vector store
        raw_results = vector_store.similarity_search_with_score(
            query=request.query,
            k=request.top_k
        )

        response: List[ChunkResult] = []
        for doc, score in raw_results:
            # Retrieve page_number (preserve 1-indexed page)
            page_num = doc.metadata.get("page_number")
            if page_num is None:
                page_num = doc.metadata.get("page", 0) + 1

            response.append(
                ChunkResult(
                    chunk_text=doc.page_content,
                    page_number=int(page_num),
                    score=round(float(score), 4)
                )
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing vector similarity search: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port)
