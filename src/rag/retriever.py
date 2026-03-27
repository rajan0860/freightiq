from __future__ import annotations

from typing import List, Dict, Optional
from langchain_core.documents import Document

from src.rag.vector_store import DisruptionVectorStore

class DisruptionRetriever:
    """
    High-level retrieval interface for the agent to query the vector store.
    Provides structured methods for getting context.
    """
    
    def __init__(self):
        self.vstore = DisruptionVectorStore()
        
    def query(self, query: str, filters: Optional[Dict[str, str]] = None, k: int = 5) -> List[Document]:
        """
        Search for disruption info.
        
        Args:
            query: The text to semantic search (e.g. "strike at port of rotterdam")
            filters: Optional metadata filters (e.g. {"region": "Europe"})
            k: Number of documents to return
        """
        return self.vstore.similarity_search(query, k=k, filter_dict=filters)
        
    def format_docs(self, docs: List[Document]) -> str:
        """
        Convert a list of LangChain documents into a single formatted string 
        suitable for feeding into the LLM context window.
        """
        formatted = []
        for d in docs:
            source = d.metadata.get("source", "Unknown Source")
            date = d.metadata.get("date", "Unknown Date")
            doc_type = d.metadata.get("type", "document")
            
            formatted.append(f"[{doc_type.upper()} | {source} | {date}]\n{d.page_content}")
            
        return "\n\n---\n\n".join(formatted)
