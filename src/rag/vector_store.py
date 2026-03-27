from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag.embeddings import get_embedding_function

class DisruptionVectorStore:
    """
    Manages the ChromaDB instance for storing and retrieving supply chain 
    disruption events and news articles.
    """
    
    def __init__(self):
        # Default to local persistent storage
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "supply_chain_docs")
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_function = get_embedding_function()
        
        # LangChain wrapper around Chroma
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """
        Convert raw dicts into LangChain Documents and upsert them to ChromaDB.
        Expected dict format: {"id": str, "description": str, "metadata_key": "val"...}
        """
        lc_docs = []
        for doc in documents:
            # Separate the main text block from the metadata
            text = f"{doc.get('title', '')}\n\n{doc.get('description', '')}"
            
            # Keep all other keys as metadata
            metadata = {k: str(v) for k, v in doc.items() if k not in ["title", "description", "content"]}
            
            lc_docs.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                    id=doc.get("id")
                )
            )
            
        if lc_docs:
            self.vectorstore.add_documents(lc_docs)
            print(f"Upserted {len(lc_docs)} documents into '{self.collection_name}' collection.")
            
    def similarity_search(self, query: str, k: int = 4, filter_dict: Optional[Dict[str, str]] = None) -> List[Document]:
        """
        Perform a similarity search against the vector store.
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
