"""
PDF ingestion and chunking for the FreightIQ RAG pipeline.

Loads PDF documents (e.g. supply chain reports, disruption bulletins),
splits them into chunks, and returns structured dicts ready for the
vector store.

Usage:
    from src.ingestion.pdf_loader import PDFLoader

    loader = PDFLoader()
    docs = loader.load_and_chunk("path/to/report.pdf")
"""

from __future__ import annotations

import os
from typing import List

import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFLoader:
    """Load and chunk PDF files for RAG ingestion."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_and_chunk(self, file_path: str) -> List[dict]:
        """
        Load a PDF, extract text page-by-page, chunk it, and return
        a list of dicts compatible with ``DisruptionVectorStore.upsert_documents``.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        pages_text = self._extract_text(file_path)
        full_text = "\n\n".join(pages_text)

        chunks = self.splitter.split_text(full_text)
        basename = os.path.basename(file_path)

        return [
            {
                "id": f"{basename}-chunk-{i}",
                "title": f"{basename} (chunk {i + 1}/{len(chunks)})",
                "description": chunk,
                "source": basename,
                "type": "pdf_document",
            }
            for i, chunk in enumerate(chunks)
        ]

    def load_directory(self, dir_path: str) -> List[dict]:
        """Load and chunk all PDF files in a directory."""
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        all_docs = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.lower().endswith(".pdf"):
                pdf_path = os.path.join(dir_path, fname)
                try:
                    docs = self.load_and_chunk(pdf_path)
                    all_docs.extend(docs)
                    print(f"  📄 Loaded {fname} → {len(docs)} chunks")
                except Exception as e:
                    print(f"  ⚠️  Failed to load {fname}: {e}")

        return all_docs

    @staticmethod
    def _extract_text(file_path: str) -> List[str]:
        """Extract text from each page of a PDF using pdfplumber."""
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        return pages
